package com.example.driverguardian.ai.physical.temporal

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * 2-State Gaussian Adaptive Hidden Markov Model matching Python AdaptiveHMM.
 *
 * Implements Baum-Welch (EM) parameter estimation, forward-backward in log-space,
 * online adaptation with Exponential Moving Average, and state sorting.
 */
class AdaptiveHmm(
    val nStates: Int = 2,
    val learningRate: Double = 0.01,
    val adaptInterval: Int = 300,
    val positiveState: String = "low",
    val minVariance: Double = 1e-5,
    val emIterations: Int = 20
) {
    init {
        require(positiveState == "low" || positiveState == "high") { "positiveState must be 'low' or 'high'" }
        require(nStates == 2 && learningRate > 0.0 && learningRate <= 1.0) { "HMM requires two states and 0 < learningRate <= 1" }
        require(adaptInterval >= 2 && minVariance > 0.0 && emIterations >= 1) { "Invalid adaptation parameters" }
    }

    var pi = DoubleArray(nStates) { 1.0 / nStates }
    var a = Array(nStates) { DoubleArray(nStates) { 1.0 / nStates } }
    var means = DoubleArray(nStates) { 0.0 }
    var vars = DoubleArray(nStates) { 1.0 }
    var posterior: DoubleArray? = null
    var initialized = false
    val buffer = ArrayDeque<Double>(adaptInterval)

    fun reset() {
        pi = DoubleArray(nStates) { 1.0 / nStates }
        a = Array(nStates) { DoubleArray(nStates) { 1.0 / nStates } }
        means = DoubleArray(nStates) { 0.0 }
        vars = DoubleArray(nStates) { 1.0 }
        posterior = null
        initialized = false
        buffer.clear()
    }

    fun fitInitial(data: DoubleArray): AdaptiveHmm {
        val x = asVector(data)
        val labels = kMeans1D(x)
        estimateFromLabels(x, labels)

        for (iter in 0 until emIterations) {
            val (gamma, xi) = forwardBackward(x)
            mStep(x, gamma, xi, alpha = 1.0)
        }

        sortStates()
        posterior = null
        initialized = true
        buffer.clear()
        return this
    }

    fun predict(value: Double): Pair<Int, DoubleArray> {
        check(initialized) { "HMM is not initialized" }
        require(value.isFinite()) { "Observation must be finite" }

        val prior = DoubleArray(nStates)
        if (posterior == null) {
            System.arraycopy(pi, 0, prior, 0, nStates)
        } else {
            val post = posterior!!
            for (j in 0 until nStates) {
                var sum = 0.0
                for (i in 0 until nStates) {
                    sum += post[i] * a[i][j]
                }
                prior[j] = sum
            }
        }

        val logB = logEmissionProbSingle(value)
        val logP = DoubleArray(nStates) { k ->
            ln(max(prior[k], 1e-300)) + logB[k]
        }

        val lse = logSumExp1D(logP)
        val nextPosterior = DoubleArray(nStates) { k ->
            exp(logP[k] - lse)
        }

        this.posterior = nextPosterior
        val state = if (nextPosterior[0] >= nextPosterior[1]) 0 else 1
        return state to nextPosterior.clone()
    }

    fun updateOnline(value: Double): Boolean {
        if (!initialized) return false

        if (buffer.size >= adaptInterval) {
            buffer.removeFirst()
        }
        buffer.addLast(value)

        if (buffer.size < adaptInterval) return false

        val x = buffer.toDoubleArray()
        val (gamma, xi) = forwardBackward(x)

        val gammaSum0 = gamma.sumOf { it[0] }
        val gammaSum1 = gamma.sumOf { it[1] }
        if (gammaSum0 < 2.0 || gammaSum1 < 2.0) {
            buffer.clear()
            return false
        }

        mStep(x, gamma, xi, alpha = learningRate)

        val lastGamma = gamma.last()
        val lastSum = max(lastGamma[0] + lastGamma[1], 1e-12)
        posterior = doubleArrayOf(lastGamma[0] / lastSum, lastGamma[1] / lastSum)

        sortStates()
        buffer.clear()
        return true
    }

    private fun asVector(data: DoubleArray): DoubleArray {
        require(data.size >= nStates * 2) { "Insufficient data to initialize HMM" }
        for (v in data) {
            require(v.isFinite()) { "Data contains NaN or inf" }
        }
        return data.clone()
    }

    private fun kMeans1D(x: DoubleArray, iterations: Int = 25): IntArray {
        val sorted = x.clone().apply { sort() }
        val p33 = percentile(sorted, 100.0 / 3.0)
        val p66 = percentile(sorted, 200.0 / 3.0)

        val centers = if (kotlin.math.abs(p33 - p66) > 1e-6) {
            doubleArrayOf(p33, p66)
        } else {
            doubleArrayOf(sorted.first(), sorted.last())
        }

        val labels = IntArray(x.size)
        for (iter in 0 until iterations) {
            for (i in x.indices) {
                val d0 = kotlin.math.abs(x[i] - centers[0])
                val d1 = kotlin.math.abs(x[i] - centers[1])
                labels[i] = if (d0 <= d1) 0 else 1
            }

            var sum0 = 0.0
            var count0 = 0
            var sum1 = 0.0
            var count1 = 0
            for (i in x.indices) {
                if (labels[i] == 0) {
                    sum0 += x[i]
                    count0++
                } else {
                    sum1 += x[i]
                    count1++
                }
            }

            val newC0 = if (count0 > 0) sum0 / count0 else centers[0]
            val newC1 = if (count1 > 0) sum1 / count1 else centers[1]

            if (kotlin.math.abs(centers[0] - newC0) < 1e-6 && kotlin.math.abs(centers[1] - newC1) < 1e-6) {
                break
            }
            centers[0] = newC0
            centers[1] = newC1
        }
        return labels
    }

    private fun estimateFromLabels(x: DoubleArray, labels: IntArray) {
        val eps = 1e-3
        var c0 = eps
        var c1 = eps
        for (l in labels) {
            if (l == 0) c0 += 1.0 else c1 += 1.0
        }
        val totalCounts = c0 + c1
        pi = doubleArrayOf(c0 / totalCounts, c1 / totalCounts)

        val trans = Array(nStates) { DoubleArray(nStates) { eps } }
        for (i in 0 until labels.size - 1) {
            trans[labels[i]][labels[i + 1]] += 1.0
        }
        val rowSum0 = trans[0][0] + trans[0][1]
        val rowSum1 = trans[1][0] + trans[1][1]
        a = arrayOf(
            doubleArrayOf(trans[0][0] / rowSum0, trans[0][1] / rowSum0),
            doubleArrayOf(trans[1][0] / rowSum1, trans[1][1] / rowSum1)
        )

        val meanAll = x.average()
        var varAll = 0.0
        for (v in x) varAll += (v - meanAll) * (v - meanAll)
        varAll /= x.size

        for (k in 0 until nStates) {
            var sum = 0.0
            var cnt = 0
            for (i in x.indices) {
                if (labels[i] == k) {
                    sum += x[i]
                    cnt++
                }
            }
            val m = if (cnt > 0) sum / cnt else meanAll
            var v = 0.0
            if (cnt > 0) {
                for (i in x.indices) {
                    if (labels[i] == k) {
                        v += (x[i] - m) * (x[i] - m)
                    }
                }
                v /= cnt
            } else {
                v = varAll
            }
            means[k] = m
            vars[k] = max(v, minVariance)
        }
    }

    private fun logEmissionProbSingle(x: Double): DoubleArray {
        return DoubleArray(nStates) { k ->
            val v = max(vars[k], minVariance)
            val z = x - means[k]
            -0.5 * (ln(2.0 * Math.PI * v) + (z * z) / v)
        }
    }

    private fun forwardBackward(x: DoubleArray): Pair<Array<DoubleArray>, Array<Array<DoubleArray>>> {
        val t = x.size
        val logB = Array(t) { i -> logEmissionProbSingle(x[i]) }
        val logPi = DoubleArray(nStates) { k -> ln(max(pi[k], 1e-300)) }
        val logA = Array(nStates) { i -> DoubleArray(nStates) { j -> ln(max(a[i][j], 1e-300)) } }

        val logAlpha = Array(t) { DoubleArray(nStates) }
        val logBeta = Array(t) { DoubleArray(nStates) }

        for (k in 0 until nStates) {
            logAlpha[0][k] = logPi[k] + logB[0][k]
        }

        for (step in 1 until t) {
            for (j in 0 until nStates) {
                val v0 = logAlpha[step - 1][0] + logA[0][j]
                val v1 = logAlpha[step - 1][1] + logA[1][j]
                logAlpha[step][j] = logB[step][j] + logSumExpPair(v0, v1)
            }
        }

        for (step in t - 2 downTo 0) {
            for (i in 0 until nStates) {
                val v0 = logA[i][0] + logB[step + 1][0] + logBeta[step + 1][0]
                val v1 = logA[i][1] + logB[step + 1][1] + logBeta[step + 1][1]
                logBeta[step][i] = logSumExpPair(v0, v1)
            }
        }

        val gamma = Array(t) { step ->
            val lg0 = logAlpha[step][0] + logBeta[step][0]
            val lg1 = logAlpha[step][1] + logBeta[step][1]
            val lse = logSumExpPair(lg0, lg1)
            doubleArrayOf(exp(lg0 - lse), exp(lg1 - lse))
        }

        val xi = Array(max(t - 1, 0)) { step ->
            val arr = Array(nStates) { DoubleArray(nStates) }
            val lxi00 = logAlpha[step][0] + logA[0][0] + logB[step + 1][0] + logBeta[step + 1][0]
            val lxi01 = logAlpha[step][0] + logA[0][1] + logB[step + 1][1] + logBeta[step + 1][1]
            val lxi10 = logAlpha[step][1] + logA[1][0] + logB[step + 1][0] + logBeta[step + 1][0]
            val lxi11 = logAlpha[step][1] + logA[1][1] + logB[step + 1][1] + logBeta[step + 1][1]
            val lse = logSumExp4(lxi00, lxi01, lxi10, lxi11)

            arr[0][0] = exp(lxi00 - lse)
            arr[0][1] = exp(lxi01 - lse)
            arr[1][0] = exp(lxi10 - lse)
            arr[1][1] = exp(lxi11 - lse)
            arr
        }

        return gamma to xi
    }

    private fun mStep(x: DoubleArray, gamma: Array<DoubleArray>, xi: Array<Array<DoubleArray>>, alpha: Double) {
        val eps = 1e-12
        val t = x.size

        var gSum0 = 0.0
        var gSum1 = 0.0
        var gmSum0 = 0.0
        var gmSum1 = 0.0
        for (step in 0 until t) {
            gSum0 += gamma[step][0]
            gSum1 += gamma[step][1]
            gmSum0 += gamma[step][0] * x[step]
            gmSum1 += gamma[step][1] * x[step]
        }
        val w0 = max(gSum0, eps)
        val w1 = max(gSum1, eps)
        val newMeans0 = gmSum0 / w0
        val newMeans1 = gmSum1 / w1

        var gvSum0 = 0.0
        var gvSum1 = 0.0
        for (step in 0 until t) {
            val d0 = x[step] - newMeans0
            val d1 = x[step] - newMeans1
            gvSum0 += gamma[step][0] * d0 * d0
            gvSum1 += gamma[step][1] * d1 * d1
        }
        val newVars0 = gvSum0 / w0
        val newVars1 = gvSum1 / w1

        val g0Sum = max(gamma[0][0] + gamma[0][1], eps)
        val newPi0 = gamma[0][0] / g0Sum
        val newPi1 = gamma[0][1] / g0Sum

        var xiSum00 = eps
        var xiSum01 = eps
        var xiSum10 = eps
        var xiSum11 = eps
        for (step in xi.indices) {
            xiSum00 += xi[step][0][0]
            xiSum01 += xi[step][0][1]
            xiSum10 += xi[step][1][0]
            xiSum11 += xi[step][1][1]
        }
        val xiRow0 = xiSum00 + xiSum01
        val xiRow1 = xiSum10 + xiSum11

        val newA00 = if (xi.isEmpty()) a[0][0] else xiSum00 / xiRow0
        val newA01 = if (xi.isEmpty()) a[0][1] else xiSum01 / xiRow0
        val newA10 = if (xi.isEmpty()) a[1][0] else xiSum10 / xiRow1
        val newA11 = if (xi.isEmpty()) a[1][1] else xiSum11 / xiRow1

        pi[0] = (1.0 - alpha) * pi[0] + alpha * newPi0
        pi[1] = (1.0 - alpha) * pi[1] + alpha * newPi1

        a[0][0] = (1.0 - alpha) * a[0][0] + alpha * newA00
        a[0][1] = (1.0 - alpha) * a[0][1] + alpha * newA01
        a[1][0] = (1.0 - alpha) * a[1][0] + alpha * newA10
        a[1][1] = (1.0 - alpha) * a[1][1] + alpha * newA11

        val aRow0 = max(a[0][0] + a[0][1], eps)
        val aRow1 = max(a[1][0] + a[1][1], eps)
        a[0][0] /= aRow0
        a[0][1] /= aRow0
        a[1][0] /= aRow1
        a[1][1] /= aRow1

        means[0] = (1.0 - alpha) * means[0] + alpha * newMeans0
        means[1] = (1.0 - alpha) * means[1] + alpha * newMeans1

        vars[0] = max((1.0 - alpha) * vars[0] + alpha * newVars0, minVariance)
        vars[1] = max((1.0 - alpha) * vars[1] + alpha * newVars1, minVariance)
    }

    fun sortStates() {
        val swap = if (positiveState == "low") {
            means[0] < means[1]
        } else {
            means[0] > means[1]
        }

        if (swap) {
            val tmpPi = pi[0]
            pi[0] = pi[1]
            pi[1] = tmpPi

            val tmpA00 = a[0][0]
            val tmpA01 = a[0][1]
            val tmpA10 = a[1][0]
            val tmpA11 = a[1][1]
            a[0][0] = tmpA11
            a[0][1] = tmpA10
            a[1][0] = tmpA01
            a[1][1] = tmpA00

            val tmpMean = means[0]
            means[0] = means[1]
            means[1] = tmpMean

            val tmpVar = vars[0]
            vars[0] = vars[1]
            vars[1] = tmpVar

            posterior?.let { post ->
                val tmpPost = post[0]
                post[0] = post[1]
                post[1] = tmpPost
            }
        }
    }

    private fun percentile(sorted: DoubleArray, pct: Double): Double {
        val n = sorted.size
        if (n == 0) return 0.0
        val rank = (pct / 100.0) * (n - 1)
        val lower = rank.toInt()
        val upper = if (lower + 1 < n) lower + 1 else lower
        val weight = rank - lower
        return sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }

    private fun logSumExpPair(a: Double, b: Double): Double {
        val m = max(a, b)
        return m + ln(max(exp(a - m) + exp(b - m), 1e-300))
    }

    private fun logSumExp4(a: Double, b: Double, c: Double, d: Double): Double {
        val m = max(max(a, b), max(c, d))
        return m + ln(max(exp(a - m) + exp(b - m) + exp(c - m) + exp(d - m), 1e-300))
    }

    private fun logSumExp1D(v: DoubleArray): Double {
        val m = v.maxOrNull() ?: 0.0
        var sum = 0.0
        for (x in v) sum += exp(x - m)
        return m + ln(max(sum, 1e-300))
    }
}
