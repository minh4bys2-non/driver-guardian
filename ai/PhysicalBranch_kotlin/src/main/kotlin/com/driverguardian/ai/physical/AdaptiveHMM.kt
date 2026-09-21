package com.driverguardian.ai.physical

import kotlin.math.*

/**
 * Adaptive Hidden Markov Model with Gaussian emissions and online Expectation-Maximization updates.
 *
 * Designed to dynamically adapt to a driver's individual physiological baseline (e.g. eye aperture EAR,
 * mouth opening MAR, head pitch) and detect anomalous events.
 *
 * Corresponds to AdaptiveHMM in ai/PhysicalBranch/adaptive_hmm_fsm.py.
 */
class AdaptiveHMM(
    val nStates: Int = 2,
    val learningRate: Double = 0.01,
    val adaptInterval: Int = 300,
    val positiveState: PositiveState = PositiveState.LOW,
    val minVariance: Double = 1e-5,
    val emIterations: Int = 20
) {
    var pi: DoubleArray = DoubleArray(nStates) { 1.0 / nStates }
    var A: Array<DoubleArray> = Array(nStates) { DoubleArray(nStates) { 1.0 / nStates } }
    var means: DoubleArray = DoubleArray(nStates) { 0.0 }
    var vars: DoubleArray = DoubleArray(nStates) { 1.0 }
    var posterior: DoubleArray? = null
    var initialized: Boolean = false
    val buffer: ArrayDeque<Double> = ArrayDeque(adaptInterval)

    init {
        require(nStates == 2) { "HMM requires exactly two states, got $nStates" }
        require(learningRate in 0.0..1.0 && learningRate > 0.0) { "learning_rate must be in (0, 1], got $learningRate" }
        require(adaptInterval >= 2) { "adapt_interval must be >= 2, got $adaptInterval" }
        require(minVariance > 0.0) { "min_variance must be > 0, got $minVariance" }
        require(emIterations >= 1) { "em_iterations must be >= 1, got $emIterations" }
    }

    constructor(
        nStates: Int = 2,
        learningRate: Double = 0.01,
        adaptInterval: Int = 300,
        positiveState: String,
        minVariance: Double = 1e-5,
        emIterations: Int = 20
    ) : this(
        nStates = nStates,
        learningRate = learningRate,
        adaptInterval = adaptInterval,
        positiveState = PositiveState.fromString(positiveState),
        minVariance = minVariance,
        emIterations = emIterations
    )

    fun reset() {
        pi = DoubleArray(nStates) { 1.0 / nStates }
        A = Array(nStates) { DoubleArray(nStates) { 1.0 / nStates } }
        means = DoubleArray(nStates) { 0.0 }
        vars = DoubleArray(nStates) { 1.0 }
        posterior = null
        initialized = false
        buffer.clear()
    }

    /**
     * Initializes the HMM parameters on a calibration dataset using 1D k-means followed by EM iterations.
     */
    fun fitInitial(data: DoubleArray): AdaptiveHMM {
        val x = asVector(data)
        val labels = kmeans1D(x)
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

    /**
     * Computes the filtered state posterior and returns the most likely state (0 or 1).
     */
    fun predict(value: Double): Pair<Int, DoubleArray> {
        check(initialized) { "HMM chưa được khởi tạo" }
        require(value.isFinite()) { "Observation must be finite, got $value" }

        val prior = if (posterior == null) {
            pi.clone()
        } else {
            DoubleArray(nStates) { j ->
                var sum = 0.0
                for (i in 0 until nStates) {
                    sum += posterior!![i] * A[i][j]
                }
                sum
            }
        }

        val logEmission = logEmissionProb(doubleArrayOf(value))[0]
        val logP = DoubleArray(nStates) { ln(maxOf(prior[it], 1e-300)) + logEmission[it] }
        val lse = MathUtils.logSumExp(logP)
        val post = DoubleArray(nStates) { exp(logP[it] - lse) }

        this.posterior = post.clone()
        val argmax = if (post[0] >= post[1]) 0 else 1
        return Pair(argmax, post)
    }

    /**
     * Buffers incoming observations and runs an online EM adaptation step once the buffer is full.
     */
    fun updateOnline(value: Double): Boolean {
        if (!initialized) return false

        buffer.addLast(value)
        if (buffer.size > adaptInterval) {
            buffer.removeFirst()
        }
        if (buffer.size < adaptInterval) return false

        val x = buffer.toDoubleArray()
        val (gamma, xi) = forwardBackward(x)

        // Validate sufficient state occupancy
        for (k in 0 until nStates) {
            var sumGammaK = 0.0
            for (t in 0 until x.size) sumGammaK += gamma[t][k]
            if (sumGammaK < 2.0) {
                buffer.clear()
                return false
            }
        }

        mStep(x, gamma, xi, alpha = learningRate)

        val lastGamma = gamma[gamma.size - 1]
        val sumLastGamma = maxOf(lastGamma.sum(), 1e-12)
        this.posterior = DoubleArray(nStates) { lastGamma[it] / sumLastGamma }

        sortStates()
        buffer.clear()
        return true
    }

    private fun asVector(data: DoubleArray): DoubleArray {
        require(data.size >= nStates * 2) { "Không đủ dữ liệu để khởi tạo HMM (cần ít nhất ${nStates * 2} mẫu)" }
        for (v in data) {
            require(v.isFinite()) { "Dữ liệu chứa NaN hoặc inf" }
        }
        return data.clone()
    }

    private fun kmeans1D(x: DoubleArray, iterations: Int = 25): IntArray {
        val p1 = MathUtils.percentile(x, 100.0 / 3.0)
        val p2 = MathUtils.percentile(x, 200.0 / 3.0)
        val centers = doubleArrayOf(p1, p2)

        if (abs(centers[0] - centers[1]) < 1e-6) {
            var minV = x[0]
            var maxV = x[0]
            for (v in x) {
                if (v < minV) minV = v
                if (v > maxV) maxV = v
            }
            centers[0] = minV
            centers[1] = maxV
            if (abs(centers[0] - centers[1]) < 1e-6) {
                centers[1] = centers[0] + 1e-4
            }
        }

        val labels = IntArray(x.size)
        for (iter in 0 until iterations) {
            // Assign labels
            for (i in x.indices) {
                val d0 = abs(x[i] - centers[0])
                val d1 = abs(x[i] - centers[1])
                labels[i] = if (d0 <= d1) 0 else 1
            }

            // Update centers
            var sum0 = 0.0; var count0 = 0
            var sum1 = 0.0; var count1 = 0
            for (i in x.indices) {
                if (labels[i] == 0) {
                    sum0 += x[i]
                    count0++
                } else {
                    sum1 += x[i]
                    count1++
                }
            }

            val newCenter0 = if (count0 > 0) sum0 / count0 else centers[0]
            val newCenter1 = if (count1 > 0) sum1 / count1 else centers[1]

            if (abs(centers[0] - newCenter0) < 1e-6 && abs(centers[1] - newCenter1) < 1e-6) {
                break
            }
            centers[0] = newCenter0
            centers[1] = newCenter1
        }
        return labels
    }

    private fun estimateFromLabels(x: DoubleArray, labels: IntArray) {
        val eps = 1e-3
        var count0 = eps
        var count1 = eps
        for (l in labels) {
            if (l == 0) count0 += 1.0 else count1 += 1.0
        }
        val totalCount = count0 + count1
        pi[0] = count0 / totalCount
        pi[1] = count1 / totalCount

        val trans = Array(2) { DoubleArray(2) { eps } }
        for (i in 0 until labels.size - 1) {
            val a = labels[i]
            val b = labels[i + 1]
            trans[a][b] += 1.0
        }
        for (i in 0..1) {
            val rowSum = trans[i][0] + trans[i][1]
            A[i][0] = trans[i][0] / rowSum
            A[i][1] = trans[i][1] / rowSum
        }

        val globalMean = MathUtils.mean(x)
        val globalVar = MathUtils.variance(x, ddof = 0)

        for (k in 0..1) {
            var sumK = 0.0
            var cntK = 0
            for (i in x.indices) {
                if (labels[i] == k) {
                    sumK += x[i]
                    cntK++
                }
            }
            if (cntK > 0) {
                val m = sumK / cntK
                means[k] = m
                var sumSq = 0.0
                for (i in x.indices) {
                    if (labels[i] == k) {
                        val d = x[i] - m
                        sumSq += d * d
                    }
                }
                vars[k] = maxOf(sumSq / cntK, minVariance)
            } else {
                means[k] = globalMean
                vars[k] = maxOf(globalVar, minVariance)
            }
        }
    }

    fun logEmissionProb(x: DoubleArray): Array<DoubleArray> {
        val logB = Array(x.size) { DoubleArray(nStates) }
        val ln2pi = ln(2.0 * Math.PI)
        for (k in 0 until nStates) {
            val v = maxOf(vars[k], minVariance)
            val meanK = means[k]
            val logTerm = ln2pi + ln(v)
            for (t in x.indices) {
                val z = x[t] - meanK
                logB[t][k] = -0.5 * (logTerm + (z * z) / v)
            }
        }
        return logB
    }

    fun forwardBackward(x: DoubleArray): Pair<Array<DoubleArray>, Array<Array<DoubleArray>>> {
        val tLen = x.size
        val logB = logEmissionProb(x)
        val logPi = DoubleArray(nStates) { ln(maxOf(pi[it], 1e-300)) }
        val logA = Array(nStates) { i -> DoubleArray(nStates) { j -> ln(maxOf(A[i][j], 1e-300)) } }

        val logAlpha = Array(tLen) { DoubleArray(nStates) }
        val logBeta = Array(tLen) { DoubleArray(nStates) }

        for (k in 0 until nStates) {
            logAlpha[0][k] = logPi[k] + logB[0][k]
        }

        val tempTransition = DoubleArray(nStates)
        for (t in 1 until tLen) {
            for (j in 0 until nStates) {
                for (i in 0 until nStates) {
                    tempTransition[i] = logAlpha[t - 1][i] + logA[i][j]
                }
                logAlpha[t][j] = logB[t][j] + MathUtils.logSumExp(tempTransition)
            }
        }

        for (k in 0 until nStates) {
            logBeta[tLen - 1][k] = 0.0
        }

        for (t in tLen - 2 downTo 0) {
            for (i in 0 until nStates) {
                for (j in 0 until nStates) {
                    tempTransition[j] = logA[i][j] + logB[t + 1][j] + logBeta[t + 1][j]
                }
                logBeta[t][i] = MathUtils.logSumExp(tempTransition)
            }
        }

        val gamma = Array(tLen) { DoubleArray(nStates) }
        val tempState = DoubleArray(nStates)
        for (t in 0 until tLen) {
            for (k in 0 until nStates) {
                tempState[k] = logAlpha[t][k] + logBeta[t][k]
            }
            val lse = MathUtils.logSumExp(tempState)
            for (k in 0 until nStates) {
                gamma[t][k] = exp(tempState[k] - lse)
            }
        }

        val xiCount = maxOf(tLen - 1, 0)
        val xi = Array(xiCount) { Array(nStates) { DoubleArray(nStates) } }
        val tempXi = DoubleArray(nStates * nStates)

        for (t in 0 until tLen - 1) {
            var idx = 0
            for (i in 0 until nStates) {
                for (j in 0 until nStates) {
                    val logXiVal = logAlpha[t][i] + logA[i][j] + logB[t + 1][j] + logBeta[t + 1][j]
                    xi[t][i][j] = logXiVal
                    tempXi[idx++] = logXiVal
                }
            }
            val lse = MathUtils.logSumExp(tempXi)
            for (i in 0 until nStates) {
                for (j in 0 until nStates) {
                    xi[t][i][j] = exp(xi[t][i][j] - lse)
                }
            }
        }

        return Pair(gamma, xi)
    }

    fun mStep(x: DoubleArray, gamma: Array<DoubleArray>, xi: Array<Array<DoubleArray>>, alpha: Double) {
        val eps = 1e-12

        for (k in 0 until nStates) {
            var sumGammaK = 0.0
            var sumGammaX = 0.0
            for (t in x.indices) {
                val g = gamma[t][k]
                sumGammaK += g
                sumGammaX += g * x[t]
            }
            val weight = maxOf(sumGammaK, eps)
            val meanK = sumGammaX / weight

            var sumGammaDiffSq = 0.0
            for (t in x.indices) {
                val d = x[t] - meanK
                sumGammaDiffSq += gamma[t][k] * d * d
            }
            val varK = sumGammaDiffSq / weight

            means[k] = (1.0 - alpha) * means[k] + alpha * meanK
            vars[k] = maxOf((1.0 - alpha) * vars[k] + alpha * varK, minVariance)
        }

        val sumGamma0 = maxOf(gamma[0].sum(), eps)
        for (k in 0 until nStates) {
            val newPiK = gamma[0][k] / sumGamma0
            pi[k] = (1.0 - alpha) * pi[k] + alpha * newPiK
        }

        if (xi.isNotEmpty()) {
            val counts = Array(nStates) { DoubleArray(nStates) { eps } }
            for (t in xi.indices) {
                for (i in 0 until nStates) {
                    for (j in 0 until nStates) {
                        counts[i][j] += xi[t][i][j]
                    }
                }
            }
            for (i in 0 until nStates) {
                val rowSum = maxOf(counts[i].sum(), eps)
                for (j in 0 until nStates) {
                    val newAIJ = counts[i][j] / rowSum
                    A[i][j] = (1.0 - alpha) * A[i][j] + alpha * newAIJ
                }
            }
        }

        // Renormalize rows of A
        for (i in 0 until nStates) {
            val rowSum = maxOf(A[i].sum(), eps)
            for (j in 0 until nStates) {
                A[i][j] /= rowSum
            }
        }
    }

    private fun sortStates() {
        var shouldSwap = false
        if (positiveState == PositiveState.LOW) {
            // For LOW positiveState (e.g. eye): state 0 = normal (high EAR), state 1 = event (low EAR)
            if (means[0] < means[1]) shouldSwap = true
        } else {
            // For HIGH positiveState (e.g. mouth): state 0 = normal (low MAR), state 1 = event (high MAR)
            if (means[0] > means[1]) shouldSwap = true
        }

        if (shouldSwap) {
            val tmpMean = means[0]; means[0] = means[1]; means[1] = tmpMean
            val tmpVar = vars[0]; vars[0] = vars[1]; vars[1] = tmpVar
            val tmpPi = pi[0]; pi[0] = pi[1]; pi[1] = tmpPi

            val a00 = A[0][0]; val a01 = A[0][1]
            val a10 = A[1][0]; val a11 = A[1][1]
            A[0][0] = a11; A[0][1] = a10
            A[1][0] = a01; A[1][1] = a00

            posterior?.let {
                val tmpP = it[0]; it[0] = it[1]; it[1] = tmpP
            }
        }
    }
}
