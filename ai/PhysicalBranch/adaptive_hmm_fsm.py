from collections import deque

import numpy as np

from ai.PhysicalBranch.config import DEFAULT_CONFIG
from ai.PhysicalBranch.temporal_metrics import StateMachine, WindowRatio, positive_seconds
class AdaptiveHMM:
    def __init__(
        self,
        n_states=2,
        learning_rate=None,
        adapt_interval=None,
        positive_state="low",
        min_variance=None,
        em_iterations=None,
        *, config=DEFAULT_CONFIG,
    ):
        self.config = config
        learning_rate = config.learning_rate if learning_rate is None else learning_rate
        adapt_interval = config.adapt_interval if adapt_interval is None else adapt_interval
        min_variance = config.min_variance if min_variance is None else min_variance
        em_iterations = config.em_iterations if em_iterations is None else em_iterations
        if positive_state not in ("low", "high"):
            raise ValueError("positive_state phải là 'low' hoặc 'high'")

        if n_states != 2 or not 0 < learning_rate <= 1:
            raise ValueError("HMM requires two states and 0 < learning_rate <= 1")
        if (type(adapt_interval) is not int or adapt_interval < 2
                or type(em_iterations) is not int or em_iterations < 1
                or not np.isfinite(min_variance) or min_variance <= 0
                or not np.isfinite(config.min_adapt_state_weight) or config.min_adapt_state_weight < 0):
            raise ValueError("Invalid adaptation parameters")
        self.n_states = int(n_states)
        self.learning_rate = float(learning_rate)
        self.adapt_interval = max(1, int(adapt_interval))
        self.positive_state = positive_state
        self.min_variance = float(min_variance)
        self.em_iterations = int(em_iterations)

        self.pi = np.full(self.n_states, 1.0 / self.n_states)
        self.A = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
        self.means = np.zeros(self.n_states)
        self.vars = np.ones(self.n_states)
        self.posterior = None
        self.initialized = False
        self.buffer = deque(maxlen=self.adapt_interval)

    def reset(self):
        self.__init__(
            n_states=self.n_states,
            learning_rate=self.learning_rate,
            adapt_interval=self.adapt_interval,
            positive_state=self.positive_state,
            min_variance=self.min_variance,
            em_iterations=self.em_iterations,
            config=self.config,
        )

    def fit_initial(self, data):
        x = self._as_vector(data)
        labels = self._kmeans_1d(x)
        self._estimate_from_labels(x, labels)

        for _ in range(self.em_iterations):
            gamma, xi = self._forward_backward(x)
            self._m_step(x, gamma, xi, alpha=1.0)

        self._sort_states()
        self.posterior = None
        self.initialized = True
        self.buffer.clear()
        return self

    def predict(self, value):
        if not self.initialized:
            raise RuntimeError("HMM chưa được khởi tạo")

        if not np.isfinite(value):
            raise ValueError("Observation must be finite")
        prior = self.pi if self.posterior is None else self.posterior @ self.A
        log_p = np.log(np.maximum(prior, 1e-300)) + self._log_emission_prob(np.array([value]))[0]
        self.posterior = np.exp(log_p - self._logsumexp(log_p))
        return int(np.argmax(self.posterior)), self.posterior.copy()

    def update_online(self, value):
        if not self.initialized:
            return False

        if not np.isfinite(value):
            raise ValueError("Observation must be finite")
        self.buffer.append(float(value))
        if len(self.buffer) < self.adapt_interval:
            return False

        x = np.asarray(self.buffer, dtype=float)
        gamma, xi = self._forward_backward(x)
        if np.min(gamma.sum(axis=0)) < self.config.min_adapt_state_weight:
            self.buffer.clear()
            return False
        self._m_step(x, gamma, xi, alpha=self.learning_rate)
        self.posterior = gamma[-1] / max(gamma[-1].sum(), 1e-12)
        self._sort_states()
        self.buffer.clear()
        return True

    def _as_vector(self, data):
        x = np.asarray(data, dtype=float).reshape(-1)
        if x.size < self.n_states * 2:
            raise ValueError("Không đủ dữ liệu để khởi tạo HMM")
        if not np.all(np.isfinite(x)):
            raise ValueError("Dữ liệu chứa NaN hoặc inf")
        return x

    def _kmeans_1d(self, x, iterations=None):
        iterations = self.config.kmeans_iterations if iterations is None else iterations
        centers = np.percentile(x, np.linspace(0, 100, self.n_states + 2)[1:-1])
        if np.unique(centers).size < self.n_states:
            centers = np.linspace(x.min(), x.max(), self.n_states)

        labels = np.zeros(x.size, dtype=int)
        for _ in range(iterations):
            labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
            new_centers = centers.copy()
            for k in range(self.n_states):
                if np.any(labels == k):
                    new_centers[k] = x[labels == k].mean()
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        return labels

    def _estimate_from_labels(self, x, labels):
        eps = self.config.transition_smoothing
        counts = np.bincount(labels, minlength=self.n_states).astype(float) + eps
        self.pi = counts / counts.sum()

        trans = np.full((self.n_states, self.n_states), eps)
        for a, b in zip(labels[:-1], labels[1:]):
            trans[a, b] += 1.0
        self.A = trans / trans.sum(axis=1, keepdims=True)

        for k in range(self.n_states):
            part = x[labels == k]
            self.means[k] = part.mean() if part.size else x.mean()
            self.vars[k] = max(float(part.var()) if part.size else float(x.var()), self.min_variance)

    def _log_emission_prob(self, x):
        var = np.maximum(self.vars, self.min_variance)
        z = x[:, None] - self.means[None, :]
        return -0.5 * (np.log(2 * np.pi * var[None, :]) + z * z / var[None, :])

    def _forward_backward(self, x):
        T = len(x)
        log_b = self._log_emission_prob(x)
        log_pi = np.log(np.maximum(self.pi, 1e-300))
        log_a = np.log(np.maximum(self.A, 1e-300))

        log_alpha = np.zeros((T, self.n_states))
        log_beta = np.zeros((T, self.n_states))
        log_alpha[0] = log_pi + log_b[0]

        for t in range(1, T):
            log_alpha[t] = log_b[t] + self._logsumexp(log_alpha[t - 1, :, None] + log_a, axis=0)

        for t in range(T - 2, -1, -1):
            log_beta[t] = self._logsumexp(log_a + log_b[t + 1, None, :] + log_beta[t + 1, None, :], axis=1)

        log_gamma = log_alpha + log_beta
        log_gamma -= self._logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        xi = np.zeros((max(T - 1, 0), self.n_states, self.n_states))
        for t in range(T - 1):
            log_xi = log_alpha[t, :, None] + log_a + log_b[t + 1, None, :] + log_beta[t + 1, None, :]
            xi[t] = np.exp(log_xi - self._logsumexp(log_xi))
        return gamma, xi

    def _m_step(self, x, gamma, xi, alpha):
        eps = 1e-12
        weights = np.maximum(gamma.sum(axis=0), eps)
        means = (gamma * x[:, None]).sum(axis=0) / weights
        vars_ = (gamma * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / weights

        new_pi = gamma[0] / max(gamma[0].sum(), eps)
        counts = xi.sum(axis=0) + eps
        new_A = self.A if len(xi) == 0 else counts / counts.sum(axis=1, keepdims=True)

        self.pi = (1 - alpha) * self.pi + alpha * new_pi
        self.A = (1 - alpha) * self.A + alpha * new_A
        self.A /= np.maximum(self.A.sum(axis=1, keepdims=True), eps)
        self.means = (1 - alpha) * self.means + alpha * means
        self.vars = np.maximum((1 - alpha) * self.vars + alpha * vars_, self.min_variance)

    def _sort_states(self):
        order = np.argsort(self.means)
        if self.positive_state == "low":
            order = order[::-1]
        self.pi = self.pi[order]
        self.A = self.A[order][:, order]
        self.means = self.means[order]
        self.vars = self.vars[order]
        if self.posterior is not None:
            self.posterior = self.posterior[order]

    @staticmethod
    def _logsumexp(v, axis=None):
        m = np.max(v, axis=axis, keepdims=True)
        out = m + np.log(np.maximum(np.sum(np.exp(v - m), axis=axis, keepdims=True), 1e-300))
        return float(np.squeeze(out)) if axis is None else np.squeeze(out, axis=axis)


class AdaptiveHMM_FSM:

    def __init__(self, mode="eye", fps=None, init_duration_sec=None, window_size_sec=None,
                 min_duration_ms=None, max_duration_ms=None, max_gap_sec=None,
                 learning_rate=None, adapt_interval=None, ratio_window_sec=None, *, config=DEFAULT_CONFIG):
        self.config = config
        fps = config.fps if fps is None else fps
        init_duration_sec = config.init_duration_sec if init_duration_sec is None else init_duration_sec
        window_size_sec = config.window_sec if window_size_sec is None else window_size_sec
        max_gap_sec = config.max_gap_sec if max_gap_sec is None else max_gap_sec
        if mode not in ("eye", "mouth", "pitch"):
            raise ValueError("Invalid detector configuration")
        fps = positive_seconds(fps)
        init_duration_sec = positive_seconds(init_duration_sec)
        self.mode, self.fps = mode, fps
        self.init_frames = max(4, round(fps * init_duration_sec))
        self.hmm = AdaptiveHMM(positive_state="low" if mode == "eye" else "high",
                               learning_rate=learning_rate, adapt_interval=adapt_interval, config=config)
        self.fsm = StateMachine(mode, fps, window_size_sec, min_duration_ms,
                                max_duration_ms, max_gap_sec, config=config)
        self.ratio = WindowRatio(window_size_sec if ratio_window_sec is None else ratio_window_sec, max_gap_sec, config=config)
        self.reset()

    def reset(self):
        self.hmm.reset()
        self.fsm.reset()
        self.samples = deque(maxlen=self.init_frames)
        self.ratio.reset()
        self.initialized = False
        self.normal = self.event_reference = None

    def _initialize(self):
        data = np.asarray(self.samples)
        cfg = self.config
        low, high = np.percentile(data, cfg.init_percentiles)
        separation = {"eye": cfg.eye_separation, "mouth": cfg.mouth_separation,
                      "pitch": cfg.pitch_separation}[self.mode]
        percentile = cfg.eye_baseline_percentile if self.mode == "eye" else cfg.other_baseline_percentile
        self.normal = float(np.percentile(data, percentile))
        distinct = low < self.normal * cfg.eye_distinct_ratio if self.mode == "eye" else high > self.normal + separation
        if high - low >= separation and distinct:
            self.hmm.fit_initial(data)
        else:
            normal = self.normal
            event = normal * cfg.eye_closed_ratio if self.mode == "eye" else normal + separation * cfg.event_separation_multiplier
            self.hmm.means = np.array([normal, event])
            sigma = max(abs(normal - event) / cfg.fallback_sigma_divisor, cfg.fallback_min_sigma)
            self.hmm.vars[:] = sigma ** 2
            self.hmm.A = np.array(cfg.fallback_transition, dtype=float)
            self.hmm.pi = np.array(cfg.resting_prior, dtype=float)
            self.hmm.initialized = True
        self.event_reference = float(self.hmm.means[1])
        self.initialized = True
        self.samples.clear()

    def process(self, value, timestamp=None):
        timestamp = (0 if self.fsm.time is None else self.fsm.time + 1 / self.fps) if timestamp is None else float(timestamp)
        if not np.isfinite(timestamp) or (self.fsm.time is not None and timestamp <= self.fsm.time):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if self.fsm.time is not None and timestamp - self.fsm.time > self.fsm.max_gap:
            self.hmm.posterior = None
            self.hmm.buffer.clear()
        try:
            value = float(value)
            if not np.isfinite(value) or (self.mode != "pitch" and value < 0):
                value = None
        except (TypeError, ValueError):
            value = None
        state, posterior, updated = None, None, False
        if value is not None:
            if not self.initialized:
                self.samples.append(value)
                if len(self.samples) == self.init_frames:
                    self._initialize()
            if self.initialized:
                state, posterior = self.hmm.predict(value)
                if self.mode == "eye" and value >= self.normal * self.config.eye_open_ratio:
                    state = 0
                    posterior = self.hmm.posterior = np.array(self.config.resting_prior, dtype=float)
                updated = self.hmm.update_online(value)
        else:
            self.hmm.posterior = None
            self.hmm.buffer.clear()
        out = self.fsm.process(state, timestamp)
        now = self.fsm.time
        threshold = None
        if self.mode == "eye" and self.initialized:
            opened, closed = self.normal, self.event_reference
            threshold = float(opened - self.config.perclos_closed_fraction * (opened - closed))
        if self.mode == "eye":
            active = None if value is None or threshold is None else value <= threshold
        else:
            active = None if state is None else bool(state)
        percentage, valid = self.ratio.process(active, now)
        out.update(mode=self.mode, input_value=value, initialized=self.initialized,
                   signal_missing=value is None, model_updated=updated,
                   posterior=None if posterior is None else posterior.tolist(),
                   p80_threshold=threshold, perclos=percentage if self.mode == "eye" else None,
                   pom=percentage if self.mode == "mouth" else None,
                   observed_seconds=valid)
        return out
