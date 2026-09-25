from collections import deque

import numpy as np


class AdaptiveHMM:

    def __init__(
        self,
        n_states=2,
        learning_rate=0.01,
        adapt_interval=300,
        positive_state="low",
        min_variance=1e-5,
        em_iterations=20,
    ):
        if positive_state not in ("low", "high"):
            raise ValueError("positive_state phải là 'low' hoặc 'high'")

        if n_states != 2 or not 0 < learning_rate <= 1:
            raise ValueError("HMM requires two states and 0 < learning_rate <= 1")
        if adapt_interval < 2 or min_variance <= 0 or em_iterations < 1:
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

        self.buffer.append(float(value))
        if len(self.buffer) < self.adapt_interval:
            return False

        x = np.asarray(self.buffer, dtype=float)
        gamma, xi = self._forward_backward(x)
        if np.min(gamma.sum(axis=0)) < 2:
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

    def _kmeans_1d(self, x, iterations=25):
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
        eps = 1e-3
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


class StateMachine:

    def __init__(self, mode="eye", fps=30, window_size_sec=60,
                 min_duration_ms=None, max_duration_ms=None, max_gap_sec=0.25):
        limits = {"eye": (100, 2000), "mouth": (3500, 7500), "pitch": (800, 3500)}
        if mode not in limits or fps <= 0 or window_size_sec <= 0 or max_gap_sec <= 0:
            raise ValueError("Invalid FSM configuration")
        self.mode, self.fps, self.window = mode, fps, window_size_sec
        lo, hi = limits[mode]
        self.minimum = lo if min_duration_ms is None else min_duration_ms
        self.maximum = hi if max_duration_ms is None else max_duration_ms
        if not 0 <= self.minimum < self.maximum:
            raise ValueError("Invalid event duration limits")
        self.max_gap = max_gap_sec
        self.reset()

    def reset(self):
        self.time = None
        self.start = None
        self.last_duration = 0.0
        self.events = deque()

    def process(self, state, timestamp=None):
        now = (0 if self.time is None else self.time + 1 / self.fps) if timestamp is None else float(timestamp)
        if not np.isfinite(now) or (self.time is not None and now <= self.time):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if state not in (0, 1, None):
            raise ValueError("State must be 0, 1 or None")
        if self.time is not None and now - self.time > self.max_gap:
            self.start = None
        self.time = now
        done = False
        if state is None:
            self.start = None
        elif state == 1 and self.start is None:
            self.start = now
        elif state == 0 and self.start is not None:
            self.last_duration = (now - self.start) * 1000
            done = self.minimum <= self.last_duration <= self.maximum
            if done:
                self.events.append(now)
            self.start = None
        while self.events and self.events[0] <= now - self.window:
            self.events.popleft()
        duration = 0 if self.start is None else (now - self.start) * 1000
        rate = len(self.events) * 60 / self.window
        return {
            "state": state, "signal_missing": state is None,
            "event_done": done, "current_duration_ms": duration,
            "last_event_duration_ms": self.last_duration,
            "rate_per_minute": rate, "prolonged": duration > self.maximum,
        }


class WindowRatio:
    def __init__(self, window_sec=60, max_gap_sec=0.25):
        if window_sec <= 0 or max_gap_sec <= 0:
            raise ValueError("Window and gap must be positive")
        self.window, self.max_gap = window_sec, max_gap_sec
        self.reset()

    def reset(self):
        self.intervals = deque()
        self.previous = None
        self.time = None

    def process(self, active, timestamp):
        if not np.isfinite(timestamp) or (self.time is not None and timestamp <= self.time):
            raise ValueError("Timestamps must be finite and strictly increasing")
        self.time = timestamp
        if self.previous is not None and active is not None:
            start, previous_active = self.previous
            if timestamp - start <= self.max_gap:
                self.intervals.append((start, timestamp, previous_active))
        self.previous = None if active is None else (timestamp, bool(active))
        cutoff = timestamp - self.window
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        valid = total = 0.0
        for start, end, state in self.intervals:
            duration = end - max(start, cutoff)
            valid += duration
            total += duration * state
        return (100 * total / valid if valid else None), valid


class AdaptiveHMM_FSM:

    def __init__(self, mode="eye", fps=30, init_duration_sec=5, window_size_sec=60,
                 min_duration_ms=None, max_duration_ms=None, max_gap_sec=0.25,
                 learning_rate=0.01, adapt_interval=300):
        if mode not in ("eye", "mouth", "pitch") or fps <= 0 or init_duration_sec <= 0:
            raise ValueError("Invalid detector configuration")
        self.mode, self.fps = mode, fps
        self.init_frames = max(4, round(fps * init_duration_sec))
        self.hmm = AdaptiveHMM(positive_state="low" if mode == "eye" else "high",
                               learning_rate=learning_rate, adapt_interval=adapt_interval)
        self.fsm = StateMachine(mode, fps, window_size_sec, min_duration_ms,
                                max_duration_ms, max_gap_sec)
        self.ratio = WindowRatio(window_size_sec, max_gap_sec)
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
        low, high = np.percentile(data, [5, 95])
        separation = {"eye": 0.06, "mouth": 0.15, "pitch": 10}[self.mode]
        self.normal = float(np.percentile(data, 80 if self.mode == "eye" else 20))
        distinct = low < self.normal * 0.65 if self.mode == "eye" else high > self.normal + separation
        if high - low >= separation and distinct:
            self.hmm.fit_initial(data)
        else:
            normal = self.normal
            event = normal * 0.2 if self.mode == "eye" else normal + separation * 2
            self.hmm.means = np.array([normal, event])
            sigma = max(abs(normal - event) / 4, 0.01)
            self.hmm.vars[:] = sigma ** 2
            self.hmm.A = np.array([[0.97, 0.03], [0.1, 0.9]])
            self.hmm.pi = np.array([0.99, 0.01])
            self.hmm.initialized = True
        self.event_reference = float(self.hmm.means[1])
        self.initialized = True
        self.samples.clear()

    def process(self, value, timestamp=None):
        if timestamp is not None:
            timestamp = float(timestamp)
            if not np.isfinite(timestamp) or (self.fsm.time is not None and timestamp <= self.fsm.time):
                raise ValueError("Timestamps must be finite and strictly increasing")
            if self.fsm.time is not None and timestamp - self.fsm.time > self.fsm.max_gap:
                self.hmm.posterior = None
                self.hmm.buffer.clear()
        try:
            value = float(value)
            if not np.isfinite(value):
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
                if self.mode == "eye" and value >= self.normal * 0.75:
                    state = 0
                    posterior = self.hmm.posterior = np.array([0.99, 0.01])
                updated = self.hmm.update_online(value)
        else:
            self.hmm.posterior = None
            self.hmm.buffer.clear()
        out = self.fsm.process(state, timestamp)
        now = self.fsm.time
        threshold = None
        if self.mode == "eye" and self.initialized:
            opened, closed = self.normal, self.event_reference
            threshold = float(opened - 0.8 * (opened - closed))
        if self.mode == "eye":
            active = None if value is None or threshold is None else value < threshold
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
