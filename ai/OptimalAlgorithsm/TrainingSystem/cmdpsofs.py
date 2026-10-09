from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ai.OptimalAlgorithsm.TrainingSystem.fitnessFunction import FEATURES, Fitness, load_jsonl, validate_labels


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Minimization Pareto dominance."""
    return bool(np.all(a <= b) and np.any(a < b))


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """NSGA-II style crowding distance."""
    n, m = objectives.shape
    if n == 0:
        return np.empty(0, dtype=np.float32)
    if n <= 2:
        return np.full(n, np.inf, dtype=np.float32)

    distance = np.zeros(n, dtype=np.float64)
    for j in range(m):
        order = np.argsort(objectives[:, j])
        values = objectives[order, j]
        span = values[-1] - values[0]
        if span <= 1e-12:
            continue
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        distance[order[1:-1]] += (values[2:] - values[:-2]) / span
    return distance.astype(np.float32)


@dataclass
class CMDPSOFSConfig:
    population_size: int = 50
    generations: int = 100
    dimensions: int = 9
    inertia: float = 0.729
    cognitive: float = 1.49445
    social: float = 1.49445
    velocity_limit: float = 0.20
    archive_size: int = 100
    nonuniform_b: float = 5.0
    seed: int = 42

    def __post_init__(self):
        sizes = (self.population_size, self.generations, self.archive_size)
        if any(type(v) is not int or v < 1 for v in sizes):
            raise ValueError("population_size, generations and archive_size must be positive integers")
        if type(self.dimensions) is not int or self.dimensions != 9:
            raise ValueError("Fitness requires 8 weights and 1 classification threshold")
        coefficients = (self.inertia, self.cognitive, self.social, self.velocity_limit)
        if not np.isfinite(coefficients).all() or min(coefficients) < 0:
            raise ValueError("PSO coefficients and velocity_limit must be finite and nonnegative")
        if not np.isfinite(self.nonuniform_b) or self.nonuniform_b <= 0:
            raise ValueError("nonuniform_b must be finite and positive")


@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    objectives: np.ndarray
    pbest_position: np.ndarray
    pbest_objectives: np.ndarray


class ParetoArchive:
    def __init__(self, max_size: int):
        if type(max_size) is not int or max_size < 1:
            raise ValueError("Archive size must be a positive integer")
        self.max_size = max_size
        self.positions = np.empty((0, 9), dtype=np.float32)
        self.objectives = np.empty((0, 3), dtype=np.float32)

    def __len__(self):
        return len(self.positions)

    def update(self, positions: np.ndarray, objectives: np.ndarray):
        positions = np.asarray(positions, dtype=np.float32)
        objectives = np.asarray(objectives, dtype=np.float32)
        if (positions.ndim != 2 or positions.shape[1] != 9
                or objectives.shape != (len(positions), 3)
                or not np.isfinite(positions).all() or not np.isfinite(objectives).all()):
            raise ValueError("Archive requires finite positions [N,9] and objectives [N,3]")
        if len(self) == 0:
            all_positions = positions
            all_objectives = objectives
        else:
            all_positions = np.vstack((self.positions, positions))
            all_objectives = np.vstack((self.objectives, objectives))

        _, unique_idx = np.unique(
            np.round(all_positions, decimals=8), axis=0, return_index=True
        )
        all_positions = all_positions[unique_idx]
        all_objectives = all_objectives[unique_idx]

        keep = np.ones(len(all_positions), dtype=bool)
        for i in range(len(all_positions)):
            if not keep[i]:
                continue
            for j in range(len(all_positions)):
                if i == j:
                    continue
                if dominates(all_objectives[j], all_objectives[i]):
                    keep[i] = False
                    break

        self.positions = all_positions[keep]
        self.objectives = all_objectives[keep]
        self._truncate()

    def _truncate(self):
        while len(self) > self.max_size:
            cd = crowding_distance(self.objectives)
            finite = np.where(np.isfinite(cd))[0]
            remove_idx = finite[np.argmin(cd[finite])] if len(finite) else len(self) - 1
            self.positions = np.delete(self.positions, remove_idx, axis=0)
            self.objectives = np.delete(self.objectives, remove_idx, axis=0)

    def select_leader(self, rng: np.random.Generator) -> np.ndarray:
        if len(self) == 0:
            raise RuntimeError("Pareto archive is empty")
        if len(self) == 1:
            return self.positions[0].copy()

        cd = crowding_distance(self.objectives)
        a, b = rng.choice(len(self), size=2, replace=False)
        winner = a if cd[a] >= cd[b] else b
        return self.positions[winner].copy()


class RunLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generation_log = self.output_dir / "generation_log.jsonl"
        self.population_log = self.output_dir / "population_log.jsonl"
        self.final_archive = self.output_dir / "pareto_archive.jsonl"
        self.history = []
        self.generation_log.write_text("", encoding="utf-8")
        self.population_log.write_text("", encoding="utf-8")
        self.final_archive.write_text("", encoding="utf-8")

    def save_config(self, config, data_path, fitness_method, n_samples):
        payload = {
            "algorithm": "CMDPSOFS-style continuous multi-objective PSO",
            "data_path": str(data_path),
            "evaluation_scope": "optimization data; no held-out evaluation",
            "fitness_method": fitness_method,
            "n_samples": n_samples,
            "features": FEATURES,
            "feature_selection_threshold": 0.5,
            "objectives": ["1 - recall", "1 - precision", "num_feature / 8"],
            "config": asdict(config),
        }
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def log_population(self, generation: int, particles: list[Particle]):
        with open(self.population_log, "a", encoding="utf-8") as f:
            for i, p in enumerate(particles):
                mask = p.position[:8] >= 0.5
                row = {
                    "generation": generation,
                    "particle": i,
                    "candidate": p.position.tolist(),
                    "objectives": p.objectives.tolist(),
                    "recall": float(1.0 - p.objectives[0]),
                    "precision": float(1.0 - p.objectives[1]),
                    "num_features": int(mask.sum()),
                    "selected_features": [FEATURES[j] for j in np.flatnonzero(mask)],
                    "classification_threshold": float(p.position[8]),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_generation(self, generation, particles, archive, elapsed_seconds):
        pop_obj = np.vstack([p.objectives for p in particles])
        recall = 1.0 - archive.objectives[:, 0]
        precision = 1.0 - archive.objectives[:, 1]
        num_features = (archive.positions[:, :8] >= 0.5).sum(axis=1)
        ideal_distance = np.linalg.norm(archive.objectives, axis=1)

        row = {
            "generation": generation,
            "archive_size": len(archive),
            "best_recall": float(recall.max()),
            "best_precision": float(precision.max()),
            "min_num_features": int(np.rint(num_features.min())),
            "best_ideal_distance": float(ideal_distance.min()),
            "population_mean_objectives": pop_obj.mean(axis=0).tolist(),
            "elapsed_seconds": float(elapsed_seconds),
        }
        self.history.append(row)
        with open(self.generation_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save_archive(self, archive):
        order = np.argsort(np.linalg.norm(archive.objectives, axis=1))
        with open(self.final_archive, "w", encoding="utf-8") as f:
            for idx in order:
                candidate = archive.positions[idx]
                obj = archive.objectives[idx]
                mask = candidate[:8] >= 0.5
                row = {
                    "candidate": candidate.tolist(),
                    "objectives": obj.tolist(),
                    "recall": float(1.0 - obj[0]),
                    "precision": float(1.0 - obj[1]),
                    "num_features": int(mask.sum()),
                    "num_feature_ratio": float(obj[2]),
                    "selected_features": [FEATURES[j] for j in np.flatnonzero(mask)],
                    "classification_threshold": float(candidate[8]),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save_plots(self, archive):
        if not self.history:
            return
        generations = np.array([x["generation"] for x in self.history])
        self._line_plot(generations, [x["best_recall"] for x in self.history],
                        "Best recall", "Recall", "convergence_recall.png")
        self._line_plot(generations, [x["best_precision"] for x in self.history],
                        "Best precision", "Precision", "convergence_precision.png")
        self._line_plot(generations, [x["min_num_features"] for x in self.history],
                        "Minimum selected features", "Number of features",
                        "convergence_num_features.png")
        self._line_plot(generations, [x["best_ideal_distance"] for x in self.history],
                        "Distance to ideal objective point", "Euclidean distance",
                        "convergence_ideal_distance.png")
        self._pareto_plot(archive)
        self._feature_frequency_plot(archive)

    def _line_plot(self, x, y, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y)
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=160)
        plt.close(fig)

    def _pareto_plot(self, archive):
        recall = 1.0 - archive.objectives[:, 0]
        precision = 1.0 - archive.objectives[:, 1]
        num_features = (archive.positions[:, :8] >= 0.5).sum(axis=1)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(recall, precision, num_features)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_zlabel("Num features")
        ax.set_title("Final Pareto archive")
        fig.tight_layout()
        fig.savefig(self.output_dir / "pareto_final_3d.png", dpi=160)
        plt.close(fig)

    def _feature_frequency_plot(self, archive):
        selected = archive.positions[:, :8] >= 0.5
        frequency = selected.mean(axis=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(np.arange(8), frequency)
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels(FEATURES, rotation=35, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Selection frequency")
        ax.set_title("Feature selection frequency in Pareto archive")
        fig.tight_layout()
        fig.savefig(self.output_dir / "feature_selection_frequency.png", dpi=160)
        plt.close(fig)


class CMDPSOFS:
    """
    Candidate: [w1, ..., w8, T_cls] in [0, 1]^9.
    Feature j is active iff wj >= 0.5.
    Objectives (minimize): [1-Recall, 1-Precision, NumFeature/8].

    Mutation groups:
      group 0: no mutation
      group 1: uniform mutation
      group 2: non-uniform mutation
    """

    def __init__(self, fitness, y_true, method, config, logger):
        self.fitness = fitness
        self.y_true = validate_labels(y_true, len(fitness.X))
        if method not in ("weighted_mean", "noisy_or"):
            raise ValueError(f"Unknown fitness method: {method}")
        self.method = method
        self.cfg = config
        self.logger = logger
        self.rng = np.random.default_rng(config.seed)
        self.archive = ParetoArchive(config.archive_size)
        self.particles = []

    def _evaluate(self, candidate):
        # Invalid: no selected feature. Penalize all objectives so it cannot
        # exploit NumFeature=0 and survive on the Pareto front.
        if not np.any(candidate[:8] >= 0.5):
            return np.ones(3, dtype=np.float32)
        y_pred = self.fitness.predict(candidate, method=self.method)
        return self.fitness.evaluate(candidate, y_pred, self.y_true)

    def _initialize(self):
        cfg = self.cfg
        positions = self.rng.uniform(0.0, 1.0,
                                     size=(cfg.population_size, cfg.dimensions)).astype(np.float32)
        velocities = self.rng.uniform(-cfg.velocity_limit, cfg.velocity_limit,
                                      size=(cfg.population_size, cfg.dimensions)).astype(np.float32)
        self.particles = []
        for position, velocity in zip(positions, velocities):
            objectives = self._evaluate(position)
            self.particles.append(Particle(
                position=position.copy(),
                velocity=velocity.copy(),
                objectives=objectives.copy(),
                pbest_position=position.copy(),
                pbest_objectives=objectives.copy(),
            ))
        self.archive.update(
            np.vstack([p.position for p in self.particles]),
            np.vstack([p.objectives for p in self.particles]),
        )

    def _uniform_mutation(self, position):
        child = position.copy()
        dim = self.rng.integers(self.cfg.dimensions)
        child[dim] = self.rng.random()
        return child

    def _nonuniform_mutation(self, position, generation):
        child = position.copy()
        dim = self.rng.integers(self.cfg.dimensions)
        progress = generation / max(self.cfg.generations - 1, 1)
        exponent = (1.0 - progress) ** self.cfg.nonuniform_b
        r = self.rng.random()
        if self.rng.random() < 0.5:
            distance = 1.0 - child[dim]
            child[dim] += distance * (1.0 - r ** exponent)
        else:
            distance = child[dim]
            child[dim] -= distance * (1.0 - r ** exponent)
        return np.clip(child, 0.0, 1.0)

    def _mutate(self, position, particle_index, generation):
        group = particle_index % 3
        if group == 0:
            return position
        if group == 1:
            return self._uniform_mutation(position)
        return self._nonuniform_mutation(position, generation)

    def _update_pbest(self, particle):
        if dominates(particle.objectives, particle.pbest_objectives):
            particle.pbest_position = particle.position.copy()
            particle.pbest_objectives = particle.objectives.copy()
        elif not dominates(particle.pbest_objectives, particle.objectives):
            if self.rng.random() < 0.5:
                particle.pbest_position = particle.position.copy()
                particle.pbest_objectives = particle.objectives.copy()

    def _step(self, generation):
        cfg = self.cfg
        for i, particle in enumerate(self.particles):
            leader = self.archive.select_leader(self.rng)
            r1 = self.rng.random(cfg.dimensions)
            r2 = self.rng.random(cfg.dimensions)
            particle.velocity = (
                cfg.inertia * particle.velocity
                + cfg.cognitive * r1 * (particle.pbest_position - particle.position)
                + cfg.social * r2 * (leader - particle.position)
            )
            particle.velocity = np.clip(
                particle.velocity, -cfg.velocity_limit, cfg.velocity_limit
            )
            new_position = np.clip(particle.position + particle.velocity, 0.0, 1.0)
            new_position = self._mutate(new_position, i, generation)
            particle.position = new_position.astype(np.float32)
            particle.objectives = self._evaluate(particle.position)
            self._update_pbest(particle)

        self.archive.update(
            np.vstack([p.position for p in self.particles]),
            np.vstack([p.objectives for p in self.particles]),
        )

    def run(self):
        start = time.perf_counter()
        self._initialize()

        for generation in range(self.cfg.generations):
            if generation > 0:
                self._step(generation)

            elapsed = time.perf_counter() - start
            self.logger.log_population(generation, self.particles)
            self.logger.log_generation(
                generation, self.particles, self.archive, elapsed
            )

            last = self.logger.history[-1]
            print(
                f"[{generation + 1:03d}/{self.cfg.generations}] "
                f"archive={last['archive_size']:3d} "
                f"recall={last['best_recall']:.4f} "
                f"precision={last['best_precision']:.4f} "
                f"min_features={last['min_num_features']} "
                f"time={elapsed:.1f}s"
            )

        self.logger.save_archive(self.archive)
        self.logger.save_plots(self.archive)
        return self.archive

if __name__ == "__main__":
    # ============================================================
    # Change experiment parameters directly here.
    # ============================================================
    DATA_PATH = Path(__file__).resolve().parent.parent / "drowsiness_risk.jsonl"

    OUTPUT_DIR = ("/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/OptimalAlgorithsm/TrainingSystem/output/cmdpsofs")

    # Must match a method implemented in Fitness.
    FITNESS_METHOD = "noisy_or"

    CONFIG = CMDPSOFSConfig(
        population_size=50,
        generations=100,
        inertia=0.729,
        cognitive=1.49445,
        social=1.49445,
        velocity_limit=0.20,
        archive_size=100,
        nonuniform_b=5.0,
        seed=42,
    )

    X, y = load_jsonl(DATA_PATH)
    fitness = Fitness(X)
    logger = RunLogger(OUTPUT_DIR)
    logger.save_config(
        CONFIG,
        data_path=DATA_PATH,
        fitness_method=FITNESS_METHOD,
        n_samples=len(y),
    )

    optimizer = CMDPSOFS(
        fitness=fitness,
        y_true=y,
        method=FITNESS_METHOD,
        config=CONFIG,
        logger=logger,
    )

    archive = optimizer.run()
    print(f"\nFinished. Pareto solutions: {len(archive)}")
    print(f"Outputs: {OUTPUT_DIR}")
