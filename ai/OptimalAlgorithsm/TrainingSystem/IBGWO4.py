"""Continuous multi-objective adaptation of IBGWO4 (Algorithm 3).

Reference: https://doi.org/10.3390/app15020489
Retains parent + GWO + GWO-then-PSO selection, replacing scalar ranking
with Pareto fronts and crowding distance. Uses the existing continuous
weights and threshold, not the paper's S/V binary transfer functions.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ai.OptimalAlgorithsm.TrainingSystem.cmdpsofs import (
    ParetoArchive,
    Particle,
    RunLogger,
    crowding_distance,
    dominates,
)
from ai.OptimalAlgorithsm.TrainingSystem.fitnessFunction import FEATURES, Fitness, load_jsonl


@dataclass
class IBGWO4Config:
    population_size: int = 50
    generations: int = 100
    dimensions: int = 9
    inertia: float = 0.729
    cognitive: float = 1.49445
    social: float = 1.49445
    velocity_limit: float = 0.20
    archive_size: int = 100
    seed: int = 42

    def __post_init__(self):
        if self.population_size < 3 or self.generations < 1 or self.archive_size < 1:
            raise ValueError("Require population_size >= 3, generations >= 1, archive_size >= 1")
        if self.dimensions != 9:
            raise ValueError("Fitness requires 8 weights and 1 classification threshold")
        coefficients = (self.inertia, self.cognitive, self.social, self.velocity_limit)
        if not np.all(np.isfinite(coefficients)) or min(coefficients) < 0:
            raise ValueError("PSO coefficients and velocity_limit must be finite and nonnegative")


def pareto_order(objectives):
    """Rank by nondominated front, then descending crowding distance."""
    objectives = np.asarray(objectives)
    domination = (
        np.all(objectives[:, None] <= objectives[None, :], axis=2)
        & np.any(objectives[:, None] < objectives[None, :], axis=2)
    )
    counts = domination.sum(axis=0)
    front = np.flatnonzero(counts == 0)
    order = []
    while len(front):
        distance = crowding_distance(objectives[front])
        order.extend(front[np.argsort(-distance, kind="stable")].tolist())
        counts[front] = -1
        counts -= domination[front].sum(axis=0)
        front = np.flatnonzero(counts == 0)
    return np.asarray(order, dtype=np.intp)


class IBGWO4Logger(RunLogger):
    def save_config(self, config, data_path, fitness_method, n_samples):
        payload = {
            "algorithm": "IBGWO4-style continuous multi-objective GWO-PSO",
            "reference": "https://doi.org/10.3390/app15020489",
            "selection": "parent + GWO + GWO-then-PSO; Pareto rank and crowding distance",
            "representation": "continuous weights and threshold; no binary transfer function",
            "data_path": str(data_path),
            "fitness_method": fitness_method,
            "n_samples": n_samples,
            "features": FEATURES,
            "feature_selection_threshold": 0.5,
            "objectives": ["1 - recall", "1 - precision", "num_feature / 8"],
            "config": asdict(config),
        }
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


class IBGWO4:
    def __init__(self, fitness, y_true, method, config, logger):
        self.fitness = fitness
        self.y_true = np.asarray(y_true, dtype=np.int8)
        self.method = method
        self.cfg = config
        self.logger = logger
        self.rng = np.random.default_rng(config.seed)
        self.archive = ParetoArchive(config.archive_size)
        self.particles = []

    def _evaluate(self, candidate):
        if not np.any(candidate[:8] >= 0.5):
            return np.ones(3, dtype=np.float32)
        prediction = self.fitness.predict(candidate, method=self.method)
        return self.fitness.evaluate(candidate, prediction, self.y_true)

    def _update_archive(self, particles):
        self.archive.update(
            np.vstack([p.position for p in particles]),
            np.vstack([p.objectives for p in particles]),
        )

    def _initialize(self):
        cfg = self.cfg
        positions = self.rng.random((cfg.population_size, cfg.dimensions)).astype(np.float32)
        velocities = self.rng.uniform(
            -cfg.velocity_limit, cfg.velocity_limit, size=positions.shape
        ).astype(np.float32)
        self.particles = []
        for position, velocity in zip(positions, velocities):
            objectives = self._evaluate(position)
            self.particles.append(Particle(
                position, velocity, objectives, position.copy(), objectives.copy()
            ))
        self._update_archive(self.particles)

    def _child(self, parent, position, velocity):
        position = np.clip(position, 0.0, 1.0).astype(np.float32)
        objectives = self._evaluate(position)
        improved = dominates(objectives, parent.pbest_objectives)
        if not improved and not dominates(parent.pbest_objectives, objectives):
            improved = self.rng.random() < 0.5
        return Particle(
            position=position,
            velocity=np.asarray(velocity, dtype=np.float32).copy(),
            objectives=objectives,
            pbest_position=(position if improved else parent.pbest_position).copy(),
            pbest_objectives=(objectives if improved else parent.pbest_objectives).copy(),
        )

    def _step(self, generation):
        cfg = self.cfg
        parents = self.particles
        order = pareto_order(np.vstack([p.objectives for p in parents]))
        leaders = np.vstack([parents[i].position for i in order[:3]])
        a = 2.0 * (1.0 - (generation - 1) / max(cfg.generations - 2, 1))

        gwo_population = []
        for parent in parents:
            A = 2.0 * a * self.rng.random(leaders.shape) - a
            C = 2.0 * self.rng.random(leaders.shape)
            position = (leaders - A * np.abs(C * leaders - parent.position)).mean(axis=0)
            gwo_population.append(self._child(parent, position, parent.velocity))

        self._update_archive(gwo_population)
        pso_population = []
        for wolf in gwo_population:
            leader = self.archive.select_leader(self.rng)
            r1, r2 = self.rng.random((2, cfg.dimensions))
            velocity = (
                cfg.inertia * wolf.velocity
                + cfg.cognitive * r1 * (wolf.pbest_position - wolf.position)
                + cfg.social * r2 * (leader - wolf.position)
            )
            velocity = np.clip(velocity, -cfg.velocity_limit, cfg.velocity_limit)
            pso_population.append(self._child(wolf, wolf.position + velocity, velocity))

        candidates = parents + gwo_population + pso_population
        self._update_archive(candidates)
        order = pareto_order(np.vstack([p.objectives for p in candidates]))
        self.particles = [candidates[i] for i in order[:cfg.population_size]]

    def run(self):
        start = time.perf_counter()
        self._initialize()
        for generation in range(self.cfg.generations):
            if generation > 0:
                self._step(generation)
            self.logger.log_population(generation, self.particles)
            self.logger.log_generation(
                generation, self.particles, self.archive, time.perf_counter() - start
            )
            last = self.logger.history[-1]
            print(
                f"[{generation + 1:03d}/{self.cfg.generations}] "
                f"archive={last['archive_size']:3d} "
                f"recall={last['best_recall']:.4f} "
                f"precision={last['best_precision']:.4f} "
                f"min_features={last['min_num_features']} "
                f"time={last['elapsed_seconds']:.1f}s"
            )
        self.logger.save_archive(self.archive)
        self.logger.save_plots(self.archive)
        return self.archive


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR.parent / "drowsiness_hard_200.jsonl"
    OUTPUT_DIR = BASE_DIR / "output" / "ibgwo4"
    FITNESS_METHOD = "weighted_mean"
    CONFIG = IBGWO4Config(
        population_size=50,
        generations=100,
        inertia=0.729,
        cognitive=1.49445,
        social=1.49445,
        velocity_limit=0.20,
        archive_size=100,
        seed=42,
    )

    X, y = load_jsonl(DATA_PATH)
    logger = IBGWO4Logger(OUTPUT_DIR)
    logger.save_config(CONFIG, DATA_PATH, FITNESS_METHOD, len(y))
    optimizer = IBGWO4(Fitness(X), y, FITNESS_METHOD, CONFIG, logger)
    archive = optimizer.run()
    print(f"\nFinished. Pareto solutions: {len(archive)}")
    print(f"Outputs: {OUTPUT_DIR}")
