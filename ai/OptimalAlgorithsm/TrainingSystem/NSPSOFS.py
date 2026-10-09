"""NSPSOFS adapted to continuous weights and three minimization objectives.

Reference: Xue, Zhang and Browne, Particle Swarm Optimization for Feature
Selection in Classification: A Multi-Objective Approach (Algorithm 1).
https://homepages.ecs.vuw.ac.nz/~xuebing/Papers/BingIEEETransCybernetics.pdf
The historical archive is for reporting only; leaders come from the swarm.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ai.OptimalAlgorithsm.TrainingSystem.IBGWO4 import pareto_order
from ai.OptimalAlgorithsm.TrainingSystem.cmdpsofs import (
    ParetoArchive,
    Particle,
    RunLogger,
    crowding_distance,
    dominates,
)
from ai.OptimalAlgorithsm.TrainingSystem.fitnessFunction import FEATURES, Fitness, load_jsonl, validate_labels


@dataclass
class NSPSOFSConfig:
    population_size: int = 50
    generations: int = 100
    dimensions: int = 9
    inertia: float = 0.729
    cognitive: float = 1.49445
    social: float = 1.49445
    velocity_limit: float = 0.20
    archive_size: int = 100
    leader_fraction: float = 0.20  # Configurable cutoff for least-crowded leaders.
    seed: int = 42

    def __post_init__(self):
        if any(type(v) is not int for v in (self.population_size, self.generations, self.archive_size)):
            raise ValueError("population_size, generations and archive_size must be integers")
        if self.population_size < 1 or self.generations < 1 or self.archive_size < 1:
            raise ValueError("population_size, generations and archive_size must be positive")
        if type(self.dimensions) is not int or self.dimensions != 9:
            raise ValueError("Fitness requires 8 weights and 1 classification threshold")
        if not 0 < self.leader_fraction <= 1:
            raise ValueError("leader_fraction must be in (0, 1]")
        coefficients = (self.inertia, self.cognitive, self.social, self.velocity_limit)
        if not np.all(np.isfinite(coefficients)) or min(coefficients) < 0:
            raise ValueError("PSO coefficients and velocity_limit must be finite and nonnegative")


class NSPSOFSLogger(RunLogger):
    def save_config(self, config, data_path, fitness_method, n_samples):
        payload = {
            "algorithm": "NSPSOFS-style continuous multi-objective PSO",
            "selection": "parent + offspring; Pareto rank and crowding distance",
            "leader_selection": "uniform among top leader_fraction of swarm's first front by crowding distance",
            "pbest_update": "strict Pareto dominance only",
            "archive_role": "historical reporting only; not used for leader selection",
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


class NSPSOFS:
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

    def _leaders(self):
        objectives = np.vstack([p.objectives for p in self.particles])
        domination = (
            np.all(objectives[:, None] <= objectives[None, :], axis=2)
            & np.any(objectives[:, None] < objectives[None, :], axis=2)
        )
        front = np.flatnonzero(~domination.any(axis=0))
        # Randomize ties so equally crowded extremes all have a chance to lead.
        front = self.rng.permutation(front)
        distance = crowding_distance(objectives[front])
        order = np.argsort(-distance, kind="stable")
        count = max(1, int(np.ceil(len(front) * self.cfg.leader_fraction)))
        return np.vstack([self.particles[i].position for i in front[order[:count]]])

    def _child(self, parent, leader):
        cfg = self.cfg
        r1, r2 = self.rng.random((2, cfg.dimensions))
        velocity = (
            cfg.inertia * parent.velocity
            + cfg.cognitive * r1 * (parent.pbest_position - parent.position)
            + cfg.social * r2 * (leader - parent.position)
        )
        velocity = np.clip(velocity, -cfg.velocity_limit, cfg.velocity_limit).astype(np.float32)
        position = np.clip(parent.position + velocity, 0.0, 1.0).astype(np.float32)
        objectives = self._evaluate(position)
        improved = dominates(objectives, parent.pbest_objectives)
        return Particle(
            position=position,
            velocity=velocity,
            objectives=objectives,
            pbest_position=(position if improved else parent.pbest_position).copy(),
            pbest_objectives=(objectives if improved else parent.pbest_objectives).copy(),
        )

    def _step(self):
        leaders = self._leaders()
        children = [
            self._child(parent, leaders[self.rng.integers(len(leaders))])
            for parent in self.particles
        ]
        candidates = self.particles + children
        self._update_archive(candidates)
        order = pareto_order(np.vstack([p.objectives for p in candidates]))
        self.particles = [candidates[i] for i in order[:self.cfg.population_size]]

    def run(self):
        start = time.perf_counter()
        self._initialize()
        for generation in range(self.cfg.generations):
            if generation > 0:
                self._step()
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
    DATA_PATH = BASE_DIR.parent / "drowsiness_risk.jsonl"
    OUTPUT_DIR = BASE_DIR / "output" / "nspsofs"
    FITNESS_METHOD = "noisy_or"
    CONFIG = NSPSOFSConfig(
        population_size=50,
        generations=100,
        inertia=0.729,
        cognitive=1.49445,
        social=1.49445,
        velocity_limit=0.20,
        archive_size=100,
        leader_fraction=0.20,
        seed=42,
    )

    X, y = load_jsonl(DATA_PATH)
    logger = NSPSOFSLogger(OUTPUT_DIR)
    logger.save_config(CONFIG, DATA_PATH, FITNESS_METHOD, len(y))
    optimizer = NSPSOFS(Fitness(X), y, FITNESS_METHOD, CONFIG, logger)
    archive = optimizer.run()
    print(f"\nFinished. Pareto solutions: {len(archive)}")
    print(f"Outputs: {OUTPUT_DIR}")
