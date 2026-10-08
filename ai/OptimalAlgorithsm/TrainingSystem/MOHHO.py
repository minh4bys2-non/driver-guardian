"""MOHHO adapted to continuous feature weights and three objectives.

Reference: https://bseujert.bilecik.edu.tr/index.php/bseujert/article/view/14
Uses HHO exploration, four besiege strategies and Levy dives. Archive
leaders use crowding-distance roulette; truncation reuses the project's
crowding-distance rule. Dive selection uses Pareto dominance, with random
selection among mutually nondominated parent/Y/Z candidates.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ai.OptimalAlgorithsm.TrainingSystem.cmdpsofs import (
    ParetoArchive,
    RunLogger,
    crowding_distance,
    dominates,
)
from ai.OptimalAlgorithsm.TrainingSystem.fitnessFunction import FEATURES, Fitness, load_jsonl


@dataclass
class MOHHOConfig:
    population_size: int = 50
    generations: int = 100
    dimensions: int = 9
    archive_size: int = 100
    levy_beta: float = 1.5
    levy_scale: float = 0.01
    seed: int = 42

    def __post_init__(self):
        if self.population_size < 1 or self.generations < 1 or self.archive_size < 1:
            raise ValueError("population_size, generations and archive_size must be positive")
        if self.dimensions != 9:
            raise ValueError("Fitness requires 8 weights and 1 classification threshold")
        if not 1 < self.levy_beta < 2:
            raise ValueError("levy_beta must be in (1, 2)")
        if not np.isfinite(self.levy_scale) or self.levy_scale < 0:
            raise ValueError("levy_scale must be finite and nonnegative")


@dataclass
class Hawk:
    position: np.ndarray
    objectives: np.ndarray


class MOHHOArchive(ParetoArchive):
    def select_leader(self, rng):
        if not len(self):
            raise RuntimeError("Pareto archive is empty")
        distance = crowding_distance(self.objectives).astype(np.float64)
        finite = distance[np.isfinite(distance)]
        # Give boundary solutions finite, preferential roulette weights.
        distance[np.isinf(distance)] = 2.0 * max(float(finite.max(initial=0)), 1.0)
        weights = np.maximum(distance, 1e-12)
        index = rng.choice(len(self), p=weights / weights.sum())
        return self.positions[index].copy()

class MOHHOLogger(RunLogger):
    def save_config(self, config, data_path, fitness_method, n_samples):
        payload = {
            "algorithm": "MOHHO-style continuous multi-objective Harris hawks optimization",
            "reference": "https://bseujert.bilecik.edu.tr/index.php/bseujert/article/view/14",
            "leader_selection": "archive crowding-distance roulette",
            "archive_truncation": "remove minimum crowding distance",
            "dive_selection": "uniform among nondominated parent, Y and Z",
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


class MOHHO:
    def __init__(self, fitness, y_true, method, config, logger):
        self.fitness = fitness
        self.y_true = np.asarray(y_true, dtype=np.int8)
        self.method = method
        self.cfg = config
        self.logger = logger
        self.rng = np.random.default_rng(config.seed)
        self.archive = MOHHOArchive(config.archive_size)
        self.hawks = []
        beta = config.levy_beta
        self.levy_sigma = (
            math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

    def _evaluate(self, candidate):
        if not np.any(candidate[:8] >= 0.5):
            return np.ones(3, dtype=np.float32)
        prediction = self.fitness.predict(candidate, method=self.method)
        return self.fitness.evaluate(candidate, prediction, self.y_true)

    def _hawk(self, position):
        position = np.clip(position, 0.0, 1.0).astype(np.float32)
        return Hawk(position, self._evaluate(position))

    def _update_archive(self, hawks):
        self.archive.update(
            np.vstack([h.position for h in hawks]),
            np.vstack([h.objectives for h in hawks]),
        )

    def _initialize(self):
        positions = self.rng.random((self.cfg.population_size, self.cfg.dimensions))
        self.hawks = [self._hawk(position) for position in positions]
        self._update_archive(self.hawks)

    def _levy(self):
        u = self.rng.normal(0.0, self.levy_sigma, self.cfg.dimensions)
        v = np.maximum(np.abs(self.rng.normal(size=self.cfg.dimensions)), 1e-12)
        return self.cfg.levy_scale * u / v ** (1.0 / self.cfg.levy_beta)

    def _move(self, hawk, rabbit, mean_position, positions, energy):
        position = hawk.position
        if abs(energy) >= 1.0:
            if self.rng.random() >= 0.5:
                peer = positions[self.rng.integers(len(positions))]
                r1, r2 = self.rng.random(2)
                new_position = peer - r1 * np.abs(peer - 2.0 * r2 * position)
            else:
                r3, r4 = self.rng.random(2)
                new_position = rabbit - mean_position - r3 * r4
            child = self._hawk(new_position)
            return child, [child]

        escape = self.rng.random()
        jump = 2.0 * (1.0 - self.rng.random())
        if escape >= 0.5:
            if abs(energy) >= 0.5:
                new_position = rabbit - position - energy * np.abs(jump * rabbit - position)
            else:
                new_position = rabbit - energy * np.abs(rabbit - position)
            child = self._hawk(new_position)
            return child, [child]

        target = position if abs(energy) >= 0.5 else mean_position
        y = rabbit - energy * np.abs(jump * rabbit - target)
        z = y + self.rng.random(self.cfg.dimensions) * self._levy()
        trials = [self._hawk(y), self._hawk(z)]
        candidates = [hawk] + trials
        front = [
            h for h in candidates
            if not any(dominates(other.objectives, h.objectives) for other in candidates)
        ]
        return front[self.rng.integers(len(front))], trials

    def _step(self, generation):
        positions = np.vstack([h.position for h in self.hawks])
        mean_position = positions.mean(axis=0)
        energy_scale = 2.0 * (1.0 - (generation - 1) / max(self.cfg.generations - 1, 1))
        next_hawks, evaluated = [], []
        for hawk in self.hawks:
            rabbit = self.archive.select_leader(self.rng)
            energy = energy_scale * self.rng.uniform(-1.0, 1.0)
            child, trials = self._move(hawk, rabbit, mean_position, positions, energy)
            next_hawks.append(child)
            evaluated.extend(trials)
        # Archive every evaluated dive, including nondominated unselected trials.
        self._update_archive(evaluated)
        self.hawks = next_hawks

    def run(self):
        start = time.perf_counter()
        self._initialize()
        for generation in range(self.cfg.generations):
            if generation > 0:
                self._step(generation)
            self.logger.log_population(generation, self.hawks)
            self.logger.log_generation(
                generation, self.hawks, self.archive, time.perf_counter() - start
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
    OUTPUT_DIR = BASE_DIR / "output" / "mohho"
    FITNESS_METHOD = "noisy_or"
    CONFIG = MOHHOConfig(
        population_size=50,
        generations=100,
        archive_size=100,
        levy_beta=1.5,
        levy_scale=0.01,
        seed=42,
    )

    X, y = load_jsonl(DATA_PATH)
    logger = MOHHOLogger(OUTPUT_DIR)
    logger.save_config(CONFIG, DATA_PATH, FITNESS_METHOD, len(y))
    optimizer = MOHHO(Fitness(X), y, FITNESS_METHOD, CONFIG, logger)
    archive = optimizer.run()
    print(f"\nFinished. Pareto solutions: {len(archive)}")
    print(f"Outputs: {OUTPUT_DIR}")
