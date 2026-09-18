from src.config import TrainConfig
from train.engine import run_training


def main():
    run_training(TrainConfig())


if __name__ == "__main__":
    main()
