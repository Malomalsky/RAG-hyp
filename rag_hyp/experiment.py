from .config import CONFIG
from .train_utils import main_training_loop


def main():
    main_training_loop(CONFIG)


if __name__ == "__main__":
    main()
