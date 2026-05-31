import argparse

from src.eval.run_eval import run_eval
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained TISER adapter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", default=None, help="defaults to model/<run_name>/adapter")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_samples_per_split is not None:
        cfg.eval.max_samples_per_split = args.max_samples_per_split

    run_eval(cfg, adapter_dir=args.adapter_dir)


if __name__ == "__main__":
    main()
