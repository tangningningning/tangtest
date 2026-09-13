"""Compatibility launcher for :mod:`scripts.evaluation.evaluate_v8_episode_pool`."""

if __package__:
    from .evaluation.evaluate_v8_episode_pool import build_parser, evaluate_pool, main
else:
    from evaluation.evaluate_v8_episode_pool import build_parser, evaluate_pool, main

__all__ = ["build_parser", "evaluate_pool", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
