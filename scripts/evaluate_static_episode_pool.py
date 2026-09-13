"""Compatibility launcher for the canonical Static baseline evaluator."""

if __package__:
    from .baselines.evaluate_static_episode_pool import build_parser, evaluate_pool, main
else:
    from baselines.evaluate_static_episode_pool import build_parser, evaluate_pool, main

__all__ = ["build_parser", "evaluate_pool", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
