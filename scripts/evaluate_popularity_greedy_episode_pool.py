"""Compatibility launcher for the canonical popularity-greedy baseline."""

if __package__:
    from .baselines.evaluate_popularity_greedy_episode_pool import (
        build_parser,
        evaluate_pool,
        main,
    )
else:
    from baselines.evaluate_popularity_greedy_episode_pool import (
        build_parser,
        evaluate_pool,
        main,
    )

__all__ = ["build_parser", "evaluate_pool", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
