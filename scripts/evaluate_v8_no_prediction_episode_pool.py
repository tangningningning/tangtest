"""Compatibility launcher for the canonical V8 ablation evaluator."""

if __package__:
    from .evaluation.evaluate_v8_no_prediction_episode_pool import (
        build_parser,
        evaluate_pool,
        main,
    )
else:
    from evaluation.evaluate_v8_no_prediction_episode_pool import (
        build_parser,
        evaluate_pool,
        main,
    )

__all__ = ["build_parser", "evaluate_pool", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
