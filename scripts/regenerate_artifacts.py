"""Compatibility launcher for the canonical artifact generator."""

if __package__:
    from .generation.regenerate_artifacts import *
    from .generation.regenerate_artifacts import main
else:
    from generation.regenerate_artifacts import *
    from generation.regenerate_artifacts import main


if __name__ == "__main__":
    raise SystemExit(main())
