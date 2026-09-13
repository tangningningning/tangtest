"""Compatibility launcher for the canonical single-episode generator."""

if __package__:
    from .generation.generate_episode_realization import *
    from .generation.generate_episode_realization import main
else:
    from generation.generate_episode_realization import *
    from generation.generate_episode_realization import main


if __name__ == "__main__":
    raise SystemExit(main())
