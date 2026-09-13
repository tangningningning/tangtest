"""Compatibility launcher for the canonical episode-pool generator."""

if __package__:
    from .generation.generate_episode_pool import *
    from .generation.generate_episode_pool import main
else:
    from generation.generate_episode_pool import *
    from generation.generate_episode_pool import main


if __name__ == "__main__":
    raise SystemExit(main())
