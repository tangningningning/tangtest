#!/usr/bin/env python3
"""Backward-compatible entry point for the cache update evolution plot."""

from __future__ import annotations

try:  # Support both ``python scripts/...py`` and package imports.
    from .plot_cache_update_evolution import main, plot_cache_update_evolution
except ImportError:  # pragma: no cover - direct script execution path
    from plot_cache_update_evolution import main, plot_cache_update_evolution


plot_cache_update_events = plot_cache_update_evolution


if __name__ == "__main__":
    raise SystemExit(main())
