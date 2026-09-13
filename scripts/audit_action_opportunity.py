"""Compatibility launcher for the canonical action-opportunity diagnostic."""

if __package__:
    from .diagnostics.audit_action_opportunity import *
    from .diagnostics.audit_action_opportunity import main
else:
    from diagnostics.audit_action_opportunity import *
    from diagnostics.audit_action_opportunity import main


if __name__ == "__main__":
    raise SystemExit(main())
