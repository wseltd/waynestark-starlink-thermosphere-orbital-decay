"""``python -m thermosphere_consequence`` — the same entry point as the script.

Dispatches to :func:`thermosphere_consequence.cli.main.main`, the function the
``thermosphere-consequence`` console script also resolves to, so the two routes
cannot behave differently. Nothing else lives here: a module-level ``-m`` shim
that grew logic of its own would be a third entry point wearing the costume of
a second.

Importing this module runs the CLI, which is what ``-m`` means. That is why the
package's own ``__init__`` stays free of side effects — a bare
``import thermosphere_consequence`` must not run anything, and does not.
"""

from __future__ import annotations

from .cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
