"""Allow ``python -m decode_engine`` to invoke the command-line tool."""

from .cli import main


raise SystemExit(main())
