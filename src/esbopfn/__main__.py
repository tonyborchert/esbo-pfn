"""Allow ``python -m esbopfn`` as an alias of the ``pfn`` console script."""

from esbopfn.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
