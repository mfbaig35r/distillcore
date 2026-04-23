try:
    from .server import main
except ImportError:
    import sys

    print(
        "distillcore MCP server requires the [mcp] extra.\n"
        "Install with: pip install distillcore[mcp]",
        file=sys.stderr,
    )
    raise SystemExit(1)

main()
