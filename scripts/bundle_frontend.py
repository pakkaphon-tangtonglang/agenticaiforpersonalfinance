"""Bundle the static frontend for iHost KMITL upload.

Copies index.html, styles.css, app.js and generates a config.js pointing
at the deployed backend. Upload the resulting folder's contents to the
iHost web root via FTP.

Example:
    uv run python scripts/bundle_frontend.py --api-base https://<app>.onrender.com
"""

import argparse
from pathlib import Path

from finance_ai.core.frontend_bundle import bundle_frontend

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATIC_DIR = _PROJECT_ROOT / "src" / "finance_ai" / "static"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "dist" / "ihost"


def main() -> None:
    """Parse arguments and build the iHost upload bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-base",
        required=True,
        help="Backend URL, e.g. https://<app>.onrender.com (must be HTTPS)",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help=f"Output directory (default: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    bundled = bundle_frontend(
        api_base_url=args.api_base,
        source_dir=_STATIC_DIR,
        output_dir=Path(args.output),
    )
    print(f"Bundled {len(bundled)} files into {args.output}:")
    for path in bundled:
        print(f"  {path.name}")
    print("Upload these files to the iHost web root (public_html).")


if __name__ == "__main__":
    main()
