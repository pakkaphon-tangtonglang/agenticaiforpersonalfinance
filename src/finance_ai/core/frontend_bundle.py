"""Frontend deploy bundle builder for iHost upload.

Copies the static frontend files and generates a ``config.js`` pointing
at the deployed backend, producing a folder ready for FTP upload.

Example:
    >>> bundle_frontend(
    ...     "https://app.onrender.com", static_dir, Path("dist/ihost")
    ... )
"""

import shutil
from pathlib import Path

from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

_COPIED_FILES = ("index.html", "styles.css", "app.js")


def _validate_api_base_url(api_base_url: str) -> None:
    """Reject empty or non-HTTPS API base URLs.

    Args:
        api_base_url: The backend base URL to validate.

    Raises:
        ValueError: When the URL is empty or not HTTPS. The iHost page is
            served over HTTPS, so a non-HTTPS API base would be blocked
            by browsers as mixed content.

    Example:
        >>> _validate_api_base_url("https://app.onrender.com")
    """
    if not api_base_url:
        raise ValueError(
            "api_base_url must not be empty. "
            "Pass the backend URL, e.g. https://<app>.onrender.com"
        )
    if not api_base_url.startswith("https://"):
        raise ValueError(
            "api_base_url must use HTTPS "
            f"(the iHost page is HTTPS; mixed content is blocked). Received: {api_base_url}"
        )


def _render_config_js(api_base_url: str) -> str:
    """Render the config.js contents for the given API base.

    Args:
        api_base_url: HTTPS backend base URL.

    Returns:
        JavaScript source assigning ``window.FINANCE_API_BASE``.

    Example:
        >>> _render_config_js("https://app.onrender.com")
    """
    return f'window.FINANCE_API_BASE = "{api_base_url}";\n'


def bundle_frontend(
    api_base_url: str,
    source_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Build the iHost upload folder: static files + generated config.js.

    Args:
        api_base_url: HTTPS backend base URL written into config.js.
        source_dir: Directory holding index.html, styles.css, app.js.
        output_dir: Bundle destination (created when missing).

    Returns:
        Paths of the bundled files.

    Raises:
        ValueError: When api_base_url is empty or not HTTPS.
        FileNotFoundError: When a required static file is missing.

    Example:
        >>> files = bundle_frontend("https://app.onrender.com", static, out)
    """
    _validate_api_base_url(api_base_url)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundled = _copy_static_files(source_dir, output_dir)
    bundled.append(_write_config_js(api_base_url, output_dir))
    logger.info("Bundled %d files into %s", len(bundled), output_dir)
    return bundled


def _copy_static_files(source_dir: Path, output_dir: Path) -> list[Path]:
    """Copy the required static files into the bundle directory.

    Args:
        source_dir: Directory holding the static files.
        output_dir: Bundle destination.

    Returns:
        Paths of the copied files.

    Raises:
        FileNotFoundError: When a required file is missing.
    """
    copied: list[Path] = []
    for name in _COPIED_FILES:
        source_path = source_dir / name
        if not source_path.is_file():
            raise FileNotFoundError(f"Required static file not found: {source_path}")
        destination = output_dir / name
        shutil.copyfile(source_path, destination)
        copied.append(destination)
    return copied


def _write_config_js(api_base_url: str, output_dir: Path) -> Path:
    """Write the generated config.js into the bundle directory.

    Args:
        api_base_url: HTTPS backend base URL.
        output_dir: Bundle destination.

    Returns:
        Path of the written config.js.
    """
    config_path = output_dir / "config.js"
    config_path.write_text(_render_config_js(api_base_url), encoding="utf-8")
    return config_path
