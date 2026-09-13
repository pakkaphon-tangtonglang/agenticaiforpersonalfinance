"""Tests for LINE webhook signature verification."""

import base64
import hashlib
import hmac

from finance_ai.line.signature import verify_line_signature

CHANNEL_SECRET = "test-channel-secret"
BODY = '{"destination": "Ux", "events": []}'


def _sign(body: str, secret: str) -> str:
    """Compute a LINE-style HMAC-SHA256 signature for a body."""
    digest = hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


class TestVerifyLineSignature:
    """Tests for verify_line_signature."""

    def test_valid_signature_passes(self) -> None:
        """A correctly signed body verifies."""
        assert verify_line_signature(CHANNEL_SECRET, BODY, _sign(BODY, CHANNEL_SECRET))

    def test_wrong_signature_fails(self) -> None:
        """A tampered body is rejected."""
        assert not verify_line_signature(CHANNEL_SECRET, BODY, _sign("other", CHANNEL_SECRET))

    def test_empty_signature_fails(self) -> None:
        """A missing signature header is rejected."""
        assert not verify_line_signature(CHANNEL_SECRET, BODY, "")

    def test_empty_channel_secret_fails(self) -> None:
        """Verification fails when credentials are not configured."""
        assert not verify_line_signature("", BODY, _sign(BODY, CHANNEL_SECRET))
