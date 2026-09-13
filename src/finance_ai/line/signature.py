"""LINE webhook signature verification (HMAC-SHA256 via official SDK)."""

from linebot.v3.webhook import SignatureValidator


def verify_line_signature(channel_secret: str, body: str, signature: str) -> bool:
    """Verify the X-Line-Signature header against the raw request body.

    Args:
        channel_secret: LINE Messaging API channel secret.
        body: Raw request body string (exact bytes as received).
        signature: Value of the X-Line-Signature header.

    Returns:
        bool: True when the signature is valid and a secret is configured.

    Example:
        >>> verify_line_signature(secret, body, signature)
        True
    """
    if not channel_secret:
        return False
    validator: SignatureValidator = SignatureValidator(channel_secret)
    return bool(validator.validate(body, signature or ""))
