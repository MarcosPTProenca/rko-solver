"""Authenticate and bound HTTP JSON before framework parsing or job admission.

Objective:
- Reject unauthenticated/oversized/deep/nonfinite input without exposing secrets or paths.

Technical approach:
- Use a constant-time bearer comparison and bounded ASGI body collection/deadline.
- Check structural depth before JSON parsing and replay only the bounded body downstream.

Flow (pseudocode):
1. Allow only the cheap health probe without authentication.
2. Reject missing/wrong credentials and unsupported body content types.
3. Bound streamed bytes, total read time, UTF-8, JSON depth and numeric constants.
4. Deliver the validated bounded request to FastAPI without logging it.

Important criteria:
- API secrets are deployment inputs, not committed defaults or request query parameters.
- No decoder, import target, pickle or filesystem operation comes from request code.

Inputs/Outputs:
- Input: ASGI scope/messages and trusted secret/body limit.
- Output: normal bounded request or a static redacted HTTP failure.
"""

import asyncio
import json
import math
import secrets
from starlette.types import ASGIApp
from starlette.types import Message
from starlette.types import Receive
from starlette.types import Scope
from starlette.types import Send


class RequestBoundary:
    """Enforce bearer authentication and bounded JSON at the ASGI boundary.

    Responsibilities:
    - Authenticate before buffering request data.
    - Reject excessive/deep/nonfinite bodies before Pydantic/adapter processing.

    Attributes:
        app (ASGIApp): Downstream application.
        secret (bytes): Expected deployment-injected bearer token, never logged.
        max_body (int): Maximum encoded request bytes.
    """

    def __init__(self, app: ASGIApp, api_key: str, max_body: int) -> None:
        """Store trusted server policy without accepting a default credential.

        Args:
            app (ASGIApp): Framework application to call after boundary checks.
            api_key (str): Required server-generated secret.
            max_body (int): Maximum complete body bytes.
        """
        self.app, self.secret, self.max_body = app, api_key.encode("utf-8"), max_body

    @staticmethod
    def check_json(body: bytes) -> None:
        """Bound nesting before decoding finite UTF-8 JSON, including escaped strings.

        Args:
            body (bytes): Already byte-bounded complete payload.

        Raises:
            ValueError: If nesting, encoding, syntax or numeric constants are invalid.
        """
        depth, quoted, escaped = 0, False, False
        for value in body:
            if quoted:
                if escaped:
                    escaped = False
                elif value == 92:
                    escaped = True
                elif value == 34:
                    quoted = False
            elif value == 34:
                quoted = True
            elif value in (91, 123):
                depth += 1
                if depth > 32:
                    raise ValueError("JSON nesting exceeds server policy.")
            elif value in (93, 125):
                depth -= 1
        try:
            json.loads(
                body.decode("utf-8"),
                parse_constant=RequestBoundary.reject_constant,
                parse_float=RequestBoundary.finite_float,
            )
        except (UnicodeError, RecursionError) as exc:
            raise ValueError("Invalid JSON encoding/depth.") from exc

    @staticmethod
    def finite_float(value: str) -> float:
        """Reject finite-looking JSON exponents that overflow Python's float domain.

        Args:
            value (str): Parser-provided numeric text, never echoed.

        Returns:
            float: Finite numeric value.

        Raises:
            ValueError: If the number overflows or cannot be represented finitely.
        """
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite JSON is not admitted.")
        return number

    @staticmethod
    def reject_constant(value: str) -> None:
        """Reject JSON NaN/Infinity instead of letting nonfinite values enter instance data.

        Args:
            value (str): Parser-provided nonstandard numeric constant, never echoed.

        Raises:
            ValueError: Always, because nonfinite constants are not admitted.
        """
        raise ValueError("Nonfinite JSON is not admitted.")

    @staticmethod
    async def respond(send: Send, status: int, detail: str) -> None:
        """Send a static redacted failure and close the request connection.

        Args:
            send (Send): ASGI response channel.
            status (int): Safe HTTP status.
            detail (str): Server-owned fixed message, not callback/input text.
        """
        body = json.dumps({"detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close"),
                    (b"x-content-type-options", b"nosniff"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Authenticate and collect only bounded bodies before invoking framework parsing.

        Args:
            scope (Scope): ASGI connection metadata.
            receive (Receive): Incoming HTTP messages.
            send (Send): Outgoing response messages.
        """
        if scope["type"] != "http" or (scope["path"] == "/health" and scope["method"] == "GET"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        authorization = headers.get(b"authorization", b"")
        scheme, _, token = authorization.partition(b" ")
        if (
            len(authorization) > 512
            or scheme.lower() != b"bearer"
            or not secrets.compare_digest(token, self.secret)
        ):
            await self.respond(send, 401, "Authentication required.")
            return
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            await self.respond(send, 400, "Invalid request length.")
            return
        if length < 0 or length > self.max_body:
            await self.respond(send, 413, "Request body exceeds server policy.")
            return
        if (
            scope["method"] == "POST"
            and headers.get(b"content-type", b"").split(b";")[0].lower() != b"application/json"
        ):
            await self.respond(send, 415, "Use application/json instance values.")
            return
        body = bytearray()
        try:
            async with asyncio.timeout(5):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    if len(body) + len(chunk) > self.max_body:
                        await self.respond(send, 413, "Request body exceeds server policy.")
                        return
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await self.respond(send, 408, "Request body deadline exceeded.")
            return
        if body:
            try:
                self.check_json(bytes(body))
            except ValueError:
                await self.respond(send, 400, "Invalid bounded finite JSON.")
                return
        delivered = False

        async def replay() -> Message:
            """Replay one validated body, then preserve normal disconnect semantics.

            Returns:
                Message: Complete bounded HTTP request or subsequent connection event.
            """
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
