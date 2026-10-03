from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GoogleUserPayload:
    sub: str
    email: str
    email_verified: bool
    name: str | None = None
    picture: str | None = None


class GoogleTokenVerifier(Protocol):
    def verify(self, id_token_string: str) -> GoogleUserPayload:
        """Verifies a Google ID token and returns the extracted payload or raises ValueError."""
        ...


class DefaultGoogleIdTokenVerifier:
    def __init__(self, client_id: str | None = None):
        self.client_id = client_id

    def verify(self, id_token_string: str) -> GoogleUserPayload:
        from google.auth.transport import requests
        from google.oauth2 import id_token

        request = requests.Request()
        try:
            payload = id_token.verify_oauth2_token(
                id_token_string,
                request,
                audience=self.client_id,
                clock_skew_in_seconds=10,
            )
        except Exception as error:
            raise ValueError(f"Invalid Google ID token: {error}") from error

        sub = payload.get("sub")
        email = payload.get("email")
        if not sub or not email:
            raise ValueError("Google ID token missing required claims ('sub', 'email')")

        email_verified = payload.get("email_verified", True)
        if not email_verified:
            raise ValueError("Google email is not verified")

        return GoogleUserPayload(
            sub=str(sub),
            email=str(email).lower(),
            email_verified=bool(email_verified),
            name=payload.get("name"),
            picture=payload.get("picture"),
        )
