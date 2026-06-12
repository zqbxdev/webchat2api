from __future__ import annotations

from typing import Any


def build_session_kwargs(*, impersonate: str | None = None, verify: bool = True, account: dict | None = None, **session_kwargs: Any) -> dict[str, object]:
    if impersonate:
        session_kwargs["impersonate"] = impersonate
    session_kwargs["verify"] = verify
    from services.proxy_service import proxy_settings

    return proxy_settings.build_session_kwargs(account=account, **session_kwargs)


def create_session(*, impersonate: str | None = None, verify: bool = True, account: dict | None = None, **session_kwargs: Any):
    from curl_cffi import requests

    return requests.Session(**build_session_kwargs(impersonate=impersonate, verify=verify, account=account, **session_kwargs))
