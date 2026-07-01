from __future__ import annotations

from typing import Any

def build_session_kwargs(*, account_proxy: str = "", impersonate: str | None = None, verify: bool = True, **session_kwargs: Any) -> dict[str, object]:
    if impersonate:
        session_kwargs["impersonate"] = impersonate
    session_kwargs["verify"] = verify
    from services.proxy_service import proxy_settings

    return proxy_settings.build_session_kwargs(account_proxy=account_proxy, **session_kwargs)


def create_session(*, account_proxy: str = "", impersonate: str | None = None, verify: bool = True, **session_kwargs: Any):
    from curl_cffi import requests

    return requests.Session(**build_session_kwargs(account_proxy=account_proxy, impersonate=impersonate, verify=verify, **session_kwargs))
