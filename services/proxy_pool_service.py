"""Proxy pool management: import, list, delete, assign to accounts."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from urllib.parse import quote as url_quote

from services.config import config
from services.storage.base import StorageBackend


def _clean(value: object) -> str:
    return str(value or "").strip()


def _parse_proxy_line(line: str) -> dict[str, Any] | None:
    """Parse 'IP:port:username:password' into a proxy pool item."""
    parts = line.strip().split(":")
    if len(parts) < 2:
        return None
    if len(parts) == 2:
        host, port = parts[0], parts[1]
        username, password = "", ""
    elif len(parts) == 4:
        host, port, username, password = parts
    else:
        host = parts[0]
        port = parts[1]
        username = parts[2]
        password = ":".join(parts[3:])
    host = host.strip()
    port = port.strip()
    username = username.strip()
    password = password.strip()
    if not host or not port:
        return None
    try:
        port_int = int(port)
    except ValueError:
        return None
    if port_int < 1 or port_int > 65535:
        return None
    if username and password:
        url = f"http://{url_quote(username, safe='')}:{url_quote(password, safe='')}@{host}:{port_int}"
    else:
        url = f"http://{host}:{port_int}"
    return {
        "id": str(uuid.uuid4()),
        "url": url,
        "host": host,
        "port": port_int,
        "username": username,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def make_subscription_item(
    subscription_id: str,
    node_id: str,
    port: int,
    protocol: str,
    region: str,
    raw_node: dict[str, Any],
    latency_ms: int,
) -> dict[str, Any]:
    """构造一个订阅式代理池 item（url=socks5://127.0.0.1:port，由 mihomo listener 提供出口）。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "id": str(uuid.uuid4()),
        "url": f"socks5://127.0.0.1:{port}",
        "host": "127.0.0.1",
        "port": port,
        "username": "",
        "created_at": now,
        "source": "subscription",
        "subscription_id": subscription_id,
        "node_id": node_id,
        "region": region,
        "protocol": protocol,
        "raw_node": raw_node,
        "local_port": port,
        "health": "ok",
        "last_check_at": now,
        "latency_ms": latency_ms,
    }


class ProxyPoolService:
    def __init__(self, storage: StorageBackend):
        self._storage = storage
        self._lock = Lock()
        self._items: list[dict[str, Any]] = self._storage.load_proxy_pool()

    def list_items(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._items)

    def import_proxies(self, text: str) -> dict[str, Any]:
        """Import proxies from text (one per line, format: IP:port:user:pass)."""
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        added = 0
        skipped = 0
        existing_urls = set()
        with self._lock:
            existing_urls = {item["url"] for item in self._items}
            for line in lines:
                parsed = _parse_proxy_line(line)
                if parsed is None:
                    skipped += 1
                    continue
                if parsed["url"] in existing_urls:
                    skipped += 1
                    continue
                self._items.append(parsed)
                existing_urls.add(parsed["url"])
                added += 1
            self._save()
        return {"added": added, "skipped": skipped, "total": len(self._items)}

    def delete_proxies(self, ids: list[str]) -> dict[str, Any]:
        """Delete proxies by their IDs."""
        id_set = set(ids)
        with self._lock:
            before = len(self._items)
            self._items = [item for item in self._items if item["id"] not in id_set]
            removed = before - len(self._items)
            self._save()
        return {"removed": removed, "total": len(self._items)}

    def clear_all(self) -> dict[str, Any]:
        """Remove all proxies from the pool."""
        with self._lock:
            removed = len(self._items)
            self._items = []
            self._save()
        return {"removed": removed, "total": 0}

    def assign_to_accounts(self) -> dict[str, Any]:
        """Assign proxies round-robin to all accounts."""
        from services.account_service import account_service

        with self._lock:
            pool = list(self._items)

        if not pool:
            return {"assigned": 0, "error": "代理池为空，请先导入代理"}

        accounts = account_service.list_accounts()
        if not accounts:
            return {"assigned": 0, "error": "没有账号可分配"}

        assigned = 0
        for i, account in enumerate(accounts):
            proxy_item = pool[i % len(pool)]
            access_token = _clean(account.get("access_token"))
            if not access_token:
                continue
            account_service.update_account(access_token, {"proxy": proxy_item["url"]}, provider=account.get("provider"))
            assigned += 1

        return {"assigned": assigned, "total_proxies": len(pool), "total_accounts": len(accounts)}

    def clear_assignments(self) -> dict[str, Any]:
        """Clear proxy assignments from all accounts."""
        from services.account_service import account_service

        accounts = account_service.list_accounts()
        cleared = 0
        for account in accounts:
            access_token = _clean(account.get("access_token"))
            if not access_token:
                continue
            if _clean(account.get("proxy")):
                account_service.update_account(access_token, {"proxy": ""}, provider=account.get("provider"))
                cleared += 1

        return {"cleared": cleared}

    def add_subscription_items(self, subscription_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        """批量追加订阅 item，去重按 (subscription_id, node_id)。"""
        with self._lock:
            existing_keys = {
                (it.get("subscription_id"), it.get("node_id"))
                for it in self._items
                if it.get("source") == "subscription"
            }
            added = 0
            for item in items:
                key = (item.get("subscription_id"), item.get("node_id"))
                if key in existing_keys:
                    continue
                self._items.append(item)
                existing_keys.add(key)
                added += 1
            self._save()
        return {"added": added, "total": len(self._items)}

    def replace_subscription_items(self, subscription_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        """整体替换某订阅的 item：删旧加新（sync 后调用）。"""
        with self._lock:
            before = sum(1 for it in self._items if it.get("subscription_id") == subscription_id)
            self._items = [it for it in self._items if it.get("subscription_id") != subscription_id]
            existing_keys = {
                (it.get("subscription_id"), it.get("node_id"))
                for it in self._items
                if it.get("source") == "subscription"
            }
            added = 0
            for item in items:
                key = (item.get("subscription_id"), item.get("node_id"))
                if key in existing_keys:
                    continue
                self._items.append(item)
                existing_keys.add(key)
                added += 1
            self._save()
        return {"removed": before, "added": added, "total": len(self._items)}

    def delete_subscription_items(self, subscription_id: str) -> dict[str, Any]:
        """删除某订阅的所有 item（删订阅源时调用）。"""
        with self._lock:
            before = len(self._items)
            self._items = [it for it in self._items if it.get("subscription_id") != subscription_id]
            removed = before - len(self._items)
            self._save()
        return {"removed": removed, "total": len(self._items)}

    def update_subscription_health(
        self, subscription_id: str, health_map: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """更新某订阅 item 的 health/latency/last_check_at。health_map: node_id -> {health, latency_ms}。"""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        updated = 0
        with self._lock:
            for it in self._items:
                if it.get("subscription_id") != subscription_id:
                    continue
                nid = it.get("node_id")
                if nid in health_map:
                    info = health_map[nid]
                    it["health"] = info.get("health", "unknown")
                    it["latency_ms"] = info.get("latency_ms", 0)
                    it["last_check_at"] = now
                    updated += 1
            self._save()
        return {"updated": updated}

    def reassign_invalid_accounts(self) -> dict[str, Any]:
        """找出 proxy 失效的账号（空或不在可用集合）并重新分配，不动仍有效的账号。"""
        from services.account_service import account_service

        with self._lock:
            pool = list(self._items)
        if not pool:
            return {"reassigned": 0, "error": "代理池为空"}
        usable = [it for it in pool if it.get("health", "unknown") != "down"]
        if not usable:
            return {"reassigned": 0, "error": "无可用代理"}
        usable_urls = {it["url"] for it in usable}
        accounts = account_service.list_accounts()
        if not accounts:
            return {"reassigned": 0, "error": "没有账号"}
        reassigned = 0
        pick = 0
        for account in accounts:
            access_token = _clean(account.get("access_token"))
            if not access_token:
                continue
            cur = _clean(account.get("proxy"))
            if cur and cur in usable_urls:
                continue
            new_item = usable[pick % len(usable)]
            pick += 1
            account_service.update_account(access_token, {"proxy": new_item["url"]}, provider=account.get("provider"))
            reassigned += 1
        return {"reassigned": reassigned, "total_usable": len(usable), "total_accounts": len(accounts)}

    def _save(self) -> None:
        self._storage.save_proxy_pool(self._items)


proxy_pool_service = ProxyPoolService(config.get_storage_backend())
