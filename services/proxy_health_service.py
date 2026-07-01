"""订阅代理池健康服务：周期/手动同步订阅 → 测活 → 更新 proxy_pool → 重分配失效账号。

闭环：mihomo_manager.build_and_load（拉订阅+筛+起 mihomo 多端口）→ probe_all（curl_cffi
测 chatgpt.com）→ make_subscription_item + replace_subscription_items（只留可用）
→ reassign_invalid_accounts（仅失效账号换出口）。

调度仿 BackupService（threading.Thread + stop_event.wait）。sync 单飞锁防并发。
长任务异步化：run_sync_now 返回 task_id，前端轮询 get_task_status。

首版限制：多订阅时每个订阅 build_and_load 会覆盖 mihomo 配置（以最后订阅为准）；多订阅
节点合并留后续。当前用户单订阅场景不受影响。
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from services.config import config
from services.mihomo_manager import mihomo_manager, probe_all
from services.proxy_pool_service import make_subscription_item, proxy_pool_service


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _parse_region(raw: str) -> list[str]:
    return [k.strip() for k in str(raw or "").split(",") if k.strip()]


class ProxyHealthService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sync_lock = threading.Lock()  # sync 单飞锁
        self._tasks: dict[str, dict[str, Any]] = {}
        self._storage = config.get_storage_backend()

    # ── 调度 ──
    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="proxy-health-scheduler")
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()
            thread = self._thread
            self._thread = None
        if thread and thread.is_alive():
            thread.join(timeout=2)

    def _run(self) -> None:
        """周期 sync：启动后立即跑一次（若有启用订阅），之后按 sync_interval_hours 周期。"""
        # 首次启动立即同步（补强5）
        self._try_sync(task_id=None, subscription_id=None)
        while not self._stop_event.is_set():
            try:
                settings = config.get_mihomo_settings()
                interval = max(1, int(settings.get("sync_interval_hours") or 24)) * 3600
            except Exception:
                interval = 24 * 3600
            self._stop_event.wait(interval)
            if self._stop_event.is_set():
                break
            self._try_sync(task_id=None, subscription_id=None)

    # ── sync 单飞 + 异步 task ──
    def run_sync_now(self, subscription_id: str | None = None) -> dict[str, Any]:
        """立即触发一次 sync（异步），返回 task_id 供轮询。若已有 sync 在跑则返回 skipped。"""
        task_id = str(uuid.uuid4())
        with self._lock:
            self._tasks[task_id] = {
                "status": "pending",
                "progress": 0,
                "result": None,
                "error": None,
                "started_at": _now(),
            }
        threading.Thread(
            target=self._try_sync, args=(task_id, subscription_id), daemon=True, name="proxy-sync"
        ).start()
        return {"task_id": task_id}

    def _try_sync(self, task_id: str | None, subscription_id: str | None) -> None:
        """单飞：抢不到锁则跳过（手动 sync 记 skipped）。抢到则跑 run_sync。"""
        if not self._sync_lock.acquire(blocking=False):
            if task_id:
                self._update_task(task_id, status="skipped", error="another sync running")
            return
        try:
            if task_id:
                self._update_task(task_id, status="running", progress=10)
            result = self.run_sync(subscription_id)
            if task_id:
                status = "error" if result.get("errors") and not result.get("usable") else "done"
                self._update_task(task_id, status=status, progress=100, result=result)
        except Exception as e:
            if task_id:
                self._update_task(task_id, status="error", error=f"{type(e).__name__}: {e}")
        finally:
            self._sync_lock.release()

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._tasks.get(task_id, {"status": "unknown", "error": "task not found"}))

    def _update_task(self, task_id: str, **fields) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.update(fields)

    # ── sync 核心闭环 ──
    def run_sync(self, subscription_id: str | None = None) -> dict[str, Any]:
        """同步执行 sync 闭环（阻塞，调用方负责单飞）。返回 {total, usable, reassigned, errors}。"""
        settings = config.get_mihomo_settings()
        test_url = str(settings.get("test_url") or "https://chatgpt.com/api/auth/csrf")
        timeout = float(settings.get("test_timeout") or 15)
        concurrency = int(settings.get("concurrency") or 16)
        retry = int(settings.get("probe_retry") or 1)

        subs = self._storage.load_subscriptions()
        if subscription_id:
            subs = [s for s in subs if s.get("id") == subscription_id]
        subs = [s for s in subs if s.get("enabled", True)]
        if not subs:
            return {"error": "无启用的订阅源", "total": 0, "usable": 0, "reassigned": 0, "errors": []}

        total_all = 0
        usable_all = 0
        errors: list[str] = []

        for sub in subs:
            region = _parse_region(sub.get("region_keywords") or settings.get("region_keywords") or "")
            try:
                port_map, mapped = mihomo_manager.build_and_load(sub["url"], region)
            except Exception as e:
                msg = f"{sub.get('name')}: {type(e).__name__}: {e}"
                errors.append(msg)
                self._update_sub_last_sync(sub["id"], error=msg)
                continue

            results = probe_all(
                port_map, test_url, timeout, concurrency, retry,
            )
            items: list[dict[str, Any]] = []
            region_str = ",".join(region)
            for node_id, node, _orig in mapped:
                r = results.get(node_id, {})
                if r.get("ok"):
                    items.append(
                        make_subscription_item(
                            sub["id"], node_id, port_map[node_id],
                            str(node.get("type") or ""), region_str, dict(node),
                            int(r.get("latency_ms") or 0),
                        )
                    )
            proxy_pool_service.replace_subscription_items(sub["id"], items)
            self._update_sub_last_sync(sub["id"], total=len(mapped), usable=len(items))
            total_all += len(mapped)
            usable_all += len(items)

        # sync 后重分配失效账号（仅失效的，补强决策）
        try:
            reassign = proxy_pool_service.reassign_invalid_accounts()
            reassigned = int(reassign.get("reassigned") or 0)
        except Exception as e:
            reassigned = 0
            errors.append(f"reassign_failed: {e}")

        return {
            "total": total_all,
            "usable": usable_all,
            "reassigned": reassigned,
            "errors": errors,
        }

    # ── 订阅源 CRUD ──
    def list_subscriptions(self) -> list[dict[str, Any]]:
        return self._storage.load_subscriptions()

    def add_subscription(self, name: str, url: str, region_keywords: str = "") -> dict[str, Any]:
        subs = self._storage.load_subscriptions()
        sub = {
            "id": str(uuid.uuid4()),
            "name": str(name or "").strip() or "未命名订阅",
            "url": str(url or "").strip(),
            "region_keywords": str(region_keywords or "").strip(),
            "enabled": True,
            "last_sync_at": "",
            "last_total": 0,
            "last_usable": 0,
            "last_error": "",
        }
        subs.append(sub)
        self._storage.save_subscriptions(subs)
        return sub

    def delete_subscription(self, sub_id: str) -> dict[str, Any]:
        subs = self._storage.load_subscriptions()
        before = len(subs)
        subs = [s for s in subs if s.get("id") != sub_id]
        removed = before - len(subs)
        self._storage.save_subscriptions(subs)
        if removed:
            proxy_pool_service.delete_subscription_items(sub_id)
        return {"removed": removed}

    def update_subscription(self, sub_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        subs = self._storage.load_subscriptions()
        updated = None
        for s in subs:
            if s.get("id") == sub_id:
                for k in ("name", "url", "region_keywords", "enabled"):
                    if k in fields:
                        s[k] = fields[k]
                updated = s
        self._storage.save_subscriptions(subs)
        return updated or {"error": "subscription not found"}

    def _update_sub_last_sync(
        self, sub_id: str, total: int = 0, usable: int = 0, error: str = ""
    ) -> None:
        subs = self._storage.load_subscriptions()
        for s in subs:
            if s.get("id") == sub_id:
                s["last_sync_at"] = _now()
                s["last_total"] = total
                s["last_usable"] = usable
                s["last_error"] = error
        self._storage.save_subscriptions(subs)


proxy_health_service = ProxyHealthService()
