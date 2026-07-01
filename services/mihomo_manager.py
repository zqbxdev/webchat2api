"""mihomo 代理内核生命周期管理 + 订阅节点配置生成。

把订阅里异构的加密协议节点（VLESS/VMess/SS/AnyTLS…）通过 mihomo `listeners` 多端口
统一成本地 `socks5://127.0.0.1:<port>` 入口，每条 listener 的 `proxy` 字段钉死一个节点
（直出绕过主路由），供 proxy_pool 消费。消费侧（curl_cffi/account.proxy）零改动。

设计要点（见 plans/groovy-painting-nebula.md Review 补强）：
- node_id 用节点内容指纹（server+port+type+uuid/password），同节点跨 sync 端口稳定。
- 配置变更用 stop+start（external-controller reload 对 listeners 增删未验证，首版不依赖）。
- start 失败不抛异常，返回 False，由 lifespan 容错降级（不拖垮主服务）。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml
from curl_cffi import requests as ccreq

from services.config import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MIHOMO_DIR = DATA_DIR / "mihomo"
GH_API_LATEST = "https://api.github.com/repos/MetaCubeX/mihomo/releases/latest"
DEFAULT_REGION_KEYWORDS = ["🇺🇸", "美国", "美國", "United States", "America"]


def log(msg: str) -> None:
    print(f"[mihomo] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────
# 订阅拉取 / 解析 / 筛选 / 配置生成（共享逻辑，scripts/sync_subscription.py 也复用）
# ──────────────────────────────────────────────────────────────
def fetch_subscription(url: str, timeout: float = 30.0) -> str:
    """带 clash UA 拉取订阅原文。"""
    s = ccreq.Session()
    try:
        r = s.get(url, headers={"user-agent": "clash-verge/v1.0"}, timeout=timeout)
        r.raise_for_status()
        return r.text
    finally:
        s.close()


def parse_proxies(yaml_text: str) -> list[dict]:
    """解析 Clash YAML，只取真实 proxies（丢弃 proxy-groups）。"""
    data = yaml.safe_load(yaml_text) or {}
    proxies = data.get("proxies") or []
    return [p for p in proxies if isinstance(p, dict)]


def _node_fingerprint(node: dict) -> str:
    """节点内容指纹：server+port+type+uuid/password。同节点跨 sync 稳定。"""
    raw = "|".join(
        str(node.get(k) or "")
        for k in ("server", "port", "type", "uuid", "password", "username")
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def filter_region(
    nodes: list[dict], keywords: list[str], prefix: str = "us"
) -> list[tuple[str, dict, str]]:
    """按 name 关键词筛选节点，返回 [(node_id, 重命名副本, 原始name)]。

    node_id = prefix + 内容指纹；同指纹节点去重（重复节点只留一个）。
    """
    seen: set[str] = set()
    out: list[tuple[str, dict, str]] = []
    for i, node in enumerate(nodes):
        orig = str(node.get("name") or f"node-{i}")
        if not any(k and k in orig for k in keywords):
            continue
        fp = _node_fingerprint(node)
        if fp in seen:
            continue
        seen.add(fp)
        node_id = f"{prefix}-{fp}"
        renamed = dict(node)
        renamed["name"] = node_id
        out.append((node_id, renamed, orig))
    return out


def allocate_ports(
    new_node_ids: list[str],
    old_port_map: dict[str, int],
    port_base: int,
    range_size: int,
) -> dict[str, int]:
    """端口分配：内容稳定的节点复用旧端口（跨 sync 不变），新节点分配未用端口（不挤占）。"""
    used_ports = set(old_port_map.values())
    new_map: dict[str, int] = {}
    for nid in new_node_ids:
        if nid in old_port_map:
            new_map[nid] = old_port_map[nid]
    next_port = port_base
    for nid in sorted(nid for nid in new_node_ids if nid not in new_map):
        while next_port in used_ports or next_port in new_map.values():
            next_port += 1
            if next_port >= port_base + range_size:
                raise RuntimeError(f"端口范围 {range_size} 耗尽，无法为新节点分配端口")
        new_map[nid] = next_port
        used_ports.add(next_port)
        next_port += 1
    return new_map


def build_mihomo_config(
    mapped: list[tuple[str, dict, str]],
    port_map: dict[str, int],
    external_controller: str = "127.0.0.1:9090",
    external_controller_secret: str = "",
) -> tuple[dict, dict[str, int]]:
    """每节点一个 socks listener，proxy 钉死该节点。端口由 port_map 给定（已由 allocate_ports 分配）。"""
    proxies: list[dict] = []
    listeners: list[dict] = []
    for node_id, node, _orig in mapped:
        port = port_map.get(node_id)
        if port is None:
            continue
        proxies.append(node)
        listeners.append(
            {
                "name": f"lis-{node_id}",
                "type": "socks",
                "listen": "127.0.0.1",
                "port": port,
                "proxy": node_id,
            }
        )
    cfg: dict[str, Any] = {
        "log-level": "warning",
        "mode": "direct",
        "allow-lan": False,
        "ipv6": False,
        "tcp-concurrent": True,
        "external-controller": external_controller,
        "proxies": proxies,
        "listeners": listeners,
    }
    if external_controller_secret:
        cfg["secret"] = external_controller_secret
    return cfg, port_map


# ──────────────────────────────────────────────────────────────
# 测活
# ──────────────────────────────────────────────────────────────
def probe(port: int, test_url: str, timeout: float, retry: int = 1) -> dict:
    """通过指定 socks5 端口访问 test_url。chatgpt.com 用 csrf 判据，其他用 200。失败重试。"""
    is_chatgpt = "chatgpt.com" in test_url
    last: dict = {"ok": False, "status": 0, "latency_ms": 0, "error": "not tested"}
    for attempt in range(retry + 1):
        s = ccreq.Session(impersonate="edge101", verify=True, proxy=f"socks5://127.0.0.1:{port}")
        t0 = time.perf_counter()
        try:
            r = s.get(test_url, headers={"user-agent": "Mozilla/5.0"}, timeout=timeout)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            body = r.text if r.status_code == 200 else ""
            if is_chatgpt:
                ok = r.status_code == 200 and "csrftoken" in body.lower()
            else:
                ok = r.status_code == 200
            if ok:
                return {"ok": True, "status": int(r.status_code), "latency_ms": latency_ms, "error": None}
            error = "200 but no csrfToken (likely CF challenge)" if r.status_code == 200 else f"HTTP {r.status_code}"
            last = {"ok": False, "status": int(r.status_code), "latency_ms": latency_ms, "error": error}
        except Exception as e:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            last = {"ok": False, "status": 0, "latency_ms": latency_ms, "error": f"{type(e).__name__}: {e}"[:300]}
        finally:
            s.close()
        if attempt < retry:
            time.sleep(0.3)
    return last


def probe_all(
    port_map: dict[str, int], test_url: str, timeout: float, workers: int, retry: int = 1,
    progress_cb: Any = None,
) -> dict[str, dict]:
    """并发测活所有端口。progress_cb(done, total, ok) 可选进度回调。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(probe, p, test_url, timeout, retry): nid for nid, p in port_map.items()}
        total = len(fut)
        done = 0
        for f in as_completed(fut):
            nid = fut[f]
            results[nid] = f.result()
            done += 1
            if progress_cb:
                ok = sum(1 for v in results.values() if v["ok"])
                progress_cb(done, total, ok)
    return results


# ──────────────────────────────────────────────────────────────
# mihomo 二进制管理
# ──────────────────────────────────────────────────────────────
def _arch_keys() -> list[str]:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return ["linux-amd64-compatible", "linux-amd64"]
    if m in ("aarch64", "arm64"):
        return ["linux-arm64"]
    raise RuntimeError(f"不支持的架构：{m}")


def _pick_asset_url(meta: dict, keys: list[str]) -> str:
    assets = {a["name"].lower(): a["browser_download_url"] for a in meta.get("assets", [])}
    for k in keys:
        for name, url in assets.items():
            if name.endswith(".gz") and k in name:
                return url
    raise RuntimeError(f"未找到匹配架构 {keys} 的 mihomo release 资产")


def ensure_mihomo_binary(target: Path) -> Path:
    """确保 mihomo 二进制存在，不存在则从 GitHub release 下载。"""
    if target.exists() and os.access(target, os.X_OK):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    log("查询 mihomo 最新版本……")
    req = urllib.request.Request(GH_API_LATEST, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        meta = json.load(r)
    tag = meta.get("tag_name", "unknown")
    url = _pick_asset_url(meta, _arch_keys())
    gz_path = target.with_suffix(target.suffix + ".gz")
    log(f"下载 mihomo {tag}：{url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(gz_path, "wb") as f:
        shutil.copyfileobj(r, f)
    with gzip.open(gz_path, "rb") as fi, open(target, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.chmod(target, 0o755)
    gz_path.unlink(missing_ok=True)
    log(f"mihomo 就绪：{target}")
    return target


# ──────────────────────────────────────────────────────────────
# 进程辅助
# ──────────────────────────────────────────────────────────────
def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _tail(path: Path, n: int) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "(读日志失败)"


def cleanup_stale_mihomo(pid_path: Path) -> None:
    """终止旧 mihomo 残留进程并清理 PID 文件。"""
    if not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        pid_path.unlink(missing_ok=True)
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return
    except PermissionError:
        log(f"残留 mihomo(pid={pid}) 无权限管理，请手动 pkill -f mihomo")
        return
    log(f"终止残留 mihomo(pid={pid})……")
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(0.3)
            except ProcessLookupError:
                break
        else:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    pid_path.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────
# MihomoManager 单例
# ──────────────────────────────────────────────────────────────
class MihomoManager:
    """mihomo 常驻进程生命周期 + 订阅配置加载。线程安全（RLock）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._logf: Any = None
        self._ready = False
        self._port_map: dict[str, int] = {}
        self._mapped: list[tuple[str, dict, str]] = []
        self._runtime_config = MIHOMO_DIR / "runtime.yaml"
        self._work_dir = MIHOMO_DIR / "home"
        self._log_path = MIHOMO_DIR / "mihomo.log"
        self._pid_path = MIHOMO_DIR / "mihomo.pid"
        self._port_map_path = MIHOMO_DIR / "port_map.json"

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready and self._proc is not None and self._proc.poll() is None

    @property
    def port_map(self) -> dict[str, int]:
        with self._lock:
            return dict(self._port_map)

    @property
    def mapped(self) -> list[tuple[str, dict, str]]:
        with self._lock:
            return list(self._mapped)

    def _resolve_bin(self) -> Path:
        settings = config.get_mihomo_settings()
        bin_path = Path(str(settings.get("bin_path") or "scripts/.bin/mihomo"))
        if not bin_path.is_absolute():
            bin_path = PROJECT_ROOT / bin_path
        return bin_path

    def _write_empty_config(self) -> None:
        """空配置（无 listener，仅 controller），供首次启动。"""
        settings = config.get_mihomo_settings()
        cfg = {
            "log-level": "warning",
            "mode": "direct",
            "allow-lan": False,
            "external-controller": settings.get("external_controller") or "127.0.0.1:9090",
            "proxies": [],
            "listeners": [],
        }
        if settings.get("external_controller_secret"):
            cfg["secret"] = settings["external_controller_secret"]
        self._runtime_config.parent.mkdir(parents=True, exist_ok=True)
        self._runtime_config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    def _start(self) -> bool:
        """内部启动，不持锁。失败返回 False 不抛。"""
        try:
            bin_path = ensure_mihomo_binary(self._resolve_bin())
            settings = config.get_mihomo_settings()
            port_base = int(settings.get("port_base") or 30000)
            # 端口范围预检：若 port_base 已被占（非自己），报错
            if is_port_open(port_base) and not self._is_own_port(port_base):
                log(f"端口 {port_base} 已被占用，mihomo 启动中止")
                return False
            cleanup_stale_mihomo(self._pid_path)
            if not self._runtime_config.exists():
                self._write_empty_config()
            self._work_dir.mkdir(parents=True, exist_ok=True)
            self._logf = open(self._log_path, "w")
            self._proc = subprocess.Popen(
                [str(bin_path), "-f", str(self._runtime_config), "-d", str(self._work_dir)],
                stdout=self._logf,
                stderr=subprocess.STDOUT,
            )
            self._pid_path.write_text(str(self._proc.pid))
            if self._wait_ready():
                self._ready = True
                log(f"mihomo 启动成功(pid={self._proc.pid})")
                return True
            log("mihomo 启动超时")
            self._kill_proc()
            return False
        except Exception as e:
            log(f"mihomo 启动失败：{type(e).__name__}: {e}")
            self._ready = False
            return False

    def _is_own_port(self, port: int) -> bool:
        """端口是否由当前 mihomo 占用（粗判：在 port_map 里）。"""
        return port in self._port_map.values()

    def _wait_ready(self, deadline_s: float = 20.0) -> bool:
        settings = config.get_mihomo_settings()
        ctrl = str(settings.get("external_controller") or "127.0.0.1:9090")
        host, _, port_s = ctrl.partition(":")
        try:
            ctrl_port = int(port_s)
        except ValueError:
            ctrl_port = 9090
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < deadline_s:
            if self._proc and self._proc.poll() is not None:
                log(f"mihomo 进程已退出(code={self._proc.returncode})\n{_tail(self._log_path, 30)}")
                return False
            if is_port_open(ctrl_port, host or "127.0.0.1"):
                return True
            time.sleep(0.3)
        return False

    def _kill_proc(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._logf:
            try:
                self._logf.close()
            except Exception:
                pass
            self._logf = None
        self._pid_path.unlink(missing_ok=True)

    def _stop(self) -> None:
        """内部停止，不持锁。"""
        self._ready = False
        self._kill_proc()
        log("mihomo 已停止")

    def start(self) -> bool:
        """启动 mihomo（用 runtime.yaml）。失败不抛，返回 False。"""
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return True
            return self._start()

    def stop(self) -> None:
        with self._lock:
            self._stop()

    def is_healthy(self) -> bool:
        return self.ready

    def apply_new_config(
        self, cfg: dict, port_map: dict[str, int], mapped: list[tuple[str, dict, str]]
    ) -> bool:
        """写入新配置并 stop+start 应用（首版不依赖 reload）。返回是否就绪。"""
        with self._lock:
            self._runtime_config.parent.mkdir(parents=True, exist_ok=True)
            self._runtime_config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            self._port_map = dict(port_map)
            self._mapped = list(mapped)
            self._stop()
            return self._start()

    def build_and_load(
        self, subscription_url: str, region_keywords: list[str]
    ) -> tuple[dict[str, int], list[tuple[str, dict, str]]]:
        """拉订阅→筛→分配端口(复用旧映射)→生成配置→apply_new_config。返回(port_map, mapped)。"""
        yaml_text = fetch_subscription(subscription_url)
        nodes = parse_proxies(yaml_text)
        mapped = filter_region(nodes, region_keywords)
        if not mapped:
            raise ValueError("订阅中没有命中地区关键词的节点")
        settings = config.get_mihomo_settings()
        port_base = int(settings.get("port_base") or 30000)
        range_size = int(settings.get("port_range_size") or 500)
        old_port_map = self._load_port_map()
        new_node_ids = [nid for nid, _, _ in mapped]
        port_map = allocate_ports(new_node_ids, old_port_map, port_base, range_size)
        cfg, _ = build_mihomo_config(
            mapped,
            port_map,
            str(settings.get("external_controller") or "127.0.0.1:9090"),
            str(settings.get("external_controller_secret") or ""),
        )
        self._save_port_map(port_map)
        if not self.apply_new_config(cfg, port_map, mapped):
            raise RuntimeError("mihomo 应用新配置失败")
        return port_map, mapped

    def _load_port_map(self) -> dict[str, int]:
        if self._port_map_path.exists():
            try:
                data = json.loads(self._port_map_path.read_text())
                return {str(k): int(v) for k, v in data.items()} if isinstance(data, dict) else {}
            except (json.JSONDecodeError, ValueError, OSError):
                return {}
        return {}

    def _save_port_map(self, port_map: dict[str, int]) -> None:
        self._port_map_path.parent.mkdir(parents=True, exist_ok=True)
        self._port_map_path.write_text(json.dumps(port_map))

    def get_socks5_url(self, node_id: str) -> str:
        port = self._port_map.get(node_id)
        return f"socks5://127.0.0.1:{port}" if port else ""

    def get_node_meta(self, node_id: str) -> dict | None:
        """返回某节点的元信息（原始 name、节点字段）。"""
        with self._lock:
            for nid, node, orig in self._mapped:
                if nid == node_id:
                    return {"node_id": nid, "name": orig, "node": node}
        return None


mihomo_manager = MihomoManager()
