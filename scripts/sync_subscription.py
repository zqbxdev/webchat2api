#!/usr/bin/env python3
"""阶段一脚本：从 Clash 订阅拉取美国节点 → 用 mihomo 把每个节点转成独立本地
SOCKS5 端口 → 并发用这些端口测 chatgpt.com 连通性 → 剔除超时/被挡节点，输出可用节点。

这是「订阅节点代理池」扩展（方案 C）的第一步：独立、可手动跑通，先验证技术链路，
跑通后再把逻辑沉淀进 webchat2api 项目（阶段二）。

用法（项目根目录执行）：
    uv run --with pyyaml python scripts/sync_subscription.py [选项]

产物写入 --out-dir（默认 data/subscription_us）：
    usable_socks5.txt   一行一个 socks5://127.0.0.1:<port>，可直接灌 proxy_pool
    usable_nodes.yaml   可用节点原样 Clash 配置（name 已还原，可给其它客户端加载）
    report.json         每个节点的测活详情（status/latency/error）
    mihomo.yaml         生成的 mihomo 配置（调试用）
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from curl_cffi import requests as ccreq

# ──────────────────────────────────────────────────────────────
# 默认值
# ──────────────────────────────────────────────────────────────
DEFAULT_SUB_URL = "https://slp.19930905.xyz/c/?token=87ebe4bebccf282d558c8271a95c9e36&cdk=lkdx"
DEFAULT_REGION_KEYWORDS = ["🇺🇸", "美国", "美國", "United States", "America"]
DEFAULT_TEST_URL = "https://chatgpt.com/api/auth/csrf"
DEFAULT_PORT_BASE = 30000
DEFAULT_TIMEOUT = 15.0
DEFAULT_CONCURRENCY = 16
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GH_API_LATEST = "https://api.github.com/repos/MetaCubeX/mihomo/releases/latest"


def log(msg: str) -> None:
    print(f"[sync] {msg}", flush=True)


def mask_url(url: str) -> str:
    """订阅 URL 脱敏，避免 token 完整打印到日志。"""
    if "token=" not in url:
        return url
    import re

    def _sub(m: "re.Match[str]") -> str:
        tok = m.group(1)
        if len(tok) <= 8:
            return "token=***"
        return f"token={tok[:4]}...{tok[-4:]}"

    return re.sub(r"token=([^&]+)", _sub, url)


# ──────────────────────────────────────────────────────────────
# 1. 拉取订阅
# ──────────────────────────────────────────────────────────────
def fetch_subscription(url: str, timeout: float = 30.0) -> str:
    log(f"拉取订阅：{mask_url(url)}")
    s = ccreq.Session()
    try:
        r = s.get(url, headers={"user-agent": "clash-verge/v1.0"}, timeout=timeout)
        r.raise_for_status()
        return r.text
    finally:
        s.close()


# ──────────────────────────────────────────────────────────────
# 2. 解析 + 3. 地区筛选
# ──────────────────────────────────────────────────────────────
def parse_proxies(yaml_text: str) -> list[dict]:
    """解析 Clash YAML，只取真实 proxies（丢弃 proxy-groups）。"""
    data = yaml.safe_load(yaml_text) or {}
    proxies = data.get("proxies") or []
    return [p for p in proxies if isinstance(p, dict)]


def filter_region(
    nodes: list[dict], keywords: list[str], prefix: str = "us"
) -> list[tuple[str, dict, str]]:
    """按 name 关键词筛选节点，返回 [(node_id, 重命名后的节点副本, 原始name)]。

    node_id = prefix + 节点在订阅中的原始下标，保证单次运行内唯一；原始 name 仅存档。
    注意：若订阅节点顺序/数量变化，同一下标会指向不同节点，跨日不应依赖 id 稳定性。
    """
    out = []
    for i, node in enumerate(nodes):
        orig = str(node.get("name") or f"node-{i}")
        if any(k and k in orig for k in keywords):
            node_id = f"{prefix}-{i:04d}"
            renamed = dict(node)
            renamed["name"] = node_id
            out.append((node_id, renamed, orig))
    return out


# ──────────────────────────────────────────────────────────────
# 4. 生成 mihomo 配置（proxies + listeners 多端口）
# ──────────────────────────────────────────────────────────────
def build_mihomo_config(
    mapped: list[tuple[str, dict, str]], port_base: int
) -> tuple[dict, dict[str, int]]:
    """每节点一个 socks listener，proxy 字段钉死该节点 → 直出绕过主路由。"""
    proxies: list[dict] = []
    listeners: list[dict] = []
    port_map: dict[str, int] = {}
    for idx, (node_id, node, _orig) in enumerate(mapped):
        proxies.append(node)
        port = port_base + idx
        port_map[node_id] = port
        listeners.append(
            {
                "name": f"lis-{node_id}",
                "type": "socks",
                "listen": "127.0.0.1",
                "port": port,
                "proxy": node_id,
            }
        )
    config = {
        "log-level": "warning",
        "mode": "direct",
        "allow-lan": False,
        "ipv6": False,
        "tcp-concurrent": True,
        "proxies": proxies,
        "listeners": listeners,
    }
    return config, port_map


# ──────────────────────────────────────────────────────────────
# 5. mihomo 二进制管理（自动下载）
# ──────────────────────────────────────────────────────────────
def _arch_keys() -> list[str]:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        # 优先 compatible 变体（兼容老 CPU，无 AVX 等指令集要求），其次普通版
        return ["linux-amd64-compatible", "linux-amd64"]
    if m in ("aarch64", "arm64"):
        return ["linux-arm64"]
    raise SystemExit(f"不支持的架构：{m}")


def _pick_asset_url(meta: dict, keys: list[str]) -> str:
    assets = {a["name"].lower(): a["browser_download_url"] for a in meta.get("assets", [])}
    for k in keys:
        for name, url in assets.items():
            if name.endswith(".gz") and k in name:
                return url
    raise SystemExit(f"未找到匹配架构 {keys} 的 mihomo release 资产")


def ensure_mihomo_binary(target: Path) -> Path:
    if target.exists() and os.access(target, os.X_OK):
        log(f"mihomo 二进制已存在：{target}")
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
# 6. mihomo 进程生命周期
# ──────────────────────────────────────────────────────────────
def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def start_mihomo(
    bin_path: Path, config_path: Path, work_dir: Path, log_path: Path
) -> tuple[subprocess.Popen, object]:
    work_dir.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w")
    proc = subprocess.Popen(
        [str(bin_path), "-f", str(config_path), "-d", str(work_dir)],
        stdout=logf,
        stderr=subprocess.STDOUT,
    )
    return proc, logf


def wait_mihomo_ready(
    proc: subprocess.Popen, port_map: dict[str, int], log_path: Path, deadline_s: float = 20.0
) -> None:
    if not port_map:
        return
    first_port = next(iter(port_map.values()))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < deadline_s:
        if proc.poll() is not None:
            tail = _tail(log_path, 40)
            raise SystemExit(f"mihomo 进程已退出(code={proc.returncode})。日志末尾：\n{tail}")
        if is_port_open(first_port):
            time.sleep(0.5)  # 让其余 listener 端口完成绑定，减少首测假阴性
            return
        time.sleep(0.3)
    raise SystemExit(f"mihomo 启动超时（{deadline_s:.0f}s 内端口 {first_port} 未就绪）")


def _tail(path: Path, n: int) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "(读日志失败)"


def cleanup_stale_mihomo(pid_path: Path) -> None:
    """若存在旧 mihomo PID 文件，尝试终止残留进程并清理文件。"""
    if not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        pid_path.unlink(missing_ok=True)
        return
    try:
        os.kill(pid, 0)  # 探活，不发送信号
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return
    except PermissionError:
        log(f"残留 mihomo(pid={pid}) 存在但无权限管理，请手动 pkill -f mihomo")
        return
    log(f"发现残留 mihomo(pid={pid})，正在终止……")
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
# 7. 并发测活
# ──────────────────────────────────────────────────────────────
def probe(port: int, test_url: str, timeout: float) -> dict:
    """通过指定 socks5 端口访问 chatgpt.com，收紧判据：status==200 且响应含 csrfToken。"""
    s = ccreq.Session(impersonate="edge101", verify=True, proxy=f"socks5://127.0.0.1:{port}")
    t0 = time.perf_counter()
    try:
        r = s.get(test_url, headers={"user-agent": "Mozilla/5.0"}, timeout=timeout)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        body = r.text if r.status_code == 200 else ""
        ok = r.status_code == 200 and "csrftoken" in body.lower()
        if ok:
            error = None
        elif r.status_code != 200:
            error = f"HTTP {r.status_code}"
        else:
            error = "200 but no csrfToken (likely CF challenge / non-JSON)"
        return {"ok": ok, "status": int(r.status_code), "latency_ms": latency_ms, "error": error}
    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        msg = f"{type(e).__name__}: {e}"
        return {"ok": False, "status": 0, "latency_ms": latency_ms, "error": msg[:300]}
    finally:
        s.close()


def probe_all(
    port_map: dict[str, int], test_url: str, timeout: float, workers: int
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(probe, p, test_url, timeout): nid for nid, p in port_map.items()}
        done = 0
        total = len(fut)
        for f in as_completed(fut):
            nid = fut[f]
            results[nid] = f.result()
            done += 1
            if done % 10 == 0 or done == total:
                ok = sum(1 for v in results.values() if v["ok"])
                log(f"测活进度 {done}/{total}，可用 {ok}")
    return results


# ──────────────────────────────────────────────────────────────
# 8. 输出产物
# ──────────────────────────────────────────────────────────────
def emit(
    out_dir: Path,
    mapped: list[tuple[str, dict, str]],
    port_map: dict[str, int],
    results: dict[str, dict],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # node_id -> 原始信息
    meta = {nid: {"orig_name": orig, "node": node} for nid, node, orig in mapped}

    usable_ids = [nid for nid, _, _ in mapped if results.get(nid, {}).get("ok")]

    # usable_socks5.txt
    socks5_lines = [f"socks5://127.0.0.1:{port_map[nid]}" for nid in usable_ids]
    (out_dir / "usable_socks5.txt").write_text("\n".join(socks5_lines) + ("\n" if socks5_lines else ""))

    # usable_nodes.yaml（name 还原为原始名）
    usable_proxies = []
    for nid, _, _ in mapped:
        if results.get(nid, {}).get("ok"):
            node = dict(meta[nid]["node"])
            node["name"] = meta[nid]["orig_name"]
            usable_proxies.append(node)
    (out_dir / "usable_nodes.yaml").write_text(
        yaml.safe_dump({"proxies": usable_proxies}, allow_unicode=True, sort_keys=False)
    )

    # report.json（全部节点，含失败原因）
    report = []
    for nid, _node, orig in mapped:
        r = results.get(nid, {"ok": False, "status": 0, "latency_ms": 0, "error": "not tested"})
        node = meta[nid]["node"]
        report.append(
            {
                "node_id": nid,
                "name": orig,
                "type": node.get("type"),
                "server": node.get("server"),
                "port": node.get("port"),
                "local_port": port_map.get(nid),
                "usable": r["ok"],
                "status": r["status"],
                "latency_ms": r["latency_ms"],
                "error": r["error"],
            }
        )
    report.sort(key=lambda x: (not x["usable"], x["latency_ms"] if x["usable"] else 0))
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # mihomo.yaml 由 main() 在 mihomo 启动前写入，此处不重复

    return {
        "total": len(mapped),
        "usable": len(usable_ids),
        "socks5_file": str(out_dir / "usable_socks5.txt"),
    }


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="拉取订阅美国节点并测活（mihomo 多端口）")
    p.add_argument("--url", default=DEFAULT_SUB_URL, help="订阅 URL")
    p.add_argument(
        "--region",
        default=",".join(DEFAULT_REGION_KEYWORDS),
        help="地区关键词，逗号分隔（默认美国）",
    )
    p.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE, help="本地 socks5 起始端口")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="单节点测活超时(秒)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="并发数")
    p.add_argument("--test-url", default=DEFAULT_TEST_URL, help="测活目标 URL")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "subscription_us"), help="产物目录")
    p.add_argument(
        "--mihomo-bin",
        default=str(PROJECT_ROOT / "scripts" / ".bin" / "mihomo"),
        help="mihomo 二进制路径（不存在自动下载）",
    )
    p.add_argument(
        "--keep-mihomo", action="store_true", help="跑完保留 mihomo 进程（默认退出即关）"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    keywords = [k.strip() for k in args.region.split(",") if k.strip()]
    out_dir = Path(args.out_dir)
    bin_path = Path(args.mihomo_bin)
    config_path = out_dir / "mihomo.yaml"
    work_dir = out_dir / "mihomo_home"
    log_path = out_dir / "mihomo.log"
    pid_path = out_dir / "mihomo.pid"

    # 启动前清理上次 --keep-mihomo 残留，再检查端口占用
    cleanup_stale_mihomo(pid_path)
    if is_port_open(args.port_base):
        raise SystemExit(
            f"端口 {args.port_base} 已被占用。请先清理旧 mihomo：pkill -f mihomo 或更换 --port-base"
        )

    yaml_text = fetch_subscription(args.url)
    nodes = parse_proxies(yaml_text)
    log(f"解析到真实节点 {len(nodes)} 个")

    mapped = filter_region(nodes, keywords)
    log(f"命中地区({keywords})节点 {len(mapped)} 个")
    if not mapped:
        log("没有命中任何节点，结束。")
        return 1

    config, port_map = build_mihomo_config(mapped, args.port_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    ensure_mihomo_binary(bin_path)

    log("启动 mihomo……")
    proc, logf = start_mihomo(bin_path, config_path, work_dir, log_path)
    pid_path.write_text(str(proc.pid))
    try:
        wait_mihomo_ready(proc, port_map, log_path)
        log(f"mihomo 就绪(pid={proc.pid})，开始测活（目标 {args.test_url}，并发 {args.concurrency}）")
        results = probe_all(port_map, args.test_url, args.timeout, args.concurrency)
    finally:
        if not args.keep_mihomo:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            pid_path.unlink(missing_ok=True)
            logf.close()
            log("已关闭 mihomo")
        else:
            logf.close()
            log(f"保留 mihomo(pid={proc.pid})，端口 {args.port_base}..{args.port_base + len(mapped) - 1}（PID 文件 {pid_path}）")

    summary = emit(out_dir, mapped, port_map, results)
    log(
        f"完成：{summary['usable']}/{summary['total']} 可用 → {summary['socks5_file']}"
    )
    log("提示：若要看每节点失败原因，查看 report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
