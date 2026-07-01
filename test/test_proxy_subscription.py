"""订阅代理池扩展单元测试（阶段二 M3/M4）。

覆盖 mihomo_manager 配置生成（filter_region 去重/稳定、allocate_ports 复用/新分配、
build_mihomo_config listener-proxy 一致性）+ proxy_pool_service 订阅方法
（make_subscription_item、replace/delete_subscription_items、reassign_invalid_accounts）。
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── M3: mihomo_manager 纯函数 ──
def test_filter_region_dedup_and_stable():
    from services.mihomo_manager import filter_region

    nodes = [
        {"name": "🇺🇸 美国 | No.1", "type": "vless", "server": "1.1.1.1", "port": 443, "uuid": "abc"},
        {"name": "🇺🇸 美国 | 重复", "type": "vless", "server": "1.1.1.1", "port": 443, "uuid": "abc"},
        {"name": "🇯🇵 日本", "type": "vless", "server": "2.2.2.2", "port": 443, "uuid": "def"},
    ]
    m1 = filter_region(nodes, ["🇺🇸", "美国"])
    m2 = filter_region(nodes, ["🇺🇸", "美国"])
    assert len(m1) == 1, "重复节点应去重"
    assert m1[0][0] == m2[0][0], "node_id 跨调用稳定"
    assert m1[0][0].startswith("us-")
    assert m1[0][2] == "🇺🇸 美国 | No.1", "保留原始 name"


def test_filter_region_empty_when_no_match():
    from services.mihomo_manager import filter_region

    nodes = [{"name": "🇯🇵 日本", "type": "vless", "server": "1.1.1.1", "port": 443}]
    assert filter_region(nodes, ["🇺🇸"]) == []


def test_allocate_ports_reuse_and_new():
    from services.mihomo_manager import allocate_ports

    old = {"us-aaa": 30000, "us-bbb": 30001}
    new = allocate_ports(["us-aaa", "us-ccc"], old, 30000, 500)
    assert new["us-aaa"] == 30000, "旧节点复用端口"
    assert new["us-ccc"] == 30002, "新节点分配未用端口(避开 30000/30001)"
    assert "us-bbb" not in new, "删除的节点不残留"


def test_allocate_ports_returns_deleted_node_port_after_save():
    """已删除节点的端口在 port_map 清理后可被新节点复用。"""
    from services.mihomo_manager import allocate_ports

    # 模拟 port_map 已清理（只含现有节点）
    old = {"us-aaa": 30000}
    new = allocate_ports(["us-aaa", "us-bbb"], old, 30000, 500)
    assert new["us-bbb"] == 30001


def test_build_mihomo_config_listener_proxy_match():
    from services.mihomo_manager import build_mihomo_config, filter_region

    nodes = [
        {"name": "🇺🇸 美国 | A", "type": "vless", "server": "1.1.1.1", "port": 443, "uuid": "a"},
        {"name": "🇺🇸 美国 | B", "type": "vless", "server": "2.2.2.2", "port": 8443, "uuid": "b"},
    ]
    mapped = filter_region(nodes, ["🇺🇸"])
    port_map = {nid: 30000 + i for i, (nid, _, _) in enumerate(mapped)}
    cfg, _ = build_mihomo_config(mapped, port_map)
    proxy_names = {p["name"] for p in cfg["proxies"]}
    listener_proxies = {l["proxy"] for l in cfg["listeners"]}
    assert listener_proxies == proxy_names, "每个 listener.proxy 必须对应一个 proxy"
    assert all(l["type"] == "socks" for l in cfg["listeners"])
    assert all(l["listen"] == "127.0.0.1" for l in cfg["listeners"])
    assert cfg["external-controller"]


# ── M4: proxy_pool_service 订阅方法 ──
def _make_svc():
    from services.storage.factory import create_storage_backend
    from services.proxy_pool_service import ProxyPoolService

    tmp = Path(tempfile.mkdtemp())
    return ProxyPoolService(create_storage_backend(tmp / "accounts.json"))


def test_make_subscription_item():
    from services.proxy_pool_service import make_subscription_item

    it = make_subscription_item("sub1", "us-abc", 30000, "vless", "US", {"server": "1.1.1.1"}, 100)
    assert it["url"] == "socks5://127.0.0.1:30000"
    assert it["source"] == "subscription"
    assert it["subscription_id"] == "sub1"
    assert it["node_id"] == "us-abc"
    assert it["health"] == "ok"
    assert it["local_port"] == 30000


def test_replace_and_delete_subscription_items():
    from services.proxy_pool_service import make_subscription_item

    svc = _make_svc()
    it1 = make_subscription_item("sub1", "us-a", 30000, "vless", "US", {}, 100)
    r = svc.add_subscription_items("sub1", [it1])
    assert r["added"] == 1 and len(svc.list_items()) == 1

    it2 = make_subscription_item("sub1", "us-b", 30001, "vless", "US", {}, 200)
    r = svc.replace_subscription_items("sub1", [it2])
    assert r["removed"] == 1 and r["added"] == 1
    assert len(svc.list_items()) == 1
    assert svc.list_items()[0]["node_id"] == "us-b"

    r = svc.delete_subscription_items("sub1")
    assert r["removed"] == 1 and len(svc.list_items()) == 0


def test_reassign_invalid_accounts_only_invalid():
    import services.account_service as am
    from services.proxy_pool_service import make_subscription_item

    class FakeAS:
        def __init__(self):
            self.accounts = []
            self.updates = []

        def list_accounts(self):
            return self.accounts

        def update_account(self, t, d, provider=None):
            self.updates.append(t)

    am.account_service = FakeAS()
    svc = _make_svc()
    # usable item at 30002
    svc.add_subscription_items("sub1", [make_subscription_item("sub1", "us-ok", 30002, "vless", "US", {}, 50)])
    am.account_service.accounts = [
        {"access_token": "tok1", "proxy": "socks5://127.0.0.1:30002", "provider": "gpt"},  # 有效
        {"access_token": "tok2", "proxy": "", "provider": "gpt"},  # 失效(空)
        {"access_token": "tok3", "proxy": "socks5://127.0.0.1:99999", "provider": "gpt"},  # 失效(不在pool)
    ]
    r = svc.reassign_invalid_accounts()
    assert r["reassigned"] == 2, r
    assert "tok1" not in am.account_service.updates, "有效账号不该被重分配"
    assert "tok2" in am.account_service.updates
    assert "tok3" in am.account_service.updates
