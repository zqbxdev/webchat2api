"use client";

import { useEffect, useRef, useState } from "react";
import {
  LoaderCircle,
  Network,
  Plus,
  RefreshCw,
  Shuffle,
  Trash2,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import {
  addSubscription,
  assignProxyPool,
  clearProxyPoolAssignments,
  deleteProxyPool,
  deleteSubscription,
  fetchProxyPool,
  fetchSubscriptions,
  getSyncStatus,
  importProxyPool,
  syncProxyPool,
  type ProxyPoolItem,
  type SubscriptionSource,
} from "@/lib/api";

export function ProxyPoolCard() {
  const didLoadRef = useRef(false);
  const [items, setItems] = useState<ProxyPoolItem[]>([]);
  const [subs, setSubs] = useState<SubscriptionSource[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isImporting, setIsImporting] = useState(false);
  const [isAssigning, setIsAssigning] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [showImport, setShowImport] = useState(false);
  const [importText, setImportText] = useState("");

  // 订阅源
  const [showAddSub, setShowAddSub] = useState(false);
  const [subForm, setSubForm] = useState({ name: "", url: "", region_keywords: "🇺🇸,美国,美國,United States,America" });
  const [isAddingSub, setIsAddingSub] = useState(false);
  const [syncTaskId, setSyncTaskId] = useState<string | null>(null);
  const [syncStatus, setSyncStatus] = useState<string>("");

  const load = async () => {
    setIsLoading(true);
    try {
      const data = await fetchProxyPool();
      setItems(data.items);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "加载代理池失败");
    } finally {
      setIsLoading(false);
    }
  };

  const loadSubs = async () => {
    try {
      const data = await fetchSubscriptions();
      setSubs(data.items);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "加载订阅源失败");
    }
  };

  useEffect(() => {
    if (didLoadRef.current) return;
    didLoadRef.current = true;
    void load();
    void loadSubs();
  }, []);

  // 同步状态轮询
  useEffect(() => {
    if (!syncTaskId) return;
    const timer = window.setInterval(async () => {
      try {
        const s = await getSyncStatus(syncTaskId);
        setSyncStatus(s.status);
        if (s.status === "done" || s.status === "error" || s.status === "skipped") {
          window.clearInterval(timer);
          setSyncTaskId(null);
          if (s.status === "done") {
            const r = s.result;
            toast.success(`同步完成：可用 ${r?.usable ?? 0}/${r?.total ?? 0}，重分配 ${r?.reassigned ?? 0} 账号`);
            void load();
            void loadSubs();
          } else {
            toast.error(`同步${s.status === "skipped" ? "跳过（另一任务在跑）" : "失败"}：${s.error ?? ""}`);
          }
        }
      } catch (error) {
        window.clearInterval(timer);
        setSyncTaskId(null);
        toast.error(error instanceof Error ? error.message : "查询同步状态失败");
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [syncTaskId]);

  const handleImport = async () => {
    const text = importText.trim();
    if (!text) {
      toast.error("请输入代理列表");
      return;
    }
    setIsImporting(true);
    try {
      const data = await importProxyPool(text);
      setItems(data.items);
      setImportText("");
      setShowImport(false);
      toast.success(`导入完成：新增 ${data.added ?? 0} 个，跳过 ${data.skipped ?? 0} 个`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "导入失败");
    } finally {
      setIsImporting(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      const data = await deleteProxyPool([id]);
      setItems(data.items);
      toast.success("已删除");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "删除失败");
    }
  };

  const handleClearAll = async () => {
    try {
      const data = await deleteProxyPool([]);
      setItems(data.items);
      toast.success(`已清空代理池（删除 ${data.removed ?? 0} 个）`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "清空失败");
    }
  };

  const handleAssign = async () => {
    setIsAssigning(true);
    try {
      const data = await assignProxyPool();
      if (data.error) {
        toast.error(data.error);
      } else {
        toast.success(`已分配 ${data.assigned} 个账号（${data.total_proxies} 个代理轮转分配给 ${data.total_accounts} 个账号）`);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "分配失败");
    } finally {
      setIsAssigning(false);
    }
  };

  const handleClearAssignments = async () => {
    setIsClearing(true);
    try {
      const data = await clearProxyPoolAssignments();
      toast.success(`已清除 ${data.cleared} 个账号的代理分配`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "清除失败");
    } finally {
      setIsClearing(false);
    }
  };

  const handleAddSub = async () => {
    if (!subForm.url.trim()) {
      toast.error("请输入订阅 URL");
      return;
    }
    setIsAddingSub(true);
    try {
      const data = await addSubscription(subForm.name, subForm.url, subForm.region_keywords);
      setSubs(data.items);
      setSubForm({ ...subForm, name: "", url: "" });
      setShowAddSub(false);
      toast.success("订阅源已添加");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "添加失败");
    } finally {
      setIsAddingSub(false);
    }
  };

  const handleDeleteSub = async (id: string) => {
    try {
      const data = await deleteSubscription(id);
      setSubs(data.items);
      void load();
      toast.success("订阅源已删除（含其代理项）");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "删除失败");
    }
  };

  const handleSync = async () => {
    try {
      const r = await syncProxyPool();
      setSyncTaskId(r.task_id);
      setSyncStatus("pending");
      toast.info("同步已启动（拉订阅 + 测活 + 更新池，约 1-2 分钟）");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "启动同步失败");
    }
  };

  const syncing = !!syncTaskId;

  return (
    <Card className="rounded-2xl border-white/80 bg-white/90 shadow-sm">
      <CardContent className="space-y-6 p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-xl bg-stone-100">
              <Network className="size-5 text-stone-600" />
            </div>
            <div>
              <h2 className="text-lg font-semibold tracking-tight">代理池</h2>
              <p className="text-sm text-stone-500">
                批量导入代理或订阅节点，自动分配给账号，每个账号使用独立出口。
              </p>
            </div>
          </div>
          <Badge variant={items.length > 0 ? "success" : "secondary"} className="w-fit rounded-md px-2.5 py-1">
            {items.length > 0 ? `${items.length} 个代理` : "未配置"}
          </Badge>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-10">
            <LoaderCircle className="size-5 animate-spin text-stone-400" />
          </div>
        ) : (
          <>
            {/* Action buttons */}
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" className="h-9 rounded-xl border-stone-200 bg-white px-4 text-stone-700" onClick={() => setShowImport(!showImport)}>
                <Plus className="size-4" />
                导入代理
              </Button>
              <Button variant="outline" className="h-9 rounded-xl border-stone-200 bg-white px-4 text-stone-700" onClick={() => setShowAddSub(!showAddSub)}>
                <Plus className="size-4" />
                添加订阅源
              </Button>
              <Button className="h-9 rounded-xl bg-stone-950 px-4 text-white hover:bg-stone-800" onClick={() => void handleAssign()} disabled={isAssigning || items.length === 0}>
                {isAssigning ? <LoaderCircle className="size-4 animate-spin" /> : <Shuffle className="size-4" />}
                自动分配
              </Button>
              <Button variant="outline" className="h-9 rounded-xl border-stone-200 bg-white px-4 text-stone-700" onClick={() => void handleClearAssignments()} disabled={isClearing}>
                {isClearing ? <LoaderCircle className="size-4 animate-spin" /> : <XCircle className="size-4" />}
                清除分配
              </Button>
              {items.length > 0 && (
                <Button variant="outline" className="h-9 rounded-xl border-rose-200 bg-white px-4 text-rose-600 hover:bg-rose-50" onClick={() => void handleClearAll()}>
                  <Trash2 className="size-4" />
                  清空代理池
                </Button>
              )}
            </div>

            {/* Import area */}
            {showImport && (
              <div className="space-y-3 rounded-xl border border-stone-200 bg-stone-50 p-4">
                <label className="text-sm font-medium text-stone-700">
                  批量导入（每行一个，格式：IP:端口:用户名:密码）
                </label>
                <Textarea value={importText} onChange={(e) => setImportText(e.target.value)} placeholder={"1.2.3.4:8080:user1:pass1\n5.6.7.8:1080:user2:pass2"} className="min-h-[120px] rounded-xl border-stone-200 bg-white font-mono text-xs" />
                <div className="flex gap-2">
                  <Button className="h-9 rounded-xl bg-stone-950 px-4 text-white hover:bg-stone-800" onClick={() => void handleImport()} disabled={isImporting}>
                    {isImporting ? <LoaderCircle className="size-4 animate-spin" /> : <Plus className="size-4" />}
                    确认导入
                  </Button>
                  <Button variant="outline" className="h-9 rounded-xl border-stone-200 bg-white px-4 text-stone-700" onClick={() => { setShowImport(false); setImportText(""); }}>
                    取消
                  </Button>
                </div>
              </div>
            )}

            {/* Add subscription area */}
            {showAddSub && (
              <div className="space-y-3 rounded-xl border border-stone-200 bg-stone-50 p-4">
                <label className="text-sm font-medium text-stone-700">添加订阅源（Clash YAML 订阅链接）</label>
                <input value={subForm.name} onChange={(e) => setSubForm({ ...subForm, name: e.target.value })} placeholder="名称（如 linux.do）" className="w-full rounded-xl border border-stone-200 bg-white px-3 py-2 text-sm" />
                <input value={subForm.url} onChange={(e) => setSubForm({ ...subForm, url: e.target.value })} placeholder="订阅 URL" className="w-full rounded-xl border border-stone-200 bg-white px-3 py-2 font-mono text-xs" />
                <input value={subForm.region_keywords} onChange={(e) => setSubForm({ ...subForm, region_keywords: e.target.value })} placeholder="地区关键词（逗号分隔，如 🇺🇸,美国）" className="w-full rounded-xl border border-stone-200 bg-white px-3 py-2 text-xs" />
                <div className="flex gap-2">
                  <Button className="h-9 rounded-xl bg-stone-950 px-4 text-white hover:bg-stone-800" onClick={() => void handleAddSub()} disabled={isAddingSub}>
                    {isAddingSub ? <LoaderCircle className="size-4 animate-spin" /> : <Plus className="size-4" />}
                    确认添加
                  </Button>
                  <Button variant="outline" className="h-9 rounded-xl border-stone-200 bg-white px-4 text-stone-700" onClick={() => setShowAddSub(false)}>
                    取消
                  </Button>
                </div>
              </div>
            )}

            {/* Subscriptions list */}
            {subs.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium text-stone-700">订阅源</div>
                  <Button className="h-8 rounded-xl bg-stone-950 px-3 text-xs text-white hover:bg-stone-800" onClick={() => void handleSync()} disabled={syncing}>
                    {syncing ? <LoaderCircle className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
                    {syncing ? `同步中(${syncStatus})` : "立即同步"}
                  </Button>
                </div>
                <div className="max-h-[200px] overflow-y-auto rounded-xl border border-stone-200 bg-white">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 border-b border-stone-100 bg-stone-50">
                      <tr>
                        <th className="px-3 py-2 text-left font-medium text-stone-600">名称</th>
                        <th className="px-3 py-2 text-left font-medium text-stone-600">地区</th>
                        <th className="px-3 py-2 text-left font-medium text-stone-600">上次同步</th>
                        <th className="px-3 py-2 text-right font-medium text-stone-600">操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      {subs.map((s) => (
                        <tr key={s.id} className="border-b border-stone-50 last:border-0">
                          <td className="px-3 py-2 text-xs text-stone-800">{s.name}</td>
                          <td className="px-3 py-2 text-xs text-stone-500">{s.region_keywords || "-"}</td>
                          <td className="px-3 py-2 text-xs text-stone-500">
                            {s.last_sync_at ? `${s.last_usable}/${s.last_total} @ ${s.last_sync_at.slice(5,16)}` : "未同步"}
                            {s.last_error && <span className="ml-1 text-rose-500">⚠</span>}
                          </td>
                          <td className="px-3 py-2 text-right">
                            <button className="text-stone-400 hover:text-rose-500" onClick={() => void handleDeleteSub(s.id)}>
                              <Trash2 className="size-3.5" />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Proxy list */}
            {items.length > 0 && (
              <div className="space-y-2">
                <div className="text-sm font-medium text-stone-700">代理列表</div>
                <div className="max-h-[300px] overflow-y-auto rounded-xl border border-stone-200 bg-white">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 border-b border-stone-100 bg-stone-50">
                      <tr>
                        <th className="px-4 py-2 text-left font-medium text-stone-600">地址</th>
                        <th className="px-4 py-2 text-left font-medium text-stone-600">来源</th>
                        <th className="px-4 py-2 text-left font-medium text-stone-600">协议</th>
                        <th className="px-4 py-2 text-left font-medium text-stone-600">健康</th>
                        <th className="px-4 py-2 text-right font-medium text-stone-600">操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      {items.map((item) => {
                        const isSub = item.source === "subscription";
                        return (
                          <tr key={item.id} className="border-b border-stone-50 last:border-0">
                            <td className="px-4 py-2 font-mono text-xs text-stone-800">
                              {isSub ? `socks5://127.0.0.1:${item.local_port ?? item.port}` : `${item.host}:${item.port}`}
                            </td>
                            <td className="px-4 py-2">
                              <Badge variant={isSub ? "default" : "secondary"} className="rounded px-1.5 py-0.5 text-[10px]">
                                {isSub ? "订阅" : "手动"}
                              </Badge>
                            </td>
                            <td className="px-4 py-2 text-xs text-stone-500">{item.protocol || (isSub ? "-" : "http")}</td>
                            <td className="px-4 py-2">
                              {isSub ? (
                                <Badge variant={item.health === "ok" ? "success" : item.health === "down" ? "danger" : "secondary"} className="rounded px-1.5 py-0.5 text-[10px]">
                                  {item.health === "ok" ? `${item.latency_ms ?? 0}ms` : (item.health ?? "unknown")}
                                </Badge>
                              ) : (
                                <span className="text-xs text-stone-400">-</span>
                              )}
                            </td>
                            <td className="px-4 py-2 text-right">
                              <button className="text-stone-400 hover:text-rose-500" onClick={() => void handleDelete(item.id)}>
                                <Trash2 className="size-3.5" />
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
