"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ComponentProps } from "react";
import {
  Ban,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  CircleOff,
  Download,
  LoaderCircle,
  Pencil,
  RefreshCw,
  Search,
  Trash2,
  UserRound,
} from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  deleteAccounts,
  deleteLimitedAccounts,
  exportAccounts,
  fetchAccounts,
  refreshAccounts,
  updateAccount,
  validateAccounts,
  type Account,
  type AccountDeleteIdentifier,
  type AccountDeletePayload,
  type AccountExportProvider,
  type AccountSelectionPayload,
  type AccountStatus,
} from "@/lib/api";
import { useAuthGuard } from "@/lib/use-auth-guard";
import { cn } from "@/lib/utils";
import {
  accountProviderDefinitions,
  accountProviderSupportsTokenExport,
  getAccountProviderDefinition,
  getAccountProviderLabel,
  isProviderAccount,
  normalizeAccountProvider,
} from "@/providers/registry";
import type { AccountProviderDefinition, ProviderId } from "@/providers/types";

import { AccountImportDialog } from "./components/account-import-dialog";

const accountStatusOptions: { label: string; value: AccountStatus | "all" }[] = [
  { label: "全部状态", value: "all" },
  { label: "正常", value: "正常" },
  { label: "限流", value: "限流" },
  { label: "异常", value: "异常" },
  { label: "禁用", value: "禁用" },
];

const statusMeta: Record<
  AccountStatus,
  {
    icon: typeof CheckCircle2;
    badge: ComponentProps<typeof Badge>["variant"];
  }
> = {
  正常: { icon: CheckCircle2, badge: "success" },
  限流: { icon: CircleAlert, badge: "warning" },
  异常: { icon: CircleOff, badge: "danger" },
  禁用: { icon: Ban, badge: "secondary" },
};

const metricCards = [
  { key: "total", label: "账户总数", color: "text-stone-900", icon: UserRound },
  { key: "active", label: "正常账户", color: "text-emerald-600", icon: CheckCircle2 },
  { key: "limited", label: "限流账户", color: "text-orange-500", icon: CircleAlert },
  { key: "abnormal", label: "异常账户", color: "text-rose-500", icon: CircleOff },
  { key: "disabled", label: "禁用账户", color: "text-stone-500", icon: Ban },
  {
    key: "quota",
    label: getAccountProviderDefinition("gpt").quota.metricLabel ?? "GPT 图像额度",
    color: "text-amber-600",
    icon: RefreshCw,
  },
] as const;

type ProviderFilters = Record<ProviderId, { query: string; type: string; status: AccountStatus | "all" }>;
type ProviderPages = Record<ProviderId, number>;
type ProviderSelection = Record<ProviderId, string[]>;

const defaultProviderFilters = accountProviderDefinitions.reduce((acc, provider) => {
  acc[provider.id] = { query: "", type: "all", status: "all" };
  return acc;
}, {} as ProviderFilters);

const defaultProviderPages = accountProviderDefinitions.reduce((acc, provider) => {
  acc[provider.id] = 1;
  return acc;
}, {} as ProviderPages);

const defaultProviderSelection = accountProviderDefinitions.reduce((acc, provider) => {
  acc[provider.id] = [];
  return acc;
}, {} as ProviderSelection);

function isGeminiAccount(account: Account) {
  return isProviderAccount(account, "gemini");
}

function isUnlimitedImageQuotaAccount(account: Account) {
  const providerDefinition = getAccountProviderDefinition(account.provider);
  return providerDefinition.quota.applicable && providerDefinition.quota.unlimitedTypes.includes(account.type);
}

function imageQuotaUnknown(account: Account) {
  return Boolean(account.image_quota_unknown);
}

function formatCompact(value: number) {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}k`;
  }
  return String(value);
}

function formatQuota(account: Account) {
  const providerDefinition = getAccountProviderDefinition(account.provider);
  if (!providerDefinition.quota.applicable) {
    return providerDefinition.quota.unavailableLabel;
  }
  if (isUnlimitedImageQuotaAccount(account)) {
    return "∞";
  }
  if (imageQuotaUnknown(account)) {
    return "未知";
  }
  return String(Math.max(0, account.quota));
}

function formatRestoreAt(value?: string | null) {
  if (!value) {
    return { absolute: "—", relative: "" };
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return { absolute: value, relative: "" };
  }

  const diffMs = Math.max(0, date.getTime() - Date.now());
  const totalHours = Math.ceil(diffMs / (1000 * 60 * 60));
  const days = Math.floor(totalHours / 24);
  const hours = totalHours % 24;
  const relative = diffMs > 0 ? `剩余 ${days}d ${hours}h` : "已到恢复时间";

  const pad = (num: number) => String(num).padStart(2, "0");
  const absolute = `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;

  return { absolute, relative };
}

function formatQuotaSummary(accounts: Account[]) {
  const availableQuotaAccounts = accounts.filter(
    (account) => normalizeAccountProvider(account.provider) === "gpt" && account.status === "正常",
  );
  if (availableQuotaAccounts.length === 0) {
    return "—";
  }
  if (availableQuotaAccounts.some(isUnlimitedImageQuotaAccount)) {
    return "∞";
  }
  if (availableQuotaAccounts.some(imageQuotaUnknown)) {
    return "未知";
  }
  return formatCompact(availableQuotaAccounts.reduce((sum, account) => sum + Math.max(0, account.quota), 0));
}

function renderPrivacyEmail(email?: string | null) {
  const value = String(email || "").trim();
  if (!value) {
    return <span>—</span>;
  }
  const atIndex = value.indexOf("@");
  if (atIndex < 0) {
    return <span className="transition duration-150 blur-sm hover:blur-none">{value}</span>;
  }
  const localPart = value.slice(0, atIndex + 1);
  const domain = value.slice(atIndex + 1);
  return (
    <span className="group inline-flex max-w-full items-center">
      <span className="truncate">{localPart}</span>
      <span className="truncate transition duration-150 blur-sm group-hover:blur-none">{domain}</span>
    </span>
  );
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function displayAccountType(account: Account) {
  return account.type || "Free";
}

function displayAccountProvider(account: Account) {
  return getAccountProviderLabel(account.provider);
}

function accountToken(account: Account) {
  return typeof account.access_token === "string" ? account.access_token.trim() : "";
}

function accountProviderId(account: Account): ProviderId {
  const provider = normalizeAccountProvider(account.provider);
  return accountProviderDefinitions.some((item) => item.id === provider) ? (provider as ProviderId) : "gpt";
}

function accountRowKey(account: Account, index: number) {
  const provider = accountProviderId(account);
  const token = accountToken(account);
  if (token) return `${provider}:token:${token}`;
  const deleteIdentifier = accountDeleteIdentifier(account);
  if (deleteIdentifier?.account_id) return `${provider}:account:${deleteIdentifier.account_id}`;
  if (deleteIdentifier?.row_id) return `${provider}:row:${deleteIdentifier.row_id}`;
  return `${provider}:sanitized-${account.email ?? ""}-${account.type ?? ""}-${index}`;
}

function maskAccountToken(token: string) {
  const normalized = token.trim();
  if (!normalized) return "";
  if (normalized.length <= 12) return "token hidden";
  return `${normalized.slice(0, 6)}...${normalized.slice(-4)}`;
}

function accountTokenDisplay(account: Account) {
  const token = accountToken(account);
  if (token) return maskAccountToken(token);
  const providerDefinition = getAccountProviderDefinition(account.provider);
  return !token && isGeminiAccount(account) && account.has_gemini_session ? providerDefinition.tokenHiddenLabel : "凭据已隐藏";
}

function accountTokens(accounts: Account[]) {
  return accounts.map(accountToken).filter(Boolean);
}

type AccountDeleteCandidate = AccountDeleteIdentifier;

function accountDeleteIdentifier(account: Account): AccountDeleteCandidate | null {
  const accountId = typeof account.account_id === "string" ? account.account_id.trim() : "";
  if (accountId) return { account_id: accountId };
  const rowId = typeof account.row_id === "string" ? account.row_id.trim() : "";
  return rowId ? { row_id: rowId } : null;
}

function accountSelectionPayload(accounts: Account[]): AccountSelectionPayload {
  return {
    tokens: accountTokens(accounts),
    identifiers: accounts.map(accountDeleteIdentifier).filter((item): item is AccountDeleteCandidate => item !== null),
  };
}

function accountDeletePayloads(accounts: Account[]): AccountDeletePayload {
  return accountSelectionPayload(accounts);
}

function accountSelectionCount(payload: AccountSelectionPayload) {
  return payload.tokens.length + payload.identifiers.length;
}

function validationErrorMessage(error: string | { error?: string; message?: string } | undefined) {
  if (typeof error === "string") return error;
  return error?.error ?? error?.message ?? "";
}

function accountHasDeleteIdentifier(account: Account) {
  return Boolean(accountToken(account) || accountDeleteIdentifier(account));
}

function keepProviderSelection(selection: ProviderSelection, items: Account[]) {
  const keysByProvider = accountProviderDefinitions.reduce((acc, provider) => {
    acc[provider.id] = new Set<string>();
    return acc;
  }, {} as Record<ProviderId, Set<string>>);

  items.forEach((item, index) => {
    keysByProvider[accountProviderId(item)].add(accountRowKey(item, index));
  });

  return accountProviderDefinitions.reduce((acc, provider) => {
    acc[provider.id] = selection[provider.id].filter((id) => keysByProvider[provider.id].has(id));
    return acc;
  }, {} as ProviderSelection);
}

function mergeProviderAccounts(current: Account[], provider: ProviderId, nextItems: Account[]) {
  return [
    ...current.filter((account) => !isProviderAccount(account, provider)),
    ...nextItems.map((item) => ({ ...item, provider: normalizeAccountProvider(item.provider) || provider })),
  ];
}

function accountExportProviderLabel(provider: AccountExportProvider) {
  return getAccountProviderDefinition(provider).label;
}

function ProviderAccountSection({
  provider,
  accounts,
  filteredAccounts,
  currentRows,
  selectedIds,
  filters,
  page,
  pageSize,
  pageCount,
  safePage,
  startIndex,
  isLoading,
  isRefreshing,
  isValidating,
  isDeleting,
  isExporting,
  isUpdating,
  onFilterChange,
  onPageChange,
  onSelectChange,
  onRefresh,
  onValidate,
  onDelete,
  onDeleteLimited,
  onExport,
  onEdit,
}: {
  provider: AccountProviderDefinition;
  accounts: Account[];
  filteredAccounts: Account[];
  currentRows: Account[];
  selectedIds: string[];
  filters: ProviderFilters[ProviderId];
  page: number;
  pageSize: string;
  pageCount: number;
  safePage: number;
  startIndex: number;
  isLoading: boolean;
  isRefreshing: boolean;
  isValidating: boolean;
  isDeleting: boolean;
  isExporting: boolean;
  isUpdating: boolean;
  onFilterChange: (next: Partial<ProviderFilters[ProviderId]>) => void;
  onPageChange: (page: number) => void;
  onSelectChange: (ids: string[]) => void;
  onRefresh: (payload: AccountSelectionPayload) => void;
  onValidate: (payload: AccountSelectionPayload) => void;
  onDelete: (payload: AccountDeletePayload) => void;
  onDeleteLimited: () => void;
  onExport: (payload: AccountSelectionPayload) => void;
  onEdit: (account: Account) => void;
}) {
  const selectedSet = new Set(selectedIds);
  const selectedRows = accounts.filter((account, index) => selectedSet.has(accountRowKey(account, index)));
  const selectedPayloads = accountSelectionPayload(selectedRows);
  const selectedTokens = selectedPayloads.tokens;
  const selectedRefreshPayload = provider.refresh.enabled ? selectedPayloads : { tokens: [], identifiers: [] };
  const selectedRefreshCount = accountSelectionCount(selectedRefreshPayload);
  const canValidateProvider = provider.id === "grok";
  const selectedValidateCount = canValidateProvider ? accountSelectionCount(selectedPayloads) : 0;
  const selectedDeleteCount = accountSelectionCount(selectedPayloads);
  const limitedTokens = accountTokens(accounts.filter((item) => item.status === "限流"));
  const typeOptions = [
    { label: "全部计划/池", value: "all" },
    ...Array.from(new Set(accounts.map(displayAccountType))).map((type) => ({ label: type, value: type })),
  ];
  const allCurrentSelected =
    currentRows.length > 0 && currentRows.every((row, index) => selectedIds.includes(accountRowKey(row, startIndex + index)));
  const paginationItems = (() => {
    const items: (number | "...")[] = [];
    const start = Math.max(1, safePage - 1);
    const end = Math.min(pageCount, safePage + 1);

    if (start > 1) items.push(1);
    if (start > 2) items.push("...");
    for (let current = start; current <= end; current += 1) items.push(current);
    if (end < pageCount - 1) items.push("...");
    if (end < pageCount) items.push(pageCount);
    return items;
  })();

  const toggleSelectAll = (checked: boolean) => {
    if (checked) {
      onSelectChange(
        Array.from(
          new Set([...selectedIds, ...currentRows.map((item, index) => accountRowKey(item, startIndex + index))]),
        ),
      );
      return;
    }
    onSelectChange(
      selectedIds.filter((id) => !currentRows.some((row, index) => accountRowKey(row, startIndex + index) === id)),
    );
  };

  return (
    <section className="overflow-hidden rounded-[24px] border border-white/75 bg-white/82 shadow-sm backdrop-blur-sm">
      <div className="flex flex-col gap-4 border-b border-stone-100 px-4 py-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-semibold tracking-tight text-stone-950">{provider.label} 账号</h2>
            <Badge variant={provider.badgeVariant} className="rounded-md px-2.5 py-1">
              {accounts.length} 个
            </Badge>
          </div>
          <p className="max-w-2xl text-sm leading-6 text-stone-500">{provider.accountInfoHelp}</p>
        </div>
        <div className="grid gap-2 sm:grid-cols-3 lg:flex lg:items-center">
          <div className="relative min-w-0 sm:col-span-3 lg:min-w-[240px]">
            <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-stone-400" />
            <Input
              value={filters.query}
              onChange={(event) => onFilterChange({ query: event.target.value })}
              placeholder="搜索邮箱"
              className="h-10 rounded-xl border-stone-200 bg-white/90 pl-10 shadow-sm"
            />
          </div>
          <Select value={filters.type} onValueChange={(value) => onFilterChange({ type: value })}>
            <SelectTrigger className="h-10 w-full rounded-xl border-stone-200 bg-white/90 shadow-sm lg:w-[150px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {typeOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={filters.status} onValueChange={(value) => onFilterChange({ status: value as AccountStatus | "all" })}>
            <SelectTrigger className="h-10 w-full rounded-xl border-stone-200 bg-white/90 shadow-sm lg:w-[150px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {accountStatusOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-b border-stone-100 bg-stone-50/35 px-4 py-3 text-sm text-stone-500">
        <Button
          variant="ghost"
          className="h-8 rounded-lg px-3 text-stone-600 hover:bg-white hover:text-stone-900"
          onClick={() => onRefresh(selectedRefreshPayload)}
          disabled={!provider.refresh.enabled || selectedRefreshCount === 0 || isRefreshing}
          title={provider.refresh.rowTitle}
        >
          {isRefreshing ? <LoaderCircle className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
          {provider.refresh.selectedButtonLabel ?? provider.refresh.rowTitle}
        </Button>
        {canValidateProvider ? (
          <Button
            variant="ghost"
            className="h-8 rounded-lg px-3 text-stone-600 hover:bg-white hover:text-stone-900"
            onClick={() => onValidate(selectedPayloads)}
            disabled={selectedValidateCount === 0 || isValidating}
          >
            {isValidating ? <LoaderCircle className="size-4 animate-spin" /> : <CheckCircle2 className="size-4" />}
            验证所选 Grok 账号
          </Button>
        ) : null}
        <Button
          variant="ghost"
          className="h-8 rounded-lg px-3 text-amber-700 hover:bg-amber-50 hover:text-amber-800"
          onClick={onDeleteLimited}
          disabled={limitedTokens.length === 0 || isDeleting}
        >
          {isDeleting ? <LoaderCircle className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
          移除本组限流账号
        </Button>
        <Button
          variant="ghost"
          className="h-8 rounded-lg px-3 text-rose-600 hover:bg-rose-50 hover:text-rose-700"
          onClick={() => onDelete(selectedPayloads)}
          disabled={selectedDeleteCount === 0 || isDeleting}
        >
          {isDeleting ? <LoaderCircle className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
          删除本组所选
        </Button>
        <Button
          variant="ghost"
          className="h-8 rounded-lg px-3 text-stone-600 hover:bg-white hover:text-stone-900"
          onClick={() => onExport(provider.canExportWithoutTokens ? { tokens: [], identifiers: [] } : selectedPayloads)}
          disabled={!accountProviderSupportsTokenExport(provider.id, selectedTokens.length, selectedPayloads.identifiers.length) || isExporting || isDeleting}
        >
          {isExporting ? <LoaderCircle className="size-4 animate-spin" /> : <Download className="size-4" />}
          {accountSelectionCount(selectedPayloads) > 0 ? provider.selectedExportButtonLabel : provider.exportButtonLabel}
        </Button>
        {selectedIds.length > 0 ? (
          <span className="rounded-full border border-stone-200 bg-white px-3 py-1 text-xs font-semibold text-stone-600 shadow-sm">
            本组已选择 {selectedIds.length} 项
          </span>
        ) : null}
      </div>

      <div className="overflow-x-auto overscroll-x-contain">
        <table className="w-full min-w-[840px] text-left text-sm">
          <thead className="border-b border-stone-100 bg-stone-50/60 text-[11px] tracking-[0.18em] text-stone-500 uppercase">
            <tr>
              <th className="w-12 px-4 py-3">
                <Checkbox checked={allCurrentSelected} onCheckedChange={(checked) => toggleSelectAll(Boolean(checked))} />
              </th>
              <th className="w-56 px-4 py-3">token</th>
              <th className="w-28 px-4 py-3">计划/池</th>
              <th className="w-24 px-4 py-3">状态</th>
              <th className="w-56 px-4 py-3">账号信息</th>
              <th className="w-24 px-4 py-3">额度</th>
              <th className="w-40 px-4 py-3">恢复时间</th>
              <th className="w-18 px-4 py-3">成功</th>
              <th className="w-18 px-4 py-3">失败</th>
              <th className="w-24 px-4 py-3">操作</th>
            </tr>
          </thead>
          <tbody>
            {currentRows.map((account, index) => {
              const status = statusMeta[account.status];
              const StatusIcon = status.icon;
              const rowKey = accountRowKey(account, startIndex + index);
              const token = accountToken(account);
              const tokenDisplay = accountTokenDisplay(account);

              return (
                <tr key={rowKey} className="border-b border-stone-100/80 text-sm text-stone-600 transition-colors hover:bg-stone-50/80">
                  <td className="px-4 py-3">
                    <Checkbox
                      checked={selectedIds.includes(rowKey)}
                      onCheckedChange={(checked) => {
                        onSelectChange(
                          checked ? Array.from(new Set([...selectedIds, rowKey])) : selectedIds.filter((item) => item !== rowKey),
                        );
                      }}
                    />
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className="max-w-[240px] truncate rounded-lg bg-stone-100/70 px-2 py-1 font-medium tracking-tight text-stone-700"
                      title={token ? "完整 token 已隐藏，请使用导出功能获取凭据" : tokenDisplay}
                    >
                      {tokenDisplay}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="space-y-1">
                      <Badge variant="secondary" className="rounded-md bg-stone-100 text-stone-700">
                        {displayAccountType(account)}
                      </Badge>
                      <div className="text-[11px] leading-4 text-stone-400">{provider.metadataLabel}</div>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant={status.badge} className="inline-flex items-center gap-1 rounded-md px-2 py-1 shadow-sm">
                      <StatusIcon className="size-3.5" />
                      {account.status}
                    </Badge>
                  </td>
                  <td className="px-4 py-3">
                    <div className="space-y-1 text-xs leading-5 text-stone-500">
                      <div>{renderPrivacyEmail(account.email)}</div>
                      <div className="text-stone-400">{displayAccountProvider(account)} scoped account</div>
                      {accountProviderId(account) === "grok" && (
                        <div className="mt-1 flex flex-col gap-0.5 rounded-md bg-stone-50/50 p-1.5 text-[10px] text-stone-400 border border-stone-100/50">
                          {account.state_reason && <div className="font-medium text-stone-500">原因: {account.state_reason}</div>}
                          {account.last_check_status && <div>探测: {account.last_check_status} {account.last_check_http_status ? `(${account.last_check_http_status})` : ''}</div>}
                          {account.last_check_error && <div className="text-rose-400/80 truncate" title={account.last_check_error}>报错: {account.last_check_error}</div>}
                          {account.last_check_at && <div>检查: {new Date(account.last_check_at).toLocaleString()}</div>}
                          {account.last_success_at && <div>成功: {new Date(account.last_success_at).toLocaleString()}</div>}
                          {account.refresh_backoff_until && new Date(account.refresh_backoff_until).getTime() > Date.now() && <div>退避至: {new Date(account.refresh_backoff_until).toLocaleString()}</div>}
                          {account.cooldown_until && new Date(account.cooldown_until).getTime() > Date.now() && <div>冷却至: {new Date(account.cooldown_until).toLocaleString()}</div>}
                        </div>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant={provider.quota.applicable ? "info" : "secondary"} className="rounded-md px-2.5 py-1 shadow-sm">
                      {formatQuota(account)}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-xs leading-5 text-stone-500">
                    {(() => {
                      const restore = formatRestoreAt(account.restore_at);
                      return (
                        <div className="space-y-0.5">
                          {restore.relative ? <div className="font-medium text-stone-700">{restore.relative}</div> : null}
                          <div>{restore.absolute}</div>
                        </div>
                      );
                    })()}
                  </td>
                  <td className="px-4 py-3 text-stone-500">{account.success}</td>
                  <td className="px-4 py-3 text-stone-500">{account.fail}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-1 text-stone-400">
                      <button
                        type="button"
                        className="rounded-lg p-2 transition hover:bg-white hover:text-stone-700"
                        onClick={() => onEdit(account)}
                        disabled={isUpdating}
                      >
                        <Pencil className="size-4" />
                      </button>
                      <button
                        type="button"
                        className="rounded-lg p-2 transition hover:bg-white hover:text-stone-700"
                        onClick={() => onRefresh(accountSelectionPayload([account]))}
                        disabled={isRefreshing || !provider.refresh.enabled || !accountHasDeleteIdentifier(account)}
                        title={provider.refresh.rowTitle}
                      >
                        <RefreshCw className={cn("size-4", isRefreshing ? "animate-spin" : "")} />
                      </button>
                      <button
                        type="button"
                        className="rounded-lg p-2 transition hover:bg-rose-50 hover:text-rose-600"
                        onClick={() => onDelete(accountDeletePayloads([account]))}
                        disabled={isDeleting || !accountHasDeleteIdentifier(account)}
                      >
                        <Trash2 className="size-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        {!isLoading && currentRows.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-3 px-6 py-14 text-center">
            <div className="rounded-2xl bg-stone-100/80 p-3 text-stone-500 shadow-inner">
              <Search className="size-5" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-medium text-stone-700">没有匹配的 {provider.label} 账户</p>
              <p className="text-sm text-stone-500">调整本组筛选条件或导入新的账号。</p>
            </div>
          </div>
        ) : null}
      </div>

      <div className="border-t border-stone-100 bg-stone-50/30 px-4 py-4">
        <div className="flex items-center justify-start gap-3 overflow-x-auto whitespace-nowrap md:justify-center">
          <div className="shrink-0 text-sm text-stone-500">
            显示第 {filteredAccounts.length === 0 ? 0 : startIndex + 1} - {Math.min(startIndex + Number(pageSize), filteredAccounts.length)} 条，共 {filteredAccounts.length} 条
          </div>
          <span className="shrink-0 text-sm leading-none text-stone-500">{page} / {pageCount} 页</span>
          <Button
            variant="outline"
            size="icon"
            className="size-10 shrink-0 rounded-lg border-stone-200 bg-white"
            disabled={safePage <= 1}
            onClick={() => onPageChange(Math.max(1, safePage - 1))}
          >
            <ChevronLeft className="size-4" />
          </Button>
          {paginationItems.map((item, index) =>
            item === "..." ? (
              <span key={`ellipsis-${index}`} className="px-1 text-sm text-stone-400">...</span>
            ) : (
              <Button
                key={item}
                variant={item === safePage ? "default" : "outline"}
                className={cn(
                  "h-10 min-w-10 shrink-0 rounded-lg px-3",
                  item === safePage ? "bg-stone-950 text-white hover:bg-stone-800" : "border-stone-200 bg-white text-stone-700",
                )}
                onClick={() => onPageChange(item)}
              >
                {item}
              </Button>
            ),
          )}
          <Button
            variant="outline"
            size="icon"
            className="size-10 shrink-0 rounded-lg border-stone-200 bg-white"
            disabled={safePage >= pageCount}
            onClick={() => onPageChange(Math.min(pageCount, safePage + 1))}
          >
            <ChevronRight className="size-4" />
          </Button>
        </div>
      </div>
    </section>
  );
}

function AccountsPageContent() {
  const didLoadRef = useRef(false);
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [activeProvider, setActiveProvider] = useState<ProviderId>("gpt");
  const [selectedIds, setSelectedIds] = useState<ProviderSelection>(defaultProviderSelection);
  const [filters, setFilters] = useState<ProviderFilters>(defaultProviderFilters);
  const [pages, setPages] = useState<ProviderPages>(defaultProviderPages);
  const [pageSize] = useState("10");
  const [editingAccount, setEditingAccount] = useState<Account | null>(null);
  const [editStatus, setEditStatus] = useState<AccountStatus>("正常");
  const [editProxy, setEditProxy] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const loadAccounts = async (silent = false) => {
    if (!silent) {
      setIsLoading(true);
    }
    try {
      const results = await Promise.all(accountProviderDefinitions.map((provider) => fetchAccounts(provider.id)));
      const items = results.flatMap((result, index) =>
        result.items.map((item) => ({ ...item, provider: normalizeAccountProvider(item.provider) || accountProviderDefinitions[index].id })),
      );
      setAccounts(items);
      setSelectedIds((prev) => keepProviderSelection(prev, items));
    } catch (error) {
      const message = error instanceof Error ? error.message : "加载账户失败";
      toast.error(message);
    } finally {
      if (!silent) {
        setIsLoading(false);
      }
    }
  };

  useEffect(() => {
    if (didLoadRef.current) {
      return;
    }
    didLoadRef.current = true;
    void loadAccounts();
  }, []);

  const providerRows = useMemo(() => {
    return accountProviderDefinitions.reduce((acc, provider) => {
      const providerAccounts = accounts.filter((account) => isProviderAccount(account, provider.id));
      const providerFilters = filters[provider.id];
      const normalizedQuery = providerFilters.query.trim().toLowerCase();
      const filteredAccounts = providerAccounts.filter((account) => {
        const searchMatched = normalizedQuery.length === 0 || (account.email ?? "").toLowerCase().includes(normalizedQuery);
        const typeMatched = providerFilters.type === "all" || displayAccountType(account) === providerFilters.type;
        const statusMatched = providerFilters.status === "all" || account.status === providerFilters.status;
        return searchMatched && typeMatched && statusMatched;
      });
      const pageCount = Math.max(1, Math.ceil(filteredAccounts.length / Number(pageSize)));
      const safePage = Math.min(pages[provider.id], pageCount);
      const startIndex = (safePage - 1) * Number(pageSize);
      acc[provider.id] = {
        accounts: providerAccounts,
        filteredAccounts,
        currentRows: filteredAccounts.slice(startIndex, startIndex + Number(pageSize)),
        pageCount,
        safePage,
        startIndex,
      };
      return acc;
    }, {} as Record<ProviderId, { accounts: Account[]; filteredAccounts: Account[]; currentRows: Account[]; pageCount: number; safePage: number; startIndex: number }>);
  }, [accounts, filters, pageSize, pages]);

  const summary = useMemo(() => {
    const total = accounts.length;
    const active = accounts.filter((item) => item.status === "正常").length;
    const limited = accounts.filter((item) => item.status === "限流").length;
    const abnormal = accounts.filter((item) => item.status === "异常").length;
    const disabled = accounts.filter((item) => item.status === "禁用").length;
    const quota = formatQuotaSummary(accounts);

    return { total, active, limited, abnormal, disabled, quota };
  }, [accounts]);

  const activeDefinition = getAccountProviderDefinition(activeProvider);
  const activeRows = providerRows[activeProvider];

  const setProviderFilters = (provider: ProviderId, next: Partial<ProviderFilters[ProviderId]>) => {
    setFilters((prev) => ({ ...prev, [provider]: { ...prev[provider], ...next } }));
    setPages((prev) => ({ ...prev, [provider]: 1 }));
  };

  const setProviderPage = (provider: ProviderId, page: number) => {
    setPages((prev) => ({ ...prev, [provider]: page }));
  };

  const setProviderSelection = (provider: ProviderId, ids: string[]) => {
    setSelectedIds((prev) => ({ ...prev, [provider]: ids }));
  };

  const handleProviderMutationResult = (provider: ProviderId, items: Account[]) => {
    const scopedItems = items.filter((item) => isProviderAccount(item, provider));
    const nextAccounts = scopedItems.length === items.length ? mergeProviderAccounts(accounts, provider, scopedItems) : items;
    setAccounts(nextAccounts);
    setSelectedIds((prev) => keepProviderSelection(prev, nextAccounts));
  };

  const handleDeleteTokens = async (provider: ProviderId, payload: AccountDeletePayload) => {
    if (payload.tokens.length === 0 && payload.identifiers.length === 0) {
      toast.error(`请先选择要删除的 ${getAccountProviderLabel(provider)} 账户`);
      return;
    }

    setIsDeleting(true);
    try {
      const data = await deleteAccounts(payload, provider);
      handleProviderMutationResult(provider, data.items);
      toast.success(`删除 ${data.removed ?? 0} 个 ${getAccountProviderLabel(provider)} 账户`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "删除账户失败";
      toast.error(message);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDeleteLimitedAccounts = async (provider: ProviderId) => {
    const limitedTokens = accountTokens(providerRows[provider].accounts.filter((item) => item.status === "限流"));
    if (limitedTokens.length === 0) {
      toast.error(`没有限流 ${getAccountProviderLabel(provider)} 账户可移除`);
      return;
    }

    setIsDeleting(true);
    try {
      const data = await deleteLimitedAccounts(provider);
      handleProviderMutationResult(provider, data.items);
      toast.success(`移除 ${data.removed ?? 0} 个限流 ${getAccountProviderLabel(provider)} 账户`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "移除限流账户失败";
      toast.error(message);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleRefreshAccounts = async (provider: ProviderId, payload: AccountSelectionPayload) => {
    const providerDefinition = getAccountProviderDefinition(provider);
    if (!providerDefinition.refresh.enabled) {
      toast.error(providerDefinition.refresh.rowTitle);
      return;
    }
    if (accountSelectionCount(payload) === 0) {
      toast.error(`没有可刷新的 ${providerDefinition.label} 账户`);
      return;
    }

    setIsRefreshing(true);
    try {
      const data = await refreshAccounts(payload, provider);
      handleProviderMutationResult(provider, data.items);
      if (provider === "grok") {
         const summary = `检查 ${data.checked ?? 0} 个，更新 ${data.refreshed ?? 0} 个，状态保持 ${data.unchanged ?? 0} 个，失败 ${data.failed ?? 0} 个`;
         const explanation = (data.unchanged ?? 0) > 0 ? " (Cloudflare/上游检查不可用不等于账号失效)" : "";
         if (data.errors.length > 0) {
           const firstError = data.errors[0]?.error;
           toast.error(`${summary}${explanation}${firstError ? `，首个错误：${firstError}` : ""}`);
         } else {
           toast.success(`${summary}${explanation}`);
         }
      } else {
        if (data.errors.length > 0) {
          const firstError = data.errors[0]?.error;
          toast.error(
            `刷新成功 ${data.refreshed} 个 ${providerDefinition.label} 账户，失败 ${data.errors.length} 个${firstError ? `，首个错误：${firstError}` : ""}`,
          );
        } else {
          toast.success(`刷新成功 ${data.refreshed} 个 ${providerDefinition.label} 账户`);
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : `刷新 ${providerDefinition.label} 账户失败`;
      toast.error(message);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleValidateAccounts = async (provider: ProviderId, payload: AccountSelectionPayload) => {
    if (provider !== "grok") {
      toast.error("当前仅支持验证 Grok 账号");
      return;
    }
    if (accountSelectionCount(payload) === 0) {
      toast.error("请先选择要验证的 Grok 账号");
      return;
    }

    setIsValidating(true);
    try {
      const data = await validateAccounts(payload, provider);
      handleProviderMutationResult(provider, data.items);
      const summary = `检查 ${data.checked} 个，正常 ${data.valid} 个，异常 ${data.invalid} 个，限流 ${data.limited} 个，未验证 ${data.unverified} 个`;
      const explanation = data.unverified > 0 ? " (未验证指 Cloudflare/Browser Bridge/网络探测等不可用，非凭据无效)" : "";
      const firstError = validationErrorMessage(data.errors[0]);
      if (data.errors.length > 0) {
        toast.error(`${summary}${explanation}${firstError ? `，首个错误：${firstError}` : ""}`);
      } else {
        toast.success(`${summary}${explanation}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "验证 Grok 账号失败";
      toast.error(message);
    } finally {
      setIsValidating(false);
    }
  };

  const openEditDialog = (account: Account) => {
    setEditingAccount(account);
    setEditStatus(account.status);
    setEditProxy(String(account.proxy ?? ""));
  };

  const handleUpdateAccount = async () => {
    if (!editingAccount) {
      return;
    }

    const provider = accountProviderId(editingAccount);
    const token = accountToken(editingAccount);
    const proxy = editProxy.trim();
    const statusChanged = editStatus !== editingAccount.status;
    const proxyChanged = proxy !== String(editingAccount.proxy ?? "").trim();
    if (!token && statusChanged) {
      toast.error("脱敏账号不能在列表中直接编辑，请重新导入或通过后端管理接口处理");
      return;
    }
    if (!token && !proxyChanged) {
      toast.error("还没有检测到改动，请修改后再保存");
      return;
    }

    setIsUpdating(true);
    try {
      const data = await updateAccount(token, {
        ...(token ? { status: editStatus } : {}),
        proxy,
        account_id: editingAccount.account_id,
        row_id: editingAccount.row_id,
      }, provider);
      handleProviderMutationResult(provider, data.items);
      setEditingAccount(null);
      toast.success("账号设置已更新");
    } catch (error) {
      const message = error instanceof Error ? error.message : "更新账号失败";
      toast.error(message);
    } finally {
      setIsUpdating(false);
    }
  };

  const handleExportAccounts = async (provider: AccountExportProvider, payload: AccountSelectionPayload) => {
    if (!accountProviderSupportsTokenExport(provider, payload.tokens.length, payload.identifiers.length)) {
      const label = accountExportProviderLabel(provider);
      toast.error(`没有可导出的 ${label} 账户`);
      return;
    }

    setIsExporting(true);
    try {
      const data = await exportAccounts(provider, payload);
      downloadBlob(data.blob, data.filename);
      const label = accountExportProviderLabel(provider);
      toast.success(`${label} TXT 文件已导出`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "导出账户失败";
      toast.error(message);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <>
      <section className="relative overflow-hidden rounded-[28px] border border-white/70 bg-white/55 p-5 shadow-[var(--shadow-soft)] backdrop-blur-sm before:pointer-events-none before:absolute before:inset-x-6 before:top-0 before:h-px before:bg-white/90 lg:p-6">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-2">
            <div className="text-xs font-semibold tracking-[0.18em] text-stone-500 uppercase">Account Pool</div>
            <h1 className="text-2xl font-semibold tracking-tight text-stone-950">号池管理</h1>
            <p className="max-w-2xl text-sm leading-6 text-stone-500">GPT、Grok、Gemini 分组管理，批量操作只作用于当前服务商。</p>
          </div>

          <div className="flex flex-wrap items-center gap-2 rounded-2xl border border-white/70 bg-white/45 p-2 shadow-sm">
            <Button
              variant="outline"
              className="h-10 rounded-xl border-stone-200 bg-white/85 px-4 text-stone-700 shadow-sm hover:bg-white"
              onClick={() => void loadAccounts()}
              disabled={isLoading || isRefreshing || isValidating || isDeleting}
            >
              <RefreshCw className={cn("size-4", isLoading ? "animate-spin" : "")} />
              刷新全部
            </Button>
            <Button
              variant="outline"
              className="h-10 rounded-xl border-stone-200 bg-white/85 px-4 text-stone-700 shadow-sm hover:bg-white"
              onClick={() => void handleRefreshAccounts(activeProvider, accountSelectionPayload(activeRows.accounts))}
              disabled={!activeDefinition.refresh.enabled || isLoading || isRefreshing || isValidating || isDeleting || accountSelectionCount(accountSelectionPayload(activeRows.accounts)) === 0}
            >
              <RefreshCw className={cn("size-4", isRefreshing ? "animate-spin" : "")} />
              {activeDefinition.refresh.buttonLabel ?? activeDefinition.refresh.rowTitle}
            </Button>
            <AccountImportDialog
              disabled={isLoading || isRefreshing || isValidating || isDeleting}
              onImported={async (items, provider) => {
                handleProviderMutationResult(provider, items);
                await loadAccounts(true);
                setProviderSelection(provider, []);
                setProviderPage(provider, 1);
                setActiveProvider(provider);
              }}
            />
          </div>
        </div>
      </section>

      <Dialog open={Boolean(editingAccount)} onOpenChange={(open) => (!open ? setEditingAccount(null) : null)}>
        <DialogContent showCloseButton={false} className="rounded-2xl p-6">
          <DialogHeader className="gap-2">
            <DialogTitle>编辑账户设置</DialogTitle>
            <DialogDescription className="text-sm leading-6">
              当前账号归属 {editingAccount ? getAccountProviderLabel(editingAccount.provider) : ""}；更新请求会按该服务商定位账号。
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <label className="text-sm font-medium text-stone-700">状态</label>
            <Select value={editStatus} onValueChange={(value) => setEditStatus(value as AccountStatus)}>
              <SelectTrigger className="h-11 rounded-xl border-stone-200 bg-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {accountStatusOptions
                  .filter((option) => option.value !== "all")
                  .map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-stone-700">账号代理</label>
            <Input
              value={editProxy}
              onChange={(event) => setEditProxy(event.target.value)}
              placeholder="留空使用全局代理，例如 http://127.0.0.1:7890"
              className="h-11 rounded-xl border-stone-200 bg-white"
            />
            <p className="text-xs leading-5 text-stone-500">设置后该账号请求优先使用此代理；留空则使用系统全局代理。</p>
          </div>
          <DialogFooter className="pt-2">
            <Button
              variant="secondary"
              className="h-10 rounded-xl bg-stone-100 px-5 text-stone-700 hover:bg-stone-200"
              onClick={() => setEditingAccount(null)}
              disabled={isUpdating}
            >
              取消
            </Button>
            <Button className="h-10 rounded-xl bg-stone-950 px-5 text-white hover:bg-stone-800" onClick={() => void handleUpdateAccount()} disabled={isUpdating}>
              {isUpdating ? <LoaderCircle className="size-4 animate-spin" /> : null}
              保存修改
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <section className="space-y-4">
        <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
          {metricCards.map((item) => {
            const Icon = item.icon;
            const value = summary[item.key];
            return (
              <Card key={item.key} className="group overflow-hidden rounded-2xl border-white/80 bg-white/90 shadow-sm transition hover:-translate-y-0.5 hover:shadow-[var(--shadow-soft)]">
                <CardContent className="p-4">
                  <div className="mb-4 flex items-start justify-between">
                    <span className="text-xs font-semibold tracking-[0.12em] text-stone-400 uppercase">{item.label}</span>
                    <span className="rounded-xl bg-stone-100/80 p-2 text-stone-400 transition group-hover:bg-stone-200/70 group-hover:text-stone-600">
                      <Icon className="size-4" />
                    </span>
                  </div>
                  <div className={cn("text-[1.75rem] font-semibold tracking-tight", item.color)}>
                    <span className={typeof value === "number" ? "" : "text-[1.1rem]"}>{typeof value === "number" ? formatCompact(value) : value}</span>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
        <p className="rounded-2xl border border-white/60 bg-white/45 px-4 py-2 text-xs leading-5 text-stone-500 shadow-sm">
          Token 仅以遮蔽形式展示；导出操作按当前服务商执行。GPT 图像额度仅统计正常 GPT 账号。
        </p>
      </section>

      <section className="space-y-4">
        <div className="flex gap-2 overflow-x-auto rounded-[22px] border border-white/70 bg-white/45 p-2 shadow-sm">
          {accountProviderDefinitions.map((provider) => (
            <button
              key={provider.id}
              type="button"
              onClick={() => setActiveProvider(provider.id)}
              className={cn(
                "flex min-w-[150px] items-center justify-between rounded-2xl px-4 py-3 text-left transition",
                activeProvider === provider.id ? "bg-stone-950 text-white shadow-sm" : "bg-white/70 text-stone-600 hover:bg-white",
              )}
            >
              <span className="font-semibold">{provider.label}</span>
              <span className={cn("text-xs", activeProvider === provider.id ? "text-stone-300" : "text-stone-400")}>
                {providerRows[provider.id].accounts.length}
              </span>
            </button>
          ))}
        </div>

        {isLoading && accounts.length === 0 ? (
          <Card className="rounded-[24px] border-white/80 bg-white/90 shadow-[var(--shadow-soft)]">
            <CardContent className="flex flex-col items-center justify-center gap-3 px-6 py-14 text-center">
              <div className="rounded-xl bg-stone-100 p-3 text-stone-500">
                <LoaderCircle className="size-5 animate-spin" />
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium text-stone-700">正在加载账户</p>
                <p className="text-sm text-stone-500">按服务商同步账号列表和状态。</p>
              </div>
            </CardContent>
          </Card>
        ) : (
          <ProviderAccountSection
            provider={activeDefinition}
            accounts={activeRows.accounts}
            filteredAccounts={activeRows.filteredAccounts}
            currentRows={activeRows.currentRows}
            selectedIds={selectedIds[activeProvider]}
            filters={filters[activeProvider]}
            page={activeRows.safePage}
            pageSize={pageSize}
            pageCount={activeRows.pageCount}
            safePage={activeRows.safePage}
            startIndex={activeRows.startIndex}
            isLoading={isLoading}
            isRefreshing={isRefreshing}
            isValidating={isValidating}
            isDeleting={isDeleting}
            isExporting={isExporting}
            isUpdating={isUpdating}
            onFilterChange={(next) => setProviderFilters(activeProvider, next)}
            onPageChange={(nextPage) => setProviderPage(activeProvider, nextPage)}
            onSelectChange={(ids) => setProviderSelection(activeProvider, ids)}
            onRefresh={(payload) => void handleRefreshAccounts(activeProvider, payload)}
            onValidate={(payload) => void handleValidateAccounts(activeProvider, payload)}
            onDelete={(payload) => void handleDeleteTokens(activeProvider, payload)}
            onDeleteLimited={() => void handleDeleteLimitedAccounts(activeProvider)}
            onExport={(payload) => void handleExportAccounts(activeProvider, payload)}
            onEdit={openEditDialog}
          />
        )}
      </section>
    </>
  );
}

export default function AccountsPage() {
  const { isCheckingAuth, session } = useAuthGuard(["admin"]);

  if (isCheckingAuth || !session || session.role !== "admin") {
    return (
      <div className="flex min-h-[40vh] items-center justify-center">
        <LoaderCircle className="size-5 animate-spin text-stone-400" />
      </div>
    );
  }

  return <AccountsPageContent />;
}
