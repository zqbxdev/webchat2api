from __future__ import annotations

from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path
import time

from services.storage.base import StorageBackend

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = BASE_DIR / "config.json"
VERSION_FILE = BASE_DIR / "VERSION"
BACKUP_STATE_FILE = DATA_DIR / "backup_state.json"
DEFAULT_LOGIN_SECRET = "admin"

DEFAULT_BACKUP_INCLUDE = {
    "config": False,
    "cpa": False,
    "sub2api": False,
    "logs": True,
    "image_tasks": True,
    "accounts_snapshot": False,
    "auth_keys_snapshot": False,
    "images": False,
}

DEFAULT_IMAGE_STORAGE = {
    "enabled": False,
    "mode": "local",
    "webdav_url": "",
    "webdav_username": "",
    "webdav_password": "",
    "webdav_root_path": "webchat2api/images",
    "public_base_url": "",
}


PLACEHOLDER_AUTH_KEYS = {"change-me", "your_secret_key", "your_secret_key_here"}
PERSISTENT_CONFIG_KEYS = {
    "refresh_account_interval_minute",
    "image_retention_days",
    "image_poll_timeout_secs",
    "image_poll_interval_secs",
    "image_poll_initial_wait_secs",
    "image_account_concurrency",
    "auto_remove_invalid_accounts",
    "auto_remove_rate_limited_accounts",
    "show_search_sources",
    "log_levels",
    "sensitive_words",
    "ai_review",
    "global_system_prompt",
    "backup",
    "image_storage",
    "proxy",
    "base_url",
    "chatgpt_fingerprint",
    "grok_console_fingerprint",
    "network_profiles",
    "flaresolverr_url",
    "flaresolverr_timeout_sec",
    "enable_turnstile_solver",
    "browser_bridge_url",
    "chat_completion_cache",
    "chat_completion_message_normalization",
    "mihomo",
}


def _normalize_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _normalize_positive_int(value: object, default: int, minimum: int = 0) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = default
    return max(minimum, normalized)


def _normalize_backup_include(value: object) -> dict[str, bool]:
    source = value if isinstance(value, dict) else {}
    normalized = dict(DEFAULT_BACKUP_INCLUDE)
    for key in normalized:
        normalized[key] = _normalize_bool(source.get(key), normalized[key])
    return normalized


DEFAULT_MIHOMO_REGION = "🇺🇸,美国,美國,United States,America"


def _normalize_mihomo_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    bin_default = "scripts/.bin/mihomo"
    ctrl_default = "127.0.0.1:9090"
    test_url_default = "https://chatgpt.com/api/auth/csrf"
    return {
        "bin_path": str(source.get("bin_path") or bin_default).strip() or bin_default,
        "port_base": _normalize_positive_int(source.get("port_base"), 30000, 1024),
        "port_range_size": _normalize_positive_int(source.get("port_range_size"), 500, 1),
        "external_controller": str(source.get("external_controller") or ctrl_default).strip() or ctrl_default,
        "external_controller_secret": str(source.get("external_controller_secret") or "").strip(),
        "region_keywords": str(source.get("region_keywords") or DEFAULT_MIHOMO_REGION).strip() or DEFAULT_MIHOMO_REGION,
        "sync_interval_hours": _normalize_positive_int(source.get("sync_interval_hours"), 24, 1),
        "test_url": str(source.get("test_url") or test_url_default).strip() or test_url_default,
        "test_timeout": _normalize_positive_int(source.get("test_timeout"), 15, 1),
        "concurrency": _normalize_positive_int(source.get("concurrency"), 16, 1),
        "probe_retry": _normalize_positive_int(source.get("probe_retry"), 1, 0),
    }


def _normalize_backup_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    return {
        "enabled": _normalize_bool(source.get("enabled"), False),
        "provider": "cloudflare_r2",
        "account_id": str(source.get("account_id") or "").strip(),
        "access_key_id": str(source.get("access_key_id") or "").strip(),
        "secret_access_key": str(source.get("secret_access_key") or "").strip(),
        "bucket": str(source.get("bucket") or "").strip(),
        "prefix": str(source.get("prefix") or "backups").strip().strip("/") or "backups",
        "interval_minutes": _normalize_positive_int(source.get("interval_minutes"), 360, 1),
        "rotation_keep": _normalize_positive_int(source.get("rotation_keep"), 10, 0),
        "encrypt": _normalize_bool(source.get("encrypt"), True),
        "passphrase": str(source.get("passphrase") or "").strip(),
        "include": _normalize_backup_include(source.get("include")),
    }


def _normalize_backup_state(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    return {
        "last_started_at": str(source.get("last_started_at") or "").strip() or None,
        "last_finished_at": str(source.get("last_finished_at") or "").strip() or None,
        "last_status": str(source.get("last_status") or "idle").strip() or "idle",
        "last_error": str(source.get("last_error") or "").strip() or None,
        "last_object_key": str(source.get("last_object_key") or "").strip() or None,
    }


CHATGPT_FINGERPRINT_KEYS = {
    "user-agent",
    "impersonate",
    "oai-device-id",
    "oai-session-id",
    "sec-ch-ua",
    "sec-ch-ua-mobile",
    "sec-ch-ua-platform",
}


def _nonempty_string(value: object) -> str:
    return str(value or "").strip()


def _normalize_chatgpt_fingerprint(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    return {
        key: text
        for key in CHATGPT_FINGERPRINT_KEYS
        if (text := _nonempty_string(source.get(key)))
    }


def _normalize_grok_network_profile(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    normalized: dict[str, object] = {}
    if impersonate := _nonempty_string(source.get("impersonate")):
        normalized["impersonate"] = impersonate
    if browser := _nonempty_string(source.get("browser")):
        normalized["browser"] = browser
    user_agent = _nonempty_string(source.get("user-agent") or source.get("user_agent"))
    if user_agent:
        normalized["user-agent"] = user_agent
    for key in ("cf_clearance", "cf_cookies", "sec-ch-ua", "sec_ch_ua", "sec-ch-ua-mobile", "sec_ch_ua_mobile", "sec-ch-ua-platform", "sec_ch_ua_platform", "statsig_id", "x-statsig-id"):
        if text := _nonempty_string(source.get(key)):
            normalized[key] = text
    if "verify" in source:
        normalized["verify"] = _normalize_bool(source.get("verify"), True)
    if "timeout" in source:
        try:
            timeout = float(source.get("timeout"))
        except (TypeError, ValueError):
            timeout = 0.0
        if timeout > 0:
            normalized["timeout"] = int(timeout) if timeout.is_integer() else timeout
    return normalized


def _normalize_grok_console_profile(value: object) -> dict[str, object]:
    normalized = _normalize_grok_network_profile(value)
    normalized.pop("browser", None)
    for key in ("cf_cookies", "sec-ch-ua", "sec_ch_ua", "sec-ch-ua-mobile", "sec_ch_ua_mobile", "sec-ch-ua-platform", "sec_ch_ua_platform", "statsig_id", "x-statsig-id"):
        normalized.pop(key, None)
    return normalized


def _normalize_grok_console_fingerprint(value: object) -> dict[str, object]:
    return _normalize_grok_console_profile(value)


def _normalize_network_profiles(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    normalized: dict[str, object] = {}
    grok_console = _normalize_grok_console_profile(source.get("grok_console"))
    if grok_console:
        normalized["grok_console"] = grok_console
    grok_app_chat = _normalize_grok_network_profile(source.get("grok_app_chat"))
    if grok_app_chat:
        normalized["grok_app_chat"] = grok_app_chat
    return normalized


def _normalize_chat_completion_cache_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    return {
        "enabled": _normalize_bool(source.get("enabled"), False),
        "ttl_seconds": _normalize_positive_int(source.get("ttl_seconds"), 60, 0),
        "max_entries": _normalize_positive_int(source.get("max_entries"), 256, 1),
        "cache_stream": _normalize_bool(source.get("cache_stream"), False),
        "cache_tool_calls": _normalize_bool(source.get("cache_tool_calls"), False),
        "dedupe_inflight": _normalize_bool(source.get("dedupe_inflight"), True),
    }


def _normalize_chat_completion_message_normalization_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    return {
        "enabled": _normalize_bool(source.get("enabled"), False),
        "drop_adjacent_duplicates": _normalize_bool(source.get("drop_adjacent_duplicates"), True),
        "drop_assistant_history": _normalize_bool(source.get("drop_assistant_history"), False),
        "merge_adjacent_same_role_text": _normalize_bool(source.get("merge_adjacent_same_role_text"), False),
    }


def _normalize_image_storage_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    mode = str(source.get("mode") or "local").strip().lower()
    if mode not in {"local", "webdav", "both"}:
        mode = "local"
    enabled = _normalize_bool(source.get("enabled"), False)
    if not enabled:
        mode = "local"
    root_path = str(source.get("webdav_root_path") or DEFAULT_IMAGE_STORAGE["webdav_root_path"]).strip().strip("/")
    return {
        "enabled": enabled,
        "mode": mode,
        "webdav_url": str(source.get("webdav_url") or "").strip().rstrip("/"),
        "webdav_username": str(source.get("webdav_username") or "").strip(),
        "webdav_password": str(source.get("webdav_password") or "").strip(),
        "webdav_root_path": root_path or str(DEFAULT_IMAGE_STORAGE["webdav_root_path"]),
        "public_base_url": str(source.get("public_base_url") or "").strip().rstrip("/"),
    }


def _validate_image_storage_settings(settings: dict[str, object]) -> None:
    if not _normalize_bool(settings.get("enabled"), False):
        return
    if not str(settings.get("webdav_url") or "").strip():
        raise ValueError("启用 WebDAV 图片存储后必须填写 WebDAV URL")
    if not str(settings.get("webdav_password") or "").strip():
        raise ValueError("启用 WebDAV 图片存储后必须填写 WebDAV 密码")


@dataclass(frozen=True)
class LoadedSettings:
    auth_key: str
    refresh_account_interval_minute: int


def _normalize_auth_key(value: object) -> str:
    return str(value or "").strip()


def _is_invalid_auth_key(value: object) -> bool:
    return _normalize_auth_key(value) in PLACEHOLDER_AUTH_KEYS


def _read_json_object(path: Path, *, name: str) -> dict[str, object]:
    if not path.exists():
        return {}
    if path.is_dir():
        print(
            f"Warning: {name} at '{path}' is a directory, ignoring it and falling back to other configuration sources.",
            file=sys.stderr,
        )
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _persistent_settings(data: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in data.items() if key in PERSISTENT_CONFIG_KEYS}


def _configured_auth_key(raw_config: dict[str, object]) -> str:
    return _normalize_auth_key(
        os.getenv("LOGIN_SECRET")
        or os.getenv("WEBCHAT2API_AUTH_KEY")
        or raw_config.get("auth-key")
        or DEFAULT_LOGIN_SECRET
    )


def _load_settings() -> LoadedSettings:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_config = _read_json_object(CONFIG_FILE, name="config.json")
    auth_key = _configured_auth_key(raw_config)
    if _is_invalid_auth_key(auth_key):
        raise ValueError(
            "❌ auth-key 未设置或仍为占位值！\n"
            "请在环境变量 LOGIN_SECRET 或 WEBCHAT2API_AUTH_KEY 中设置强随机密钥，或者在 config.json 中填写非默认 auth-key。"
        )

    try:
        refresh_interval = int(raw_config.get("refresh_account_interval_minute", 5))
    except (TypeError, ValueError):
        refresh_interval = 5

    return LoadedSettings(
        auth_key=auth_key,
        refresh_account_interval_minute=refresh_interval,
    )


class ConfigStore:
    def __init__(self, path: Path):
        self.path = path
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._storage_backend: StorageBackend | None = None
        self.data = self._load()
        if _is_invalid_auth_key(self.auth_key):
            raise ValueError(
                "❌ auth-key 未设置或仍为占位值！\n"
                "请按以下任意一种方式解决：\n"
                "1. 设置强随机登录密钥：\n"
                "   LOGIN_SECRET = your_real_random_auth_key\n"
                "   或 WEBCHAT2API_AUTH_KEY = your_real_random_auth_key\n"
                "2. 或者在 config.json 中填写非默认 auth-key：\n"
                '   "auth-key": "your_real_random_auth_key"'
            )

    def _load(self) -> dict[str, object]:
        file_settings = _read_json_object(self.path, name="config.json")
        storage = self.get_storage_backend()
        storage_settings = storage.load_settings()
        if storage_settings:
            return {**file_settings, **_persistent_settings(storage_settings)}
        initial_settings = _persistent_settings(file_settings)
        if initial_settings:
            storage.save_settings(initial_settings)
        return file_settings

    def _save(self) -> None:
        self.get_storage_backend().save_settings(_persistent_settings(self.data))

    @property
    def auth_key(self) -> str:
        return _configured_auth_key(self.data)

    @property
    def accounts_file(self) -> Path:
        return DATA_DIR / "accounts.json"

    @property
    def refresh_account_interval_minute(self) -> int:
        try:
            return int(self.data.get("refresh_account_interval_minute", 5))
        except (TypeError, ValueError):
            return 5

    @property
    def image_retention_days(self) -> int:
        try:
            return max(1, int(self.data.get("image_retention_days", 30)))
        except (TypeError, ValueError):
            return 30

    @property
    def image_poll_timeout_secs(self) -> int:
        try:
            return max(1, int(self.data.get("image_poll_timeout_secs", 120)))
        except (TypeError, ValueError):
            return 120

    @property
    def image_poll_interval_secs(self) -> float:
        try:
            return max(0.5, float(self.data.get("image_poll_interval_secs", 10.0)))
        except (TypeError, ValueError):
            return 10.0

    @property
    def image_poll_initial_wait_secs(self) -> float:
        """Image generation upstream takes ~30s; polling immediately wastes requests
        and trips a transient 429. Default 10s gives the conversation document time
        to commit before the first poll."""
        try:
            return max(0.0, float(self.data.get("image_poll_initial_wait_secs", 10.0)))
        except (TypeError, ValueError):
            return 10.0

    @property
    def image_account_concurrency(self) -> int:
        try:
            return max(1, int(self.data.get("image_account_concurrency", 3)))
        except (TypeError, ValueError):
            return 3

    @property
    def auto_remove_invalid_accounts(self) -> bool:
        value = self.data.get("auto_remove_invalid_accounts", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def auto_remove_rate_limited_accounts(self) -> bool:
        value = self.data.get("auto_remove_rate_limited_accounts", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def show_search_sources(self) -> bool:
        value = self.data.get("show_search_sources", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def log_levels(self) -> list[str]:
        levels = self.data.get("log_levels")
        if not isinstance(levels, list):
            return []
        allowed = {"debug", "info", "warning", "error"}
        return [level for item in levels if (level := str(item or "").strip().lower()) in allowed]

    @property
    def sensitive_words(self) -> list[str]:
        words = self.data.get("sensitive_words")
        return [word for item in words if (word := str(item or "").strip())] if isinstance(words, list) else []

    @property
    def ai_review(self) -> dict[str, object]:
        value = self.data.get("ai_review")
        return value if isinstance(value, dict) else {}

    @property
    def global_system_prompt(self) -> str:
        return str(self.data.get("global_system_prompt") or "").strip()

    @property
    def images_dir(self) -> Path:
        path = DATA_DIR / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def image_thumbnails_dir(self) -> Path:
        path = DATA_DIR / "image_thumbnails"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cleanup_old_images(self) -> int:
        cutoff = time.time() - self.image_retention_days * 86400
        removed = 0
        for path in self.images_dir.rglob("*"):
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        for path in sorted((p for p in self.images_dir.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass
        return removed

    @property
    def base_url(self) -> str:
        return str(
            os.getenv("WEBCHAT2API_BASE_URL")
            or self.data.get("base_url")
            or ""
        ).strip().rstrip("/")

    @property
    def flaresolverr_url(self) -> str:
        return str(self.data.get("flaresolverr_url") or "").strip().rstrip("/")

    @property
    def flaresolverr_timeout_sec(self) -> int:
        try:
            return max(1, int(self.data.get("flaresolverr_timeout_sec", 60)))
        except (TypeError, ValueError):
            return 60

    @property
    def browser_bridge_url(self) -> str:
        return str(self.data.get("browser_bridge_url") or "").strip().rstrip("/")

    @property
    def app_version(self) -> str:
        try:
            value = VERSION_FILE.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return "0.0.0"
        return value or "0.0.0"

    def get(self) -> dict[str, object]:
        data = dict(self.data)
        data["base_url"] = self.base_url
        data["proxy"] = self.get_proxy_settings()
        data["refresh_account_interval_minute"] = self.refresh_account_interval_minute
        data["image_retention_days"] = self.image_retention_days
        data["image_poll_timeout_secs"] = self.image_poll_timeout_secs
        data["image_poll_interval_secs"] = self.image_poll_interval_secs
        data["image_poll_initial_wait_secs"] = self.image_poll_initial_wait_secs
        data["image_account_concurrency"] = self.image_account_concurrency
        data["auto_remove_invalid_accounts"] = self.auto_remove_invalid_accounts
        data["auto_remove_rate_limited_accounts"] = self.auto_remove_rate_limited_accounts
        data["show_search_sources"] = self.show_search_sources
        data["log_levels"] = self.log_levels
        data["sensitive_words"] = self.sensitive_words
        data["ai_review"] = self.ai_review
        data["global_system_prompt"] = self.global_system_prompt
        data["backup"] = self.get_backup_settings()
        data["image_storage"] = self.get_image_storage_settings()
        data["chatgpt_fingerprint"] = self.chatgpt_fingerprint
        data["grok_console_fingerprint"] = self.grok_console_fingerprint
        data["chat_completion_cache"] = self.get_chat_completion_cache_settings()
        data["chat_completion_message_normalization"] = self.get_chat_completion_message_normalization_settings()
        data["network_profiles"] = self.network_profiles
        data["flaresolverr_url"] = self.flaresolverr_url
        data["flaresolverr_timeout_sec"] = self.flaresolverr_timeout_sec
        data["browser_bridge_url"] = self.browser_bridge_url
        data["mihomo"] = self.get_mihomo_settings()
        data.pop("auth-key", None)
        return data

    def get_proxy_settings(self) -> str:
        return str(os.getenv("PROXY_URL") or self.data.get("proxy") or "").strip()

    @property
    def chatgpt_fingerprint(self) -> dict[str, object]:
        return _normalize_chatgpt_fingerprint(self.data.get("chatgpt_fingerprint"))

    @property
    def grok_console_fingerprint(self) -> dict[str, object]:
        return _normalize_grok_console_fingerprint(self.data.get("grok_console_fingerprint"))

    @property
    def network_profiles(self) -> dict[str, object]:
        return _normalize_network_profiles(self.data.get("network_profiles"))

    def update(self, data: dict[str, object]) -> dict[str, object]:
        next_data = dict(self.data)
        next_data.update(_persistent_settings(dict(data or {})))
        if "chatgpt_fingerprint" in next_data:
            next_data["chatgpt_fingerprint"] = _normalize_chatgpt_fingerprint(next_data.get("chatgpt_fingerprint"))
        if "grok_console_fingerprint" in next_data:
            next_data["grok_console_fingerprint"] = _normalize_grok_console_fingerprint(next_data.get("grok_console_fingerprint"))
        if "network_profiles" in next_data:
            next_data["network_profiles"] = _normalize_network_profiles(next_data.get("network_profiles"))
        if "backup" in next_data:
            next_data["backup"] = _normalize_backup_settings(next_data.get("backup"))
        if "image_storage" in next_data:
            next_data["image_storage"] = _normalize_image_storage_settings(next_data.get("image_storage"))
            _validate_image_storage_settings(next_data["image_storage"])
        if "chat_completion_cache" in next_data:
            next_data["chat_completion_cache"] = _normalize_chat_completion_cache_settings(next_data.get("chat_completion_cache"))
        if "chat_completion_message_normalization" in next_data:
            next_data["chat_completion_message_normalization"] = _normalize_chat_completion_message_normalization_settings(next_data.get("chat_completion_message_normalization"))
        if "mihomo" in next_data:
            next_data["mihomo"] = _normalize_mihomo_settings(next_data.get("mihomo"))
        next_data.pop("backup_state", None)
        self.data = next_data
        self._save()
        return self.get()

    def get_backup_settings(self) -> dict[str, object]:
        return _normalize_backup_settings(self.data.get("backup"))

    def get_mihomo_settings(self) -> dict[str, object]:
        return _normalize_mihomo_settings(self.data.get("mihomo"))

    @property
    def enable_turnstile_solver(self) -> bool:
        value = self.data.get("enable_turnstile_solver", True)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def get_image_storage_settings(self) -> dict[str, object]:
        return _normalize_image_storage_settings(self.data.get("image_storage"))

    def get_chat_completion_cache_settings(self) -> dict[str, object]:
        return _normalize_chat_completion_cache_settings(self.data.get("chat_completion_cache"))

    def get_chat_completion_message_normalization_settings(self) -> dict[str, object]:
        return _normalize_chat_completion_message_normalization_settings(self.data.get("chat_completion_message_normalization"))

    def get_storage_backend(self) -> StorageBackend:
        """获取存储后端实例（单例）"""
        if self._storage_backend is None:
            from services.storage.factory import create_storage_backend
            self._storage_backend = create_storage_backend(DATA_DIR)
        return self._storage_backend


def load_backup_state() -> dict[str, object]:
    return _normalize_backup_state(_read_json_object(BACKUP_STATE_FILE, name="backup_state.json"))


def save_backup_state(state: dict[str, object]) -> dict[str, object]:
    normalized = _normalize_backup_state(state)
    BACKUP_STATE_FILE.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return normalized


config = ConfigStore(CONFIG_FILE)
