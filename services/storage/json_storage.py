from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.storage.base import StorageBackend


class JSONStorageBackend(StorageBackend):
    """本地 JSON 文件存储后端"""

    def __init__(self, file_path: Path, auth_keys_path: Path | None = None, settings_path: Path | None = None):
        self.file_path = file_path
        self.auth_keys_path = auth_keys_path or file_path.with_name("auth_keys.json")
        self.settings_path = settings_path or file_path.with_name("settings.json")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.auth_keys_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_json_list(file_path: Path) -> list[dict[str, Any]]:
        if not file_path.exists():
            return []
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, Exception):
            return []

    @staticmethod
    def _save_json_list(file_path: Path, items: list[dict[str, Any]]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _load_json_object(file_path: Path) -> dict[str, Any]:
        if not file_path.exists():
            return {}
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, Exception):
            return {}

    @staticmethod
    def _save_json_object(file_path: Path, data: dict[str, Any]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def load_accounts(self) -> list[dict[str, Any]]:
        """从 JSON 文件加载账号数据"""
        return self._load_json_list(self.file_path)

    def save_accounts(self, accounts: list[dict[str, Any]]) -> None:
        """保存账号数据到 JSON 文件"""
        self._save_json_list(self.file_path, accounts)

    def load_auth_keys(self) -> list[dict[str, Any]]:
        """从 JSON 文件加载鉴权密钥数据"""
        if not self.auth_keys_path.exists():
            return []
        try:
            data = json.loads(self.auth_keys_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            return []
        if isinstance(data, dict):
            data = data.get("items")
        return data if isinstance(data, list) else []

    def save_auth_keys(self, auth_keys: list[dict[str, Any]]) -> None:
        """保存鉴权密钥数据到 JSON 文件"""
        self.auth_keys_path.parent.mkdir(parents=True, exist_ok=True)
        self.auth_keys_path.write_text(
            json.dumps({"items": auth_keys}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def load_proxy_pool(self) -> list[dict[str, Any]]:
        """从 JSON 文件加载代理池数据"""
        return self._load_json_list(self.file_path.with_name("proxy_pool.json"))

    def save_proxy_pool(self, items: list[dict[str, Any]]) -> None:
        """保存代理池数据到 JSON 文件"""
        self._save_json_list(self.file_path.with_name("proxy_pool.json"), items)

    def load_subscriptions(self) -> list[dict[str, Any]]:
        """从 JSON 文件加载订阅源数据"""
        return self._load_json_list(self.file_path.with_name("subscriptions.json"))

    def save_subscriptions(self, items: list[dict[str, Any]]) -> None:
        """保存订阅源数据到 JSON 文件"""
        self._save_json_list(self.file_path.with_name("subscriptions.json"), items)

    def load_settings(self) -> dict[str, Any]:
        """从 JSON 文件加载全局设置"""
        return self._load_json_object(self.settings_path)

    def save_settings(self, settings: dict[str, Any]) -> None:
        """保存全局设置到 JSON 文件"""
        self._save_json_object(self.settings_path, settings)

    def health_check(self) -> dict[str, Any]:
        """健康检查"""
        try:
            # 检查文件是否可读写
            if self.file_path.exists():
                self.file_path.read_text(encoding="utf-8")
            return {
                "status": "healthy",
                "backend": "json",
                "file_exists": self.file_path.exists(),
                "file_path": str(self.file_path),
                "auth_keys_file_exists": self.auth_keys_path.exists(),
                "auth_keys_file_path": str(self.auth_keys_path),
                "settings_file_exists": self.settings_path.exists(),
                "settings_file_path": str(self.settings_path),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "json",
                "error": str(e),
            }

    def get_backend_info(self) -> dict[str, Any]:
        """获取存储后端信息"""
        return {
            "type": "json",
            "description": "本地 JSON 文件存储",
            "file_path": str(self.file_path),
            "file_exists": self.file_path.exists(),
            "auth_keys_file_path": str(self.auth_keys_path),
            "auth_keys_file_exists": self.auth_keys_path.exists(),
            "settings_file_path": str(self.settings_path),
            "settings_file_exists": self.settings_path.exists(),
        }
