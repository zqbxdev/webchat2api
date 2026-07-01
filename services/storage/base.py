from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StorageBackend(ABC):
    """抽象存储后端基类"""

    @abstractmethod
    def load_accounts(self) -> list[dict[str, Any]]:
        """加载所有账号数据"""
        pass

    @abstractmethod
    def save_accounts(self, accounts: list[dict[str, Any]]) -> None:
        """保存所有账号数据"""
        pass

    @abstractmethod
    def load_auth_keys(self) -> list[dict[str, Any]]:
        """加载所有鉴权密钥数据"""
        pass

    @abstractmethod
    def save_auth_keys(self, auth_keys: list[dict[str, Any]]) -> None:
        """保存所有鉴权密钥数据"""
        pass

    @abstractmethod
    def load_settings(self) -> dict[str, Any]:
        """加载全局设置"""
        pass

    @abstractmethod
    def save_settings(self, settings: dict[str, Any]) -> None:
        """保存全局设置"""
        pass

    @abstractmethod
    def load_proxy_pool(self) -> list[dict[str, Any]]:
        """加载代理池数据"""
        pass

    @abstractmethod
    def save_proxy_pool(self, items: list[dict[str, Any]]) -> None:
        """保存代理池数据"""
        pass

    @abstractmethod
    def load_subscriptions(self) -> list[dict[str, Any]]:
        """加载订阅源数据"""
        pass

    @abstractmethod
    def save_subscriptions(self, items: list[dict[str, Any]]) -> None:
        """保存订阅源数据"""
        pass

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """健康检查，返回存储后端状态"""
        pass

    @abstractmethod
    def get_backend_info(self) -> dict[str, Any]:
        """获取存储后端信息"""
        pass
