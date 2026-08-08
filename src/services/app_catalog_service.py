"""本机应用目录服务：扫描桌面 / 开始菜单快捷方式，构建「应用名 → 可启动目标」目录。

数据来源（按可信度排序）：
    1. 桌面 .lnk（用户桌面 + 公共桌面）—— 用户最常用的应用都在这
    2. 开始菜单 .lnk（用户 + 公共开始菜单 Programs）
    3. 常见绿色软件目录暂不扫描（避免误扫一堆 dll/exe）

.lnk 解析：comtypes 调 Windows 原生 WScript.Shell COM（comtypes 已随 pycaw 安装，零新增依赖）；
解析失败时退化为「直接用 os.startfile 打开 .lnk 本身」（Windows 会代理解析，照样能启动）。

缓存：构建结果写 data/app_catalog.json（含时间戳），默认直接用缓存；
调用方传 refresh=True 才重建（扫描全盘开始菜单约 1~3 秒）。

匹配策略（find_app）：
    规范化（小写/去空格/去后缀）后：完全相等 > 前缀匹配 > 双向包含；命中多个按名称长度升序（更精确的排前面）。
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

_SRC_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.utils.path_utils import DATA_ROOT, ensure_data_dirs  # noqa: E402

_CATALOG_FILE = "app_catalog.json"
_NORMALIZE_RE = re.compile(r"[\s\-_（）()\[\]【】.]+")

# 常见系统/无意义快捷方式名黑名单（不收录进目录）
_NAME_BLACKLIST = {
    "卸载", "uninstall", "官网", "官方网站", "帮助", "help", "readme",
    "更新", "update", "设置", "settings", "license", "许可证",
}


def _norm(name: str) -> str:
    return _NORMALIZE_RE.sub("", (name or "").lower())


def _scan_dirs() -> list[Path]:
    """要扫描的目录列表（存在的才返回）。"""
    home = Path.home()
    candidates = [
        home / "Desktop",                                    # 用户桌面
        Path(r"C:\Users\Public\Desktop"),                    # 公共桌面
        home / r"AppData\Roaming\Microsoft\Windows\Start Menu\Programs",   # 用户开始菜单
        Path(r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs"),     # 公共开始菜单
    ]
    return [d for d in candidates if d.is_dir()]


def _resolve_lnk(lnk_path: Path) -> tuple[str, str]:
    """解析 .lnk → (目标 exe 路径, 工作目录)。失败返回 ("", "")。

    两级策略：
        1. comtypes 动态绑定调 WScript.Shell COM（必须 dynamic=True，否则属性访问报 AttributeError）
        2. 二进制兜底：直接从 .lnk 字节流里提取 ASCII / UTF-16LE 的绝对路径字符串
    """
    # —— 1. COM 动态绑定 ——
    try:
        import comtypes.client  # type: ignore
        shell = comtypes.client.CreateObject("WScript.Shell", dynamic=True)
        sc = shell.CreateShortcut(str(lnk_path))
        target = str(sc.TargetPath or "")
        workdir = str(sc.WorkingDirectory or "")
        if target:
            return target, workdir
    except Exception:  # noqa: BLE001
        pass
    # —— 2. 二进制提取兜底 ——
    try:
        data = lnk_path.read_bytes()
        # ASCII 路径
        for m in re.findall(rb"[\x20-\x7e]{6,}", data):
            s = m.decode("ascii", "ignore")
            if re.match(r"^[A-Za-z]:\\", s) and s.lower().endswith((".exe", ".bat", ".cmd", ".msc")):
                return s, ""
        # UTF-16LE 路径
        for m in re.findall(rb"(?:[\x20-\x7e]\x00){6,}", data):
            s = m.decode("utf-16le", "ignore")
            if re.match(r"^[A-Za-z]:\\", s) and s.lower().endswith((".exe", ".bat", ".cmd", ".msc")):
                return s, ""
    except Exception:  # noqa: BLE001
        pass
    return "", ""


# Windows 内置系统应用（不在快捷方式目录里，但用户常会命令打开）
_BUILTIN_APPS: dict[str, str] = {
    "记事本": "notepad.exe",
    "画图": "mspaint.exe",
    "计算器": "calc.exe",
    "命令提示符": "cmd.exe",
    "任务管理器": "taskmgr.exe",
    "控制面板": "control.exe",
    "资源管理器": "explorer.exe",
    "文件管理器": "explorer.exe",
    "截图工具": "SnippingTool.exe",
    "注册表编辑器": "regedit.exe",
    "写字板": "write.exe",
    "放大镜": "magnify.exe",
    "屏幕键盘": "osk.exe",
    "录音机": "soundrecorder.exe",
    "远程桌面": "mstsc.exe",
    "字符映射表": "charmap.exe",
}


class AppCatalogService:
    """应用目录：构建 / 查询 / 持久化。单例使用。"""

    def __init__(self) -> None:
        self._apps: dict[str, dict[str, str]] = {}   # {显示名: {"path":..., "lnk":..., "source":...}}
        self._built_at: float = 0.0
        self._load_cache()

    # ------------------------------------------------------------------
    # 构建 / 缓存
    # ------------------------------------------------------------------
    def _cache_path(self) -> Path:
        ensure_data_dirs()
        return DATA_ROOT / _CATALOG_FILE

    def _load_cache(self) -> None:
        p = self._cache_path()
        if not p.is_file():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            apps = data.get("apps")
            if isinstance(apps, dict) and apps:
                self._apps = apps
                self._built_at = float(data.get("built_at", 0.0))
        except Exception:  # noqa: BLE001
            pass

    def _save_cache(self) -> None:
        try:
            self._cache_path().write_text(
                json.dumps({"built_at": self._built_at, "apps": self._apps}, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

    def build(self, refresh: bool = False) -> int:
        """扫描并构建目录。refresh=False 且有缓存时直接用缓存。返回收录应用数。"""
        if self._apps and not refresh:
            return len(self._apps)
        apps: dict[str, dict[str, str]] = {}
        for d in _scan_dirs():
            source = "desktop" if "Desktop" in d.parts else "start_menu"
            try:
                lnks = list(d.rglob("*.lnk"))
            except Exception:  # noqa: BLE001
                continue
            for lnk in lnks:
                name = lnk.stem.strip()
                if not name or _norm(name) in {_norm(b) for b in _NAME_BLACKLIST}:
                    continue
                if any(_norm(name).startswith(_norm(b)) for b in ("卸载", "uninstall")):
                    continue
                target, _wd = _resolve_lnk(lnk)
                entry = {
                    "path": target,           # 解析出的真实 exe（可能为空）
                    "lnk": str(lnk),          # 快捷方式本身（os.startfile 可用）
                    "source": source,
                }
                # 同名去重：桌面优先于开始菜单
                if name not in apps or source == "desktop":
                    apps[name] = entry
        # 合入 Windows 内置系统应用（记事本/计算器等，不在快捷方式目录里）
        for name, exe in _BUILTIN_APPS.items():
            if name not in apps:
                apps[name] = {"path": exe, "lnk": "", "source": "builtin"}
        self._apps = apps
        self._built_at = time.time()
        self._save_cache()
        return len(apps)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @property
    def app_count(self) -> int:
        return len(self._apps)

    def all_names(self) -> list[str]:
        return sorted(self._apps.keys(), key=str.lower)

    def find_app(self, query: str, max_candidates: int = 8) -> tuple[Optional[dict[str, Any]], list[str]]:
        """模糊查找应用。

        Returns:
            (best_entry_or_None, candidates): best_entry 含 name/path/lnk；
            无精确命中时 candidates 给出最接近的名字列表（供用户重选）。
        """
        if not self._apps:
            self.build()
        q = _norm(query)
        if not q:
            return None, []

        scored: list[tuple[int, str]] = []   # (优先级分数越小越好, 名称)
        for name in self._apps:
            n = _norm(name)
            if not n:
                continue
            if n == q:
                scored.append((0, name))
            elif n.startswith(q) or q.startswith(n):
                scored.append((1, name))
            elif q in n or n in q:
                scored.append((2, name))
        scored.sort(key=lambda x: (x[0], len(x[1]), x[1].lower()))
        if not scored:
            return None, []
        best_score, best_name = scored[0]
        candidates = [name for _s, name in scored[:max_candidates]]
        if best_score <= 1:
            entry = dict(self._apps[best_name])
            entry["name"] = best_name
            return entry, candidates
        # 只有「包含」级命中：也算命中，但在结果里把候选带上让 LLM 知情
        entry = dict(self._apps[best_name])
        entry["name"] = best_name
        return entry, candidates


# ------------------------------------------------------------------
# 模块级单例
# ------------------------------------------------------------------
_CATALOG = AppCatalogService()


def get_app_catalog() -> AppCatalogService:
    return _CATALOG


__all__ = ["AppCatalogService", "get_app_catalog"]
