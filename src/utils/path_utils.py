"""统一路径工具：所有路径从此模块取，禁止硬编码本机绝对路径。
规则（对齐分步执行计划 M0 + M6 打包场景）：
1. 任何代码中不得出现 `os.getcwd()`、形如 `C:\\Users\\` 的用户名硬编码。
2. 路径分两类，绝不能混：
   - **资源路径（只读，打包进 exe）**：ASSETS_DIR / resolve_resource()
     * frozen 模式：sys._MEIPASS（exe 解压目录）
     * 开发模式：PROJECT_ROOT
   - **用户数据路径（可写，长期保留）**：DATA_ROOT 系列
     * frozen onefile / onedir 模式：exe 同级目录（PROJECT_ROOT = sys.executable.parent）
     * 开发模式：项目根
3. 运行期目录首次启动由 ensure_data_dirs() 一次性创建（幂等，exist_ok=True）。
4. PyInstaller frozen 判断：getattr(sys, 'frozen', False) or hasattr(sys, '_MEIPASS')
"""
from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path
from typing import Final


# ---------------------------------------------------------------------------
# 通用打包检测
# ---------------------------------------------------------------------------

def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包后的 exe 中（onefile 解压态 / onedir 都返回 True）。

    判定依据：sys.frozen 字段为 True OR sys 存在 _MEIPASS 属性。
    """
    return bool(getattr(sys, "frozen", False)) or hasattr(sys, "_MEIPASS")


def runtime_resources_dir() -> Path:
    """**只读资源** 根目录：打包后的资源都在 sys._MEIPASS。

    - frozen 模式 → sys._MEIPASS（exe 运行时解压目录，关闭程序即删）
    - 开发模式   → PROJECT_ROOT（项目根目录，assets/、.env、src/ 都在这里）
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).resolve()
    return PROJECT_ROOT


def resolve_resource(relative: str | os.PathLike[str]) -> Path:
    """解析一个打包资源的绝对路径（先找 runtime_resources_dir，找不到 fallback 到 PROJECT_ROOT）。

    典型用法：
        icon_path = resolve_resource("assets/icon.ico")
        style_qss = resolve_resource("assets/styles/dark.qss").read_text(encoding="utf-8")
    """
    rel = Path(relative)
    candidate = runtime_resources_dir() / rel
    if candidate.exists():
        return candidate
    fallback = PROJECT_ROOT / rel
    if fallback.exists():
        return fallback
    # 都不存在就返回 runtime_resources_dir/rel（让上层读取时抛出清晰 FileNotFound）
    return candidate


# ---------------------------------------------------------------------------
# 项目根目录（决定“用户数据放哪里”的基准点，非资源基准）
# ---------------------------------------------------------------------------

def _resolve_project_root() -> Path:
    """**用户可写数据** 的基准目录。
    - frozen 模式（onefile / onedir 都一样）：sys.executable 的父目录 = 用户 exe 所在目录
      * 这样用户把 .env 和 data/ 文件夹放到 exe 旁边，就能自动生效；换电脑拷走也没问题。
    - 开发模式：src/utils/path_utils.py → 向上两级 = 项目仓库根
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Final[Path] = _resolve_project_root()


# ---------------------------------------------------------------------------
# 用户可写路径（长期保留，不能放 sys._MEIPASS 否则重启即丢）
# ---------------------------------------------------------------------------

# .env 文件：用户习惯放 exe 旁边（PROJECT_ROOT）；如不存在则从打包资源复制一份默认模板
ENV_FILE_PATH: Final[Path] = PROJECT_ROOT / ".env"

# 数据根目录：优先读环境变量 APP_WORKDIR，默认 <PROJECT_ROOT>/data
_data_workdir = os.getenv("APP_WORKDIR") or str(PROJECT_ROOT / "data")
DATA_ROOT: Final[Path] = Path(_data_workdir).resolve()

CHECKPOINTS_DIR: Final[Path] = DATA_ROOT / "checkpoints"      # LangGraph SQLite Checkpoint
HISTORY_DB_DIR: Final[Path] = DATA_ROOT / "history"          # 对话历史 SQLite 存放
WHOOSH_INDEX_DIR: Final[Path] = DATA_ROOT / "whoosh_index"    # 文件全文索引
OCR_TEMP_DIR: Final[Path] = DATA_ROOT / "ocr_temp"            # OCR 临时图片（截图/PaddleOCR）
TTS_CACHE_DIR: Final[Path] = DATA_ROOT / "tts_cache"          # Edge-TTS 下载音频缓存

# SQLite 主库路径（设置 + 对话历史，单一 db 文件减少连接数）
APP_DB_PATH: Final[Path] = DATA_ROOT / "app.db"


# ---------------------------------------------------------------------------
# 资源路径（只读，frozen 时在 sys._MEIPASS）
# ---------------------------------------------------------------------------

# ASSETS_DIR 注意：优先从 runtime_resources_dir/assets 找（onefile 打包的资源在这里）
# 不存在才 fallback 到 PROJECT_ROOT/assets（开发/onedir 模式）
def _resolve_assets_dir() -> Path:
    if is_frozen():
        p = runtime_resources_dir() / "assets"
        if p.exists():
            return p
    return PROJECT_ROOT / "assets"


ASSETS_DIR: Final[Path] = _resolve_assets_dir()


# ---------------------------------------------------------------------------
# 启动期辅助：目录创建 + .env 模板复制（打包后用户第一次双击 exe 也能拿到 .env）
# ---------------------------------------------------------------------------

def ensure_env_file() -> None:
    """打包场景：当 PROJECT_ROOT/.env 不存在时，从资源目录复制一份默认模板（若有）。

    开发态啥也不做（自己编辑 PROJECT_ROOT/.env 填 Key）。
    """
    if ENV_FILE_PATH.exists() or not is_frozen():
        return
    src_env = resolve_resource(".env")
    if src_env.exists() and src_env.is_file():
        try:
            shutil.copy2(src_env, ENV_FILE_PATH)
        except Exception:  # noqa: BLE001
            # 复制失败不影响启动（用户可手动创建 .env），静默过
            pass


def ensure_data_dirs() -> None:
    """启动时一次性创建所有运行期目录。幂等：重复调用无副作用。

    同时确保 .env 就位，后续 main.py 启动直接调用本函数即可。
    """
    for _d in (
        DATA_ROOT,
        CHECKPOINTS_DIR,
        HISTORY_DB_DIR,
        WHOOSH_INDEX_DIR,
        OCR_TEMP_DIR,
        TTS_CACHE_DIR,
        ASSETS_DIR,
    ):
        _d.mkdir(parents=True, exist_ok=True)
    ensure_env_file()
