"""文件操作工具（LangChain BaseTool 子类实现）。

功能列表：
    1. CreateFileTool   → 创建/覆盖文本文件（UTF-8）；自动建父目录；越界安全校验；路径别名解析。
    2. SearchFilesTool  → 按文件名关键字 / 通配 搜索（当前简单实现，M3 接 Whoosh 全文索引）。

所有工具统一约定：
    * 输入路径支持「路径别名」，避免 LLM 硬编码绝对路径（对齐经验 551286：禁止跨机器绝对路径硬编码）。
    * 工具 _run() 内部捕获所有异常，返回可读 observation 字符串，绝不向上抛（ReAct 让 LLM 自纠错）。
    * 路径「越界 / 白名单外」一律拒绝执行，返回明确错误原因。
"""
from __future__ import annotations

import fnmatch
import os
import sys
import time
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

# 确保能 import src.*
_SRC_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from langchain_core.tools import BaseTool  # noqa: E402

from src.utils.path_utils import DATA_ROOT, PROJECT_ROOT  # noqa: E402


# ============================================================
# 公用：路径别名解析 + 安全校验（防止越界访问 C:\Windows 等）
# ============================================================

def _user_home() -> Path:
    """跨平台（Windows/Linux/Mac）拿用户家目录。"""
    return Path(os.path.expanduser("~"))


# —— 路径别名映射：允许 LLM 用口语化名称而不是硬编码绝对路径 ——
_PATH_ALIASES: dict[str, Path] = {
    # 中文口语 / 英文通用
    "桌面": _user_home() / "Desktop",
    "Desktop": _user_home() / "Desktop",
    "文档": _user_home() / "Documents",
    "我的文档": _user_home() / "Documents",
    "Documents": _user_home() / "Documents",
    "下载": _user_home() / "Downloads",
    "Downloads": _user_home() / "Downloads",
    # 工程目录
    "项目根": PROJECT_ROOT,
    "project": PROJECT_ROOT,
    "src_root": PROJECT_ROOT,
    # 运行期数据目录（默认根）
    "数据根": DATA_ROOT,
    "data": DATA_ROOT,
    "运行目录": DATA_ROOT,
    # 家目录
    "家": _user_home(),
    "~": _user_home(),
    "home": _user_home(),
}


# —— 允许写入 / 搜索的白名单前缀（按绝对路径） ——
_ALLOWED_PREFIXES: tuple[Path, ...] = (
    _user_home(),
    PROJECT_ROOT,
    DATA_ROOT,
    _user_home() / "Desktop",
    _user_home() / "Documents",
    _user_home() / "Downloads",
)


# —— 永远禁止访问的黑名单前缀（哪怕路径别名指到这里也拦） ——
_BLOCKED_PREFIXES_LOWER: tuple[str, ...] = (
    "c:\\windows",
    "c:\\program files",
    "c:\\program files (x86)",
    "c:\\programdata",
    "c:\\recovery",
    "c:\\system volume information",
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/var",
    "/system",
)


def _is_blocked(abs_path: Path) -> bool:
    """黑名单前缀匹配（大小写不敏感，Windows 斜杠归一）。"""
    s = str(abs_path.resolve()).lower().replace("/", "\\")
    return any(s.startswith(bp) for bp in _BLOCKED_PREFIXES_LOWER)


def _is_allowed(abs_path: Path) -> bool:
    """白名单前缀匹配。"""
    resolved = abs_path.resolve()
    for pre in _ALLOWED_PREFIXES:
        try:
            resolved.relative_to(pre.resolve())
            return True
        except ValueError:
            continue
    return False


def resolve_user_path(raw_path: str, default_root: Path = DATA_ROOT) -> Path:
    """把 LLM 给出的「原始路径字符串」解析为真实绝对 Path。
    解析优先级：
        1. 完全等于 路径别名 键 → 直接用对应 Path
        2. 以「别名/」前缀开始 → 前缀替换
        3. 绝对路径（盘符开头 / 斜杠开头）→ 直接用
        4. 相对路径 → 以 default_root 为根拼接
    """
    if not raw_path:
        raise ValueError("路径不能为空")
    raw = raw_path.strip().strip('"').strip("'")

    # 1. 完全别名匹配
    if raw in _PATH_ALIASES:
        return _PATH_ALIASES[raw]

    # 2. 前缀匹配：例如 "桌面/工作/a.txt" → 替换「桌面」前缀
    for alias, target in _PATH_ALIASES.items():
        sep1 = alias + os.sep
        sep2 = alias + "/"
        if raw.startswith(sep1):
            return target / raw[len(sep1):]
        if raw.startswith(sep2):
            return target / raw[len(sep2):]

    # 3. 绝对路径
    if raw.startswith(("\\\\", "//")) or (len(raw) >= 2 and raw[1] == ":") or raw.startswith("/"):
        return Path(raw)

    # 4. 相对路径（默认根 DATA_ROOT，安全不碰代码目录）
    return (default_root / raw).resolve()


def assert_safe_path(abs_path: Path, *, writable: bool) -> None:
    """越界 / 黑名单安全校验，不满足 raise ValueError（由上层 _run 捕获返回 observation）。"""
    if writable:
        # 写操作额外限制：不允许直接写 PROJECT_ROOT 的 src/ 下代码（防止 LLM 瞎改代码）
        try:
            abs_path.resolve().relative_to((PROJECT_ROOT / "src").resolve())
            raise ValueError(
                f"拒绝写入 {abs_path}：禁止直接修改 src/ 代码目录。"
                "如需创建业务文件，请指定路径为「数据根/xxx」「桌面/xxx」「文档/xxx」。"
            )
        except ValueError as e:
            if "拒绝写入" in str(e):
                raise
            # 不是 src/ 下面 → 放行
            pass

    if _is_blocked(abs_path):
        raise ValueError(f"拒绝访问 {abs_path}：路径位于系统黑名单（Windows/Program Files 等）。")
    if not _is_allowed(abs_path):
        raise ValueError(
            f"拒绝访问 {abs_path}：路径不在白名单内。"
            f"白名单目录：家目录/项目根/数据根/桌面/文档/下载。"
        )


# ============================================================
# Tool 1: CreateFileTool —— 创建/覆盖 UTF-8 文本文件
# ============================================================

class CreateFileArgs(BaseModel):
    """CreateFileTool 的参数 Schema（Pydantic，Field description 全中文，给 LLM 看明白）。"""

    file_path: str = Field(
        ...,
        min_length=1,
        description=(
            "【必填】目标文件路径。支持中文别名开头："
            "「桌面/xxx.txt」「文档/xxx.md」「下载/xxx」「数据根/xxx」「项目根/xxx」"
            "；也可写绝对路径或相对路径（相对路径默认放在 数据根 下）。"
            "示例：数据根/我的第一个文件.txt"
        ),
    )
    content: str = Field(
        ...,
        description="【必填】文件内容（UTF-8 文本）。支持中文、换行符、Markdown 等任意文本字符。",
    )
    overwrite: bool = Field(
        False,
        description=(
            "文件已存在时是否覆盖（默认 False：不覆盖并返回错误提示，需要 LLM 显式传 overwrite=True 才会覆盖）。"
        ),
    )


class CreateFileTool(BaseTool):
    """创建 UTF-8 文本文件。适用场景：新建 Markdown 笔记、新建 txt 待办、导出对话、保存搜索结果等。
    自动创建缺失的父目录；执行前会进行「黑名单 + 白名单 + 禁止直接改 src/ 代码目录」三层安全校验。
    """

    name: ClassVar[str] = "create_file"
    description: ClassVar[str] = (
        "Tool Name: create_file\n"
        "用途：在白名单目录（桌面 / 文档 / 下载 / 数据根 / 项目根）内创建新的 UTF-8 文本文件。\n"
        "支持路径别名（中文友好）：例如 file_path 写 「桌面/工作周报.md」「数据根/2026-08/会议记录.txt」。\n"
        "注意事项：\n"
        "  1. 严禁修改 C:\\Windows、C:\\Program Files 等系统目录（已拉黑，会被直接拒绝）。\n"
        "  2. 严禁直接写入 PROJECT_ROOT/src 代码目录（避免 LLM 误修改项目代码）。\n"
        "  3. 文件已存在但 overwrite=False（默认）时，不会覆盖，会返回错误信息让你重新调 create_file + overwrite=True。\n"
        "  4. 父目录不存在会自动创建（mkdir -p 语义）。\n"
    )
    args_schema: type[BaseModel] = CreateFileArgs
    return_direct: ClassVar[bool] = False

    # —— 实际实现 ——
    def _run(self, file_path: str, content: str, overwrite: bool = False) -> str:  # noqa: D401
        t0 = time.perf_counter_ns()
        try:
            abs_path = resolve_user_path(file_path)
            assert_safe_path(abs_path, writable=True)

            if abs_path.exists() and abs_path.is_dir():
                raise ValueError(f"路径 {abs_path} 已存在且是一个目录，不能作为文件写入。")

            if abs_path.exists() and not overwrite:
                raise FileExistsError(
                    f"文件 {abs_path} 已存在。若要覆盖，请再次调用 create_file 并传 overwrite=True。"
                )

            # 自动建父目录
            abs_path.parent.mkdir(parents=True, exist_ok=True)

            # 写入 UTF-8（UTF-8-SIG 不写 BOM，跨平台兼容性最好）
            abs_path.write_text(content, encoding="utf-8")
            size = abs_path.stat().st_size
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            return (
                f"✅ create_file 成功\n"
                f"  真实路径：{abs_path}\n"
                f"  文件大小：{size} 字节（{size/1024:.2f} KB）\n"
                f"  行数    ：{content.count(chr(10)) + 1}\n"
                f"  耗时    ：{elapsed_ms} ms"
            )
        except Exception as e:  # noqa: BLE001 —— 工具所有异常都包装成 observation 返回给 LLM
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ create_file 失败（{elapsed_ms} ms）：{type(e).__name__}: {e}"


# ============================================================
# Tool 2: SearchFilesTool —— 按文件名关键字 / 通配搜索
# ============================================================

class SearchFilesArgs(BaseModel):
    """SearchFilesTool 参数 Schema。"""

    query: str = Field(
        ...,
        min_length=1,
        description=(
            "【必填】文件名搜索条件，支持两种写法：\n"
            "  A. 关键字匹配：例如 周报 → 文件名包含「周报」不分大小写\n"
            "  B. 通配符匹配：例如 *.md / *2026*.txt / report_*.xlsx"
        ),
    )
    search_root: str = Field(
        "白名单默认",
        description=(
            "【选填】搜索根目录（路径别名），默认值「白名单默认」= 同时搜：桌面 + 文档 + 下载 + 数据根。\n"
            "也可单独指定：桌面 / 文档 / 下载 / 数据根 / 项目根 / 家目录。"
        ),
    )
    max_results: int = Field(
        20,
        ge=1,
        le=200,
        description="【选填】最多返回多少条结果（1~200，默认 20）。过多会自动截断。",
    )


class SearchFilesTool(BaseTool):
    """按文件名搜索（通配 / 关键字不区分大小写）。
    当前实现：文件系统遍历 rglob（简单够用）；M3 阶段接入 Whoosh 全文索引（按内容搜索）。
    """

    name: ClassVar[str] = "search_files"
    description: ClassVar[str] = (
        "Tool Name: search_files\n"
        "用途：在白名单目录里按文件名找文件（支持 * 通配 和 不区分大小写的关键字）。\n"
        "典型场景：\n"
        "  - 帮我找一下桌面上带「周报」的文件\n"
        "  - 搜一下最近的 Markdown 笔记（query=*.md，search_root=文档）\n"
        "  - 找下载目录里的 PDF（query=*.pdf，search_root=下载）\n"
        "注意：当前仅按【文件名】搜索；按文件【内容】搜索将在后续版本接入全文索引。"
    )
    args_schema: type[BaseModel] = SearchFilesArgs
    return_direct: ClassVar[bool] = False

    # —— 搜索根展开 ——
    @staticmethod
    def _expand_search_roots(search_root_raw: str) -> list[Path]:
        raw = search_root_raw.strip() or "白名单默认"
        if raw in ("白名单默认", "默认", "all"):
            return [
                _user_home() / "Desktop",
                _user_home() / "Documents",
                _user_home() / "Downloads",
                DATA_ROOT,
            ]
        # 单个别名
        resolved = resolve_user_path(raw)
        return [resolved]

    def _run(self, query: str, search_root: str = "白名单默认", max_results: int = 20) -> str:  # noqa: D401
        t0 = time.perf_counter_ns()
        try:
            q = query.strip()
            if not q:
                raise ValueError("query 不能为空")

            # 判断是否包含通配符
            has_wildcard = any(c in q for c in "*?[]")
            # 统一生成小写 pattern（不分大小写）
            pattern = q.lower() if not has_wildcard else None  # None = 用 fnmatch

            roots = self._expand_search_roots(search_root)
            results: list[tuple[Path, int, float]] = []  # (path, size_bytes, mtime)
            scanned_dirs = 0
            scanned_files = 0
            skipped_roots: list[str] = []

            for root in roots:
                try:
                    assert_safe_path(root, writable=False)
                except ValueError as e:
                    skipped_roots.append(f"{root}（{e}）")
                    continue
                if not root.exists() or not root.is_dir():
                    skipped_roots.append(f"{root}（目录不存在/不是目录）")
                    continue
                try:
                    for entry in root.rglob("*"):  # 递归所有
                        scanned_files += 1
                        if entry.is_dir():
                            scanned_dirs += 1
                            continue
                        if len(results) >= max_results + 50:  # 扫描阶段少少限流
                            break
                        name = entry.name
                        try:
                            if has_wildcard:
                                ok = fnmatch.fnmatch(name.lower(), q.lower())
                            else:
                                ok = pattern in name.lower()
                        except Exception:  # 非法文件名
                            continue
                        if not ok:
                            continue
                        try:
                            st = entry.stat()
                            results.append((entry, st.st_size, st.st_mtime))
                        except OSError:
                            results.append((entry, -1, 0.0))
                        if len(results) >= max_results + 50:
                            break
                except PermissionError:
                    skipped_roots.append(f"{root}（权限不足，部分子目录跳过）")
                except Exception as e:  # noqa: BLE001
                    skipped_roots.append(f"{root}（{type(e).__name__}: {e}）")

            # 按修改时间倒序
            results.sort(key=lambda x: x[2], reverse=True)
            total = len(results)
            results = results[:max_results]

            lines: list[str] = []
            lines.append(
                f"🔍 search_files 完成 | 条件=[{query}] | 根目录={[str(r) for r in roots]}"
                f" | 扫描文件数={scanned_files} | 匹配总数={total}"
            )
            if skipped_roots:
                lines.append(f"  跳过的根：{'; '.join(skipped_roots)}")
            if not results:
                lines.append("  （未找到任何匹配文件，可尝试扩大搜索根或修改匹配关键字）")
                elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
                lines.append(f"  耗时：{elapsed_ms} ms")
                return "\n".join(lines)

            lines.append(f"  Top {len(results)}（按最新修改时间排序）：")
            for i, (p, size, mt) in enumerate(results, start=1):
                size_str = (
                    f"{size/1024/1024:.2f} MB"
                    if size >= 1024 * 1024
                    else f"{size/1024:.2f} KB" if size >= 0 else "未知"
                )
                mtime_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mt)) if mt > 0 else "未知"
                lines.append(
                    f"    [{i}] {p}\n"
                    f"         大小={size_str}  修改时间={mtime_str}"
                )
            if total > max_results:
                lines.append(
                    f"  ⚠️ 结果截断：共 {total} 个匹配，只返回前 {max_results}。"
                    f"请缩小 query 或 search_root 后再搜索。"
                )
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            lines.append(f"  耗时：{elapsed_ms} ms")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ search_files 失败（{elapsed_ms} ms）：{type(e).__name__}: {e}"


__all__ = [
    "CreateFileTool",
    "SearchFilesTool",
    "DeleteFileTool",
    "RecognizeFileTool",
    "resolve_user_path",
    "assert_safe_path",
]


# ============================================================
# Tool 3: DeleteFileTool —— 删除文件/目录（高危操作：默认 dry_run 预览 + 需显式 confirm_keyword=DELETE）
# ============================================================

class DeleteFileArgs(BaseModel):
    """DeleteFileTool 参数 Schema（Pydantic，全部给 LLM 读得懂的中文说明）。"""

    target: str = Field(
        ...,
        min_length=1,
        description=(
            "【必填】删除目标：支持三种写法\n"
            "  1. 单个文件具体路径（别名/绝对/相对都可）：例 数据根/老文件.txt\n"
            "  2. 通配符批量：例 *.log、下载/*.zip、桌面/*临时*.docx\n"
            "  3. 单个目录路径：例 数据根/缓存\n"
            "⚠️  不允许直接写「项目根/src」「C:/Windows」等高危路径（三层安全校验会拦截）"
        ),
    )
    search_root: str = Field(
        "白名单默认",
        description="【选填】通配/相对路径的搜索根（默认 白名单=桌面/文档/下载/数据根）",
    )
    recursive: bool = Field(
        False,
        description="【选填】通配匹配时是否递归深入子目录（默认 False：只在 search_root 本级找；True=递归全子目录递归",
    )
    dry_run: bool = Field(
        True,
        description=(
            "【超级重要，默认 True】预览模式开关。\n"
            "  True 必须先调用 dry_run=True（默认）只返回「将删除 N 个文件/目录，共释放 XXX KB」的预览列表，不真删；\n"
            "  你确认无误后，第二次调用时才传 dry_run=False + confirm_keyword='DELETE' 才会真删。"
        ),
    )
    confirm_keyword: str | None = Field(
        None,
        description=(
            "【高危二次确认，dry_run=False 时必须 = 大写 'DELETE'】\n"
            "只有 dry_run=False 且 confirm_keyword='DELETE'（区分大小写），才会真执行删除；\n"
            "任何其他值（包括空、小写、少字）都会拒绝执行，避免误删。"
        ),
    )
    max_items: int = Field(
        100,
        ge=1,
        le=2000,
        description="【选填】单次最多允许删除多少项（文件+目录合计，默认100，上限2000）。超过会拒绝执行，防止通配写得太宽误删。",
    )


# —— 全局：项目绝对不允许删的「白名单内保护文件（即便在白名单目录，也绝对不删）——
_PROTECTED_FILE_PATTERNS_LOWER: tuple[str, ...] = (
    "app.db",          # SQLite 应用数据库
    ".env",            # 环境变量配置
    "settings.json",   # 设置文件
)
_PROTECTED_DIR_NAMES_LOWER: tuple[str, ...] = (
    "src", "venv_assistant", "venv", ".git",
    "assets", "logs",   # 代码目录/虚拟环境/git 目录
)


def _is_protected(p: Path) -> tuple[bool, str]:
    """判断路径是否是「受保护对象」（即便在白名单目录也不让删，返回 (True, 原因)）。"""
    resolved = p.resolve()
    name_low = resolved.name.lower()
    # 1. 受保护文件名（根目录的 .env / app.db）
    if name_low in _PROTECTED_FILE_PATTERNS_LOWER:
        return True, f"文件名 {resolved.name} 属于受保护核心文件（.env / app.db / settings.json 等）"
    # 2. 受保护目录名（任何层级的 src / .git / venv* 一律不删）
    for part in resolved.parts:
        if part.lower() in _PROTECTED_DIR_NAMES_LOWER:
            return True, f"路径层级包含受保护目录名 {part!r}（src/venv/.git 等）"
    # 3. PROJECT_ROOT 本身、PROJECT_ROOT 下的 data 子目录（不允许 rm -rf 项目根或数据根本体
    try:
        if resolved == PROJECT_ROOT.resolve():
            return True, "不能删除项目根目录本身"
        if resolved == DATA_ROOT.resolve():
            return True, "不能删除数据根目录本身（DATA_ROOT）"
        # 不允许删项目根的直接子目录（src / assets / tests 等）
        if resolved.parent.resolve() == PROJECT_ROOT.resolve():
            return True, "不能删除项目根下的顶层目录（src/assets/tests/venv 等，用更细的路径）"
    except Exception:  # noqa: BLE001
        pass
    return False, ""


def _size_fmt(b: int) -> str:
    """把字节数格式化为人类可读字符串（带单位）。"""
    if b < 0:
        return "未知"
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b/1024:.2f} KB"
    if b < 1024 * 1024 * 1024:
        return f"{b/1024/1024:.2f} MB"
    return f"{b/1024/1024/1024:.2f} GB"


def _walk_total_size(path: Path) -> int:
    """递归计算目录/文件总大小（字节）。"""
    try:
        if path.is_file():
            return path.stat().st_size
        total = 0
        for sub in path.rglob("*"):
            try:
                if sub.is_file():
                    total += sub.stat().st_size
            except OSError:
                continue
        return total
    except Exception:  # noqa: BLE001
        return -1


class DeleteFileTool(BaseTool):
    """删除文件/目录：默认 dry_run 预览，二次确认才真删。
    安全策略：3 层（黑名单路径→白名单前缀→受保护对象），配合 send2trash 回收站优先，避免硬删。
    """

    name: ClassVar[str] = "delete_file"
    description: ClassVar[str] = (
        "Tool Name: delete_file\n"
        "用途：删除文件/空目录/非空目录（批量通配删除）。\n"
        "⚠️  【高危操作，强制两次调用流程（必须严格遵守）：\n"
        "  第 1 次调用：默认 dry_run=True → 只返回待删除列表 + 总空间预览，**不真删任何东西**；\n"
        "  第 2 次调用：LLM 判断用户真确认了（例如用户说「确认删除」「删吧」等）→\n"
        "              传 dry_run=False + confirm_keyword='DELETE'（必须大写） 才会真删。\n"
        "  任何缺少 confirm_keyword 或不是大写 DELETE 的调用一律拒绝。\n"
        "三层安全拦截：\n"
        "  1) 路径黑名单（C:\\Windows / Program Files 等）→ 拒绝\n"
        "  2) 白名单前缀（家/项目/数据/桌面/文档/下载）→ 超出拒绝\n"
        "  3) 受保护对象（src/.git/venv/.env/app.db 等）→ 即便在白名单内也拒绝\n"
        "删除策略：\n"
        "  * 优先 send2trash（进回收站，若未安装 send2trash 则真删，真删前明确告诉用户）\n"
        "  * 单次删除项数上限 2000（防通配 * 写得太宽）\n"
        "  * 目录删除：必须 recursive=True 才能非空目录\n"
    )
    args_schema: type[BaseModel] = DeleteFileArgs
    return_direct: ClassVar[bool] = False

    @staticmethod
    def _expand_target(target_raw: str, search_root_raw: str, recursive: bool) -> list[Path]:
        """把 target（单路径/通配/目录 展开为真实存在的 Path 列表。"""
        # 1. 先尝试「完全匹配 = 具体路径存在？
        try:
            single = resolve_user_path(target_raw)
            if single.exists():
                return [single]
        except Exception:  # noqa: BLE001
            pass
        # 2. 否则 = 通配，在 search_root(s) 下 fnmatch 搜
        import fnmatch as _fnm
        q = target_raw.strip()
        roots = SearchFilesTool._expand_search_roots(search_root_raw)  # noqa: SLF001
        out: list[Path] = []
        seen: set[str] = set()
        has_wildcard = any(c in q for c in "*?[]")
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            try:
                if recursive:
                    it = root.rglob("*")
                else:
                    it = root.iterdir()
                for entry in it:
                    key = str(entry.resolve())
                    if key in seen:
                        continue
                    # 匹配：通配模式 or 纯关键字（不分大小写
                    if has_wildcard:
                        ok = _fnm.fnmatch(entry.name.lower(), q.lower())
                    else:
                        ok = q.lower() in entry.name.lower()
                    if ok:
                        seen.add(key)
                        out.append(entry)
            except (PermissionError, OSError):
                continue
        return out

    def _run(  # noqa: D401 —— 实际实现
        self,
        target: str,
        search_root: str = "白名单默认",
        recursive: bool = False,
        dry_run: bool = True,
        confirm_keyword: str | None = None,
        max_items: int = 100,
    ) -> str:
        t0 = time.perf_counter_ns()
        try:
            # —— 1. 二次确认门槛：dry_run=False 时，confirm_keyword 必须大写 DELETE
            if not dry_run:
                if (confirm_keyword or "").strip() != "DELETE":
                    raise PermissionError(
                        "高危操作二次确认失败：dry_run=False 时 confirm_keyword 必须精确等于大写字符串 'DELETE'（区分大小写）。\n"
                        "为避免误删，请先调用 dry_run=True 预览，确认无误后第二次调用传 dry_run=False confirm_keyword='DELETE'。"
                    )

            # —— 2. 展开路径 → 待处理列表
            candidates = self._expand_target(target, search_root, recursive)
            if not candidates:
                return (
                    f"🗑️ delete_file（{'DRY-RUN 预览模式' if dry_run else '⚠️ 真删模式'}：未找到任何匹配目标：target={target!r}"
                )

            # —— 3. 三层校验：逐个校验
            items: list[tuple[Path, int, bool, str]] = []  # (path, size, is_dir, protect_reason)
            protect_skipped: list[str] = []
            safe_skipped: list[str] = []
            for p in candidates:
                # 3a 安全：白/黑/前缀
                try:
                    assert_safe_path(p, writable=True)
                except ValueError as e:
                    safe_skipped.append(f"{p}（{e}）")
                    continue
                # 3b 保护：受保护对象（src/.git/.env 等）
                prot, reason = _is_protected(p)
                if prot:
                    protect_skipped.append(f"{p}（{reason}）")
                    continue
                size = _walk_total_size(p)
                items.append((p, size, p.is_dir(), ""))

            # —— 4. 数量限制
            if len(items) > max_items:
                raise ValueError(
                    f"待删除 {len(items)} 项超过 max_items={max_items}，为避免通配误删已拒绝。\n"
                    f"请缩小 target 范围或调大 max_items（最大 2000）。"
                )
            if not items and (safe_skipped or protect_skipped):
                lines = ["🗑️ delete_file：全部匹配都被安全拦截/保护，无真可删项："]
                if safe_skipped:
                    lines.append("  安全拒绝（白/黑名单/写 src 拦截）：")
                    lines += [f"    - {s}" for s in safe_skipped]
                if protect_skipped:
                    lines.append("  保护对象拒绝（src/.git/.env 等：")
                    lines += [f"    - {s}" for s in protect_skipped]
                elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
                lines.append(f"  耗时：{elapsed_ms} ms")
                return "\n".join(lines)

            # —— 5. 计算释放空间大小
            total_size = sum(max(0, it[1]) for it in items)
            files_count = sum(0 if it[2] else 1 for it in items)
            dirs_count = sum(1 if it[2] else 0 for it in items)
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            header = (
                f"🗑️ delete_file（{'✅ DRY-RUN 预览模式，未真删' if dry_run else f'⚠️ 真删模式 confirm=DELETE 已执行'}）\n"
                f"  target       = {target}\n"
                f"  search_root= {search_root}  recursive={recursive}\n"
                f"  合计删除数量 = 文件 {files_count} 个 + 目录 {dirs_count} 个  = {len(items)} 项\n"
                f"  预计/已释放空间 = {_size_fmt(total_size)}\n"
                f"  用时       = {elapsed_ms} ms\n"
            )
            lines: list[str] = [header]
            if safe_skipped:
                lines.append(f"  🔒 安全跳过（越界）：{len(safe_skipped)} 项")
                for s in safe_skipped[:5]:
                    lines.append(f"     - {s}")
                if len(safe_skipped) > 5:
                    lines.append(f"     …… 还有 {len(safe_skipped) - 5} 条")
            if protect_skipped:
                lines.append(f"  🛡️ 保护跳过：{len(protect_skipped)} 项")
                for s in protect_skipped[:5]:
                    lines.append(f"     - {s}")
                if len(protect_skipped) > 5:
                    lines.append(f"     …… 还有 {len(protect_skipped) - 5} 条")

            # —— 6. dry_run：只列预览（前 50 条，后面截
            lines.append(f"  📋 待删除列表（前 50 条）：")
            show_n = min(50, len(items))
            for i, (p, size, is_dir, _reason) in enumerate(items[:show_n], start=1):
                mark = "[DIR ]" if is_dir else "[FILE]"
                lines.append(f"    [{i:>2}] {mark} {p}   {_size_fmt(size)}")
            if len(items) > show_n:
                lines.append(f"    …… 其余 {len(items) - show_n} 条省略，确认前 {max_items} 条")

            if dry_run:
                lines.append(
                    "\n💡 下一步：确认用户要真删后，第二次调用：\n"
                    "   delete_file(target=同样的target，dry_run=False，confirm_keyword='DELETE'（必须大写）"
                )
                return "\n".join(lines)

            # —— 7. dry_run=False：真删
            import shutil as _shutil
            # 优先尝试 send2trash（回收站）
            try:
                from send2trash import send2trash  # type: ignore
                have_trash = True
            except Exception:  # noqa: BLE001
                have_trash = False
            deleted_ok: list[str] = []
            deleted_fail: list[str] = []
            method_note = "（send2trash 进回收站）" if have_trash else "（未安装 send2trash，已真删不可恢复）"
            for p, size, is_dir, _ in items:
                try:
                    if have_trash:
                        # 回收站：文件/空目录/非空目录 通吃，不需要区分
                        send2trash(str(p))
                    else:
                        # 真删：区分文件 / 空目录 / 非空目录
                        if not is_dir:
                            # 文件直接删
                            p.unlink()
                        else:
                            # 目录：先尝试 rmdir（空目录）
                            try:
                                p.rmdir()
                            except OSError:
                                # 非空目录：必须 recursive=True 才允许 rmtree
                                if not recursive:
                                    raise RuntimeError(
                                        f"目录 {p} 非空且 recursive=False，拒绝真删非空目录。"
                                        f"若确认真删，请传 recursive=True。"
                                    )
                                _shutil.rmtree(p, ignore_errors=False)
                    deleted_ok.append(f"{p}（{_size_fmt(size)}）")
                except Exception as e:  # noqa: BLE001
                    deleted_fail.append(f"{p} → {type(e).__name__}: {e}")
            lines.insert(1, f"  删除方式：{method_note}")
            lines.append(f"\n📊 真删结果：成功 {len(deleted_ok)} / 失败 {len(deleted_fail)}")
            if deleted_fail:
                lines.append("  失败项（前 10）：")
                for fail in deleted_fail[:10]:
                    lines.append(f"    ❌ {fail}")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ delete_file 失败（{elapsed_ms} ms）：{type(e).__name__}: {e}"


# ============================================================
# Tool 4: RecognizeFileTool —— 识别文件：类型/大小/时间/预览/SHA256(可选）
# ============================================================

class RecognizeFileArgs(BaseModel):
    file_path: str = Field(
        ...,
        min_length=1,
        description="【必填】要识别的文件路径（支持路径别名）。示例：数据根/报告.docx；或绝对路径",
    )
    with_preview_lines: int = Field(
        20,
        ge=0,
        le=200,
        description="【选填】如果是文本类文件，返回前 N 行预览（默认20行，0=不预览，最大200）",
    )
    with_sha256: bool = Field(
        False,
        description="【选填】是否计算 SHA256 指纹（大文件会慢，默认 False；真要算文件哈希再传 True）",
    )
    with_image_info: bool = Field(
        True,
        description="【选填】图片文件识别尺寸，若 Pillow 可用就给出 宽x高/格式（默认True，Pillow 未安装自动跳过不报错）",
    )


# —— 扩展名 → 中文大类名 映射（按优先级匹配，未命中=「其他」
_EXT_CATEGORY: dict[str, str] = {
    # 图片
    ".jpg": "图片", ".jpeg": "图片", ".png": "图片", ".gif": "图片", ".bmp": "图片",
    ".webp": "图片", ".svg": "图片", ".tiff": "图片", ".tif": "图片", ".heic": "图片",
    ".ico": "图片", ".raw": "图片",
    # 视频
    ".mp4": "视频", ".avi": "视频", ".mov": "视频", ".wmv": "视频", ".flv": "视频",
    ".mkv": "视频", ".webm": "视频", ".m4v": "视频", ".ts": "视频",
    # 音频
    ".mp3": "音频", ".wav": "音频", ".flac": "音频", ".aac": "音频", ".ogg": "音频",
    ".m4a": "音频", ".wma": "音频", ".ape": "音频",
    # 文档
    ".pdf": "文档", ".doc": "文档-Word", ".docx": "文档-Word", ".xls": "文档-Excel",
    ".xlsx": "文档-Excel", ".ppt": "文档-PPT", ".pptx": "文档-PPT",
    ".txt": "文档-纯文本", ".md": "文档-Markdown", ".rtf": "文档-RTF",
    ".csv": "文档-CSV表格", ".json": "文档-JSON", ".xml": "文档-XML",
    ".yaml": "文档-YAML", ".yml": "文档-YAML", ".ini": "文档-配置",
    ".log": "文档-日志", ".html": "文档-网页", ".htm": "文档-网页",
    ".epub": "文档-电子书", ".mobi": "文档-电子书",
    # 代码
    ".py": "代码-Python", ".js": "代码-JavaScript", ".ts": "代码-TypeScript",
    ".jsx": "代码-React", ".vue": "代码-Vue", ".java": "代码-Java",
    ".c": "代码-C", ".cpp": "代码-CPP", ".h": "代码-C头文件",
    ".cs": "代码-CSharp", ".go": "代码-Go", ".rs": "代码-Rust",
    ".rb": "代码-Ruby", ".php": "代码-PHP", ".swift": "代码-Swift",
    ".kt": "代码-Kotlin", ".r": "代码-R", ".sh": "代码-Shell",
    ".bat": "代码-Batch", ".ps1": "代码-PowerShell", ".sql": "代码-SQL",
    # 压缩包
    ".zip": "压缩包", ".rar": "压缩包", ".7z": "压缩包", ".tar": "压缩包",
    ".gz": "压缩包", ".bz2": "压缩包", ".xz": "压缩包",
    # 可执行
    ".exe": "可执行-EXE", ".msi": "安装包-MSI", ".apk": "安装包-AndroidAPK",
    ".ipa": "安装包-iOSIPA", ".app": "应用-macOS应用", ".deb": "安装包-DEB",
    ".rpm": "安装包-RPM", ".dmg": "镜像-DMG",
}

# 文本/代码/文档（可预览前 N 行的扩展）
_TEXT_EXTS: set[str] = {
    ".txt", ".md", ".log", ".csv", ".json", ".xml", ".yaml", ".yml", ".ini",
    ".py", ".js", ".ts", ".jsx", ".vue", ".java", ".c", ".cpp", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".r",
    ".sh", ".bat", ".ps1", ".sql", ".rtf ", ".html", ".htm", ".css",
    ".env", ".gitignore", ".dockerfile", ".toml", ".cfg", ".conf",
}


class RecognizeFileTool(BaseTool):
    """识别文件信息：大类 / 大小 / 修改时间 / 类型 / MIME / 文本预览 / 可选哈希 / 可选图片尺寸。
    不会修改任何文件，纯读取。
    """

    name: ClassVar[str] = "recognize_file"
    description: ClassVar[str] = (
        "Tool Name: recognize_file\n"
        "用途：读取单个文件的「识别信息（不改任何内容）。\n"
        "典型场景：\n"
        "  - 用户问「这是什么文件？」「帮我看看这个文件大小/类型/内容开头几行」\n"
        "  - delete_file 前先识别一下是不是要删的文件到底是什么（避免误删）\n"
        "返回信息：路径 / 类别（图片/视频/音频/文档-*/*代码-*/压缩包/可执行/其他）\n"
        "  / 扩展名 / 大小 / 创建/修改时间 / 可选文本预览前 N 行\n"
        "  / 可选 SHA256 指纹 / 可选图片尺寸（Pillow 有就识别）。\n"
    )
    args_schema: type[BaseModel] = RecognizeFileArgs
    return_direct: ClassVar[bool] = False

    def _run(  # noqa: D401
        self,
        file_path: str,
        with_preview_lines: int = 20,
        with_sha256: bool = False,
        with_image_info: bool = True,
    ) -> str:
        t0 = time.perf_counter_ns()
        try:
            p = resolve_user_path(file_path)
            assert_safe_path(p, writable=False)
            if not p.exists():
                raise FileNotFoundError(f"文件不存在：{p}")
            if p.is_dir():
                raise IsADirectoryError(f"目标 {p} 是目录，不是文件（请用 search_files 看目录内容）")
            ext = p.suffix.lower()
            st = p.stat()
            size = st.st_size
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
            ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(getattr(st, "st_ctime", st.st_mtime)))
            category = _EXT_CATEGORY.get(ext, "其他")
            # MIME：简单版，不需要额外依赖
            import mimetypes as _mt
            mime, _ = _mt.guess_type(str(p))
            mime = mime or "unknown/unknown"
            lines: list[str] = []
            lines.append(f"🔍 recognize_file 成功：{p.name}\n"
                     f"  真实路径：{p}\n"
                     f"  文件大类：{category}\n"
                     f"  扩展名：{ext or '（无扩展名）'}\n"
                     f"  MIME 类型：{mime}\n"
                     f"  文件大小：{_size_fmt(size)} （{size} 字节）\n"
                     f"  创建时间：{ctime}\n"
                     f"  修改时间：{mtime}\n")

            # 图片尺寸
            img_dims: str | None = None
            if with_image_info and category == "图片":
                try:
                    from PIL import Image  # type: ignore
                    with Image.open(p) as img:  # noqa: F841
                        w, h = img.size
                        img_dims = f"{w} x {h}，格式 {img.format}，模式 {img.mode}"
                        lines.append(f"  图片信息：{img_dims}\n")
                except Exception:  # noqa: BLE001
                    pass

            # SHA256
            if with_sha256:
                import hashlib as _hl
                h = _hl.sha256()
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                fp = h.hexdigest()
                lines.append(f"  SHA256 指纹：{fp[:32]}…{fp[-16:]}（共 64 位）\n")

            # 文本预览
            if with_preview_lines > 0 and (ext in _TEXT_EXTS or mime.startswith("text/") or size <= 50 * 1024):
                try:
                    preview_max = min(size, 200 * 1024)  # 最大读 200KB
                    # 统一用 UTF-8，失败就 GBK，再失败就不预览
                    content = None
                    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
                        try:
                            with p.open("r", encoding=enc, errors="strict") as f:
                                content = f.read(preview_max)
                            break
                        except UnicodeDecodeError:
                            continue
                    if content is None:
                        lines.append("  文本预览：编码无法识别为文本，跳过预览。")
                    else:
                        all_lines = content.splitlines()
                        show_lines = all_lines[:with_preview_lines]
                        lines.append(
                            f"  文本预览（前 {min(len(show_lines), with_preview_lines)} 行 / "
                            f"共 {len(all_lines)} 行，只读了前 {preview_max} 字节：\n"
                        )
                        for i, line in enumerate(show_lines, start=1):
                            esc = line if len(line) <= 200 else line[:200] + "…（行过长截断"
                            lines.append(f"    {i:>3}| {esc}")
                        if len(all_lines) > with_preview_lines:
                            lines.append(f"    …… 余下 {len(all_lines) - with_preview_lines} 行未预览")
                except Exception as e:  # noqa: BLE001
                    lines.append(f"  文本预览失败：{type(e).__name__}: {e}")
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            lines.append(f"\n  识别耗时：{elapsed_ms} ms")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ recognize_file 失败（{elapsed_ms} ms）：{type(e).__name__}: {e}"
