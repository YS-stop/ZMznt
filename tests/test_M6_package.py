"""M6 阶段验收单测（含 4 个子用例文件：T7 path_utils 打包适配、T8-T11 system_tools、T12 spec 合法性、T13 全局工具注册数）。

运行：
    $env:QT_QPA_PLATFORM="offscreen"
    .\venv_assistant\Scripts\python.exe -m pytest tests/test_M6_package.py -v
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# T7 path_utils 打包适配层（开发态）
# ---------------------------------------------------------------------------
class TestM6_PathUtilsFrozenCompat:
    def test_t7_01_is_frozen_dev_false(self) -> None:
        """开发环境（非打包）is_frozen() 返回 False。"""
        from src.utils.path_utils import is_frozen
        assert is_frozen() is False

    def test_t7_02_runtime_resources_dev_equals_project_root(self) -> None:
        """开发态 runtime_resources_dir() = PROJECT_ROOT。"""
        from src.utils.path_utils import runtime_resources_dir, PROJECT_ROOT
        assert runtime_resources_dir().resolve() == PROJECT_ROOT.resolve()

    def test_t7_03_resolve_resource_exists(self) -> None:
        """resolve_resource(".env") 或 resolve_resource("src") 至少一个存在。"""
        from src.utils.path_utils import resolve_resource, PROJECT_ROOT
        p = resolve_resource("src")
        assert p.exists(), f"resolve_resource('src') 应存在于开发态 {PROJECT_ROOT}"

    def test_t7_04_user_paths_outside_meipass(self) -> None:
        """用户数据路径必须是 PROJECT_ROOT 的子目录（非 sys._MEIPASS）。"""
        from src.utils.path_utils import DATA_ROOT, APP_DB_PATH, ENV_FILE_PATH, PROJECT_ROOT
        # 用户数据必须在 PROJECT_ROOT 下（或绝对路径自己）
        def _is_under(child: Path, parent: Path) -> bool:
            try:
                child.resolve().relative_to(parent.resolve())
                return True
            except ValueError:
                # 相对路径不在 parent 下的兜底：判断是不是自己
                return child.resolve() == parent.resolve()
        # 开发态所有用户可写路径必须在 PROJECT_ROOT 之下
        assert _is_under(DATA_ROOT, PROJECT_ROOT) or str(DATA_ROOT).startswith(
            str(PROJECT_ROOT)
        ), f"DATA_ROOT={DATA_ROOT} 不在 PROJECT_ROOT={PROJECT_ROOT} 下"
        assert _is_under(APP_DB_PATH.parent, PROJECT_ROOT) or str(APP_DB_PATH).startswith(
            str(PROJECT_ROOT)
        )
        # ENV_FILE_PATH 必须是 PROJECT_ROOT 下
        assert ENV_FILE_PATH.parent.resolve() == PROJECT_ROOT.resolve()

    def test_t7_05_ensure_data_dirs_idempotent(self) -> None:
        """ensure_data_dirs() 连续两次调用都不抛错（幂等）。"""
        from src.utils.path_utils import (
            ensure_data_dirs,
            DATA_ROOT,
            CHECKPOINTS_DIR,
            OCR_TEMP_DIR,
        )
        ensure_data_dirs()
        ensure_data_dirs()  # 第二次也不能报错
        for d in (DATA_ROOT, CHECKPOINTS_DIR, OCR_TEMP_DIR):
            assert d.exists() and d.is_dir(), f"{d} 不存在"


# ---------------------------------------------------------------------------
# T8 spec 合法性：Assistant.spec 是合法 Python 语法 + 含最小关键字
# ---------------------------------------------------------------------------
class TestM6_SpecFile:
    def test_t8_01_spec_exists_and_valid_python_syntax(self) -> None:
        p = _PROJECT_ROOT / "Assistant.spec"
        assert p.exists(), "Assistant.spec 必须在项目根目录"
        src = p.read_text(encoding="utf-8")
        # spec 文件本身是 Python 语法（虽然 PyInstaller 加了一些全局变量）
        ast.parse(src, filename=str(p))  # 语法错误会直接抛

    def test_t8_02_spec_contains_minimal_keywords(self) -> None:
        p = _PROJECT_ROOT / "Assistant.spec"
        src = p.read_text(encoding="utf-8")
        for kw in ("Analysis", "EXE", "DEVELOP_MODE", "ONE_FILE",
                   "hiddenimports", "_hiddenimports"):
            assert kw in src, f"Assistant.spec 必须包含关键字 {kw}"

    def test_t8_03_switches_default_values_safe(self) -> None:
        """默认 DEVELOP_MODE=False + ONE_FILE=False，避免用户第一次打包就 onefile+noconsole 难调试。"""
        p = _PROJECT_ROOT / "Assistant.spec"
        src = p.read_text(encoding="utf-8")
        # 简单字符串判定，不用 exec（避免执行 spec 全局变量需要 PyInstaller 运行时）
        assert "DEVELOP_MODE = False" in src
        assert "ONE_FILE     = False" in src or "ONE_FILE = False" in src


# ---------------------------------------------------------------------------
# T9 全局工具注册数 8 → 12（M6 新增 4 个 system_tools）
# ---------------------------------------------------------------------------
class TestM6_ToolRegistry:
    def test_t9_01_total_tools_12(self) -> None:
        from src.tools import get_all_tools, AVAILABLE_TOOL_NAMES
        tools = get_all_tools()
        names = AVAILABLE_TOOL_NAMES()
        assert len(tools) == 12, f"全局工具应为 12 个（旧 8 + M6 新 4），实际 {len(tools)}: {names}"
        assert len(names) == 12

    def test_t9_02_new_four_system_tools_present(self) -> None:
        from src.tools import AVAILABLE_TOOL_NAMES
        names = set(AVAILABLE_TOOL_NAMES())
        expected = {
            "system_volume", "system_screenshot", "system_power", "system_translate",
        }
        missing = expected - names
        assert not missing, f"缺少 M6 系统工具: {missing}"

    def test_t9_03_each_tool_has_callable_run(self) -> None:
        from src.tools import get_all_tools
        for t in get_all_tools():
            assert hasattr(t, "_run") and callable(getattr(t, "_run")), f"{t.name} 缺少 _run"


# ---------------------------------------------------------------------------
# T10 VolumeControlTool 基础用例（不实际改系统音量）
# ---------------------------------------------------------------------------
class TestM6_SystemVolume:
    def test_t10_01_mode_get_returns_readable(self) -> None:
        """get 模式返回固定格式字符串（即使 pycaw 不可用也返回降级提示）。"""
        from src.tools.system_tools import VolumeControlTool
        t = VolumeControlTool()
        out = t._run(mode="get")
        assert isinstance(out, str) and len(out) > 0

    def test_t10_02_mode_set_missing_volume_rejected(self) -> None:
        """mode=set 但 volume=None 直接拒绝不抛。"""
        from src.tools.system_tools import VolumeControlTool
        t = VolumeControlTool()
        out = t._run(mode="set", volume=None)
        assert "必须同时传" in out or "❌" in out

    def test_t10_03_unknown_mode_rejected(self) -> None:
        from src.tools.system_tools import VolumeControlTool
        t = VolumeControlTool()
        out = t._run(mode="this_is_bogus")
        assert "未知 mode" in out or "❌" in out


# ---------------------------------------------------------------------------
# T11 ScreenShotTool 基础用例（Qt offscreen，会走到 mss 兜底或 Qt 返回友好降级）
# ---------------------------------------------------------------------------
class TestM6_SystemScreenshot:
    def test_t11_01_default_save_returns_string(self) -> None:
        """截图返回字符串（成功路径或降级提示，不能抛异常）。"""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from src.tools.system_tools import ScreenShotTool
        t = ScreenShotTool()
        out = t._run(display_index=0)
        assert isinstance(out, str) and len(out) > 0
        # 要么成功包含"保存到"要么失败包含"失败"要么 mss 未装提示
        assert any(k in out for k in ("保存到", "成功", "失败", "兜底", "mss"))


# ---------------------------------------------------------------------------
# T12 SystemPowerTool 只测安全子集（lock/sleep 不实际调）
# ---------------------------------------------------------------------------
class TestM6_SystemPower:
    def test_t12_01_unknown_action_rejected(self) -> None:
        from src.tools.system_tools import SystemPowerTool
        t = SystemPowerTool()
        out = t._run(action="not_exist_action")
        assert "未知 action" in out or "❌" in out

    def test_t12_02_lock_mode_returns_string(self) -> None:
        """lock 模式即使不真的锁也会返回字符串（成功/失败原因），不会抛。"""
        from src.tools.system_tools import SystemPowerTool
        t = SystemPowerTool()
        # 真的调会锁屏幕！！这里不做，仅看类型
        # 实际我们只断言 "action='abort' 不抛"
        out_abort = t._run(action="abort")
        assert isinstance(out_abort, str) and len(out_abort) > 0
        assert "shutdown /a" in out_abort or "取消" in out_abort or "失败" in out_abort


# ---------------------------------------------------------------------------
# T13 TranslateTextTool 基础用例
# ---------------------------------------------------------------------------
class TestM6_SystemTranslate:
    def test_t13_01_empty_text_rejected(self) -> None:
        from src.tools.system_tools import TranslateTextTool
        t = TranslateTextTool()
        out = t._run(text="")
        assert "不能为空" in out or "❌" in out

    def test_t13_02_normal_text_returns_string(self) -> None:
        from src.tools.system_tools import TranslateTextTool
        t = TranslateTextTool()
        # 即使没有 QWEN Key，也会返回占位提示不抛
        out = t._run(text="你好世界", from_lang="zh", to_lang="en")
        assert isinstance(out, str) and len(out) > 0
        assert any(k in out for k in ("翻译结果", "QWEN_API_KEY 未配置", "调用 LLM 失败", "源语言识别"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
