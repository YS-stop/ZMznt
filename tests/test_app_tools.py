"""桌面应用工具集验证：注册 / open_app 真开记事本 / 窗口监控 / 屏幕视觉识别。"""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

from src.tools import AVAILABLE_TOOL_NAMES  # noqa: E402

names = AVAILABLE_TOOL_NAMES()
print("RESULT tool count:", len(names))
for t in ("open_app", "list_active_apps", "recognize_screen"):
    print(f"RESULT registered {t}:", t in names)
    assert t in names

# Qt 初始化（截图需要）
from PySide6.QtWidgets import QApplication  # noqa: E402
app = QApplication([])

from src.tools.app_tools import OpenAppTool, ListActiveAppsTool, RecognizeScreenTool  # noqa: E402

# 1. list_active_apps（只读监控）
lat = ListActiveAppsTool()
r = lat._run("")
print("RESULT list_active_apps first lines:")
print("\n".join(r.splitlines()[:6]))

# 2. open_app 真开记事本（内置应用，无害）
oa = OpenAppTool()
r2 = oa._run("记事本")
print("RESULT open_app 记事本:", r2.splitlines()[0])
time.sleep(2.0)

# 3. 监控里应该能看到记事本
r3 = lat._run("记事本")
print("RESULT monitor found notepad:", "notepad" in r3.lower() or "记事本" in r3)

# 4. 关掉刚才开的记事本（WM_CLOSE 优雅关闭）
import ctypes  # noqa: E402
user32 = ctypes.windll.user32
found = []

WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
def _cb(hwnd, _):
    if user32.IsWindowVisible(hwnd):
        n = user32.GetWindowTextLengthW(hwnd)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if "记事本" in buf.value or "Notepad" in buf.value:
                found.append(hwnd)
    return True
user32.EnumWindows(WNDENUMPROC(_cb), 0)
for hwnd in found:
    user32.PostMessageW(hwnd, 0x0010, 0, 0)
print("RESULT notepad closed:", len(found), "window(s)")

# 5. recognize_screen（真实截图 + Qwen-VL）
rs = RecognizeScreenTool()
r5 = rs._run("屏幕上有哪些打开的窗口？一句话概括")
print("RESULT recognize_screen:")
print(r5[:400])
