"""M6-0 验证 4 个新依赖安装成功。"""
print("=== M6 新增依赖验证 ===")
ok = 0
miss = []

for m, mod_name in [
    ("PyInstaller", "PyInstaller"),
    ("pycaw", "pycaw"),
    ("comtypes", "comtypes"),
    ("Pillow (PIL)", "PIL"),
]:
    try:
        mod = __import__(mod_name)
        v = getattr(mod, "__version__", "(built-in)")
        print(f"  ✅ {m:20s}  {v}")
        ok += 1
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ {m:20s}  MISS: {type(e).__name__}: {e}")
        miss.append(m)

import sys, struct
print(f"\n  Python {sys.version}   架构: {struct.calcsize('P')*8} 位")
print(f"\n  结果: {ok} OK / {len(miss)} MISS")
if miss:
    sys.exit(1)
sys.exit(0)
