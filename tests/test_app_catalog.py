"""应用目录服务冒烟测试：构建目录 + 模糊匹配（只读操作，不启动任何应用）。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.app_catalog_service import get_app_catalog  # noqa: E402

cat = get_app_catalog()
n = cat.build(refresh=True)
print(f"RESULT catalog size: {n}")
assert n > 0, "一个应用都没扫到，异常"

names = cat.all_names()
print("RESULT sample names:", names[:15])

# 模糊匹配演示（用目录里真实存在的名字测）
probe = names[0]
entry, cands = cat.find_app(probe)
assert entry is not None and entry["name"] == probe
print(f"RESULT exact match: {probe} -> path={entry['path'] or '(lnk only)'} lnk={entry['lnk']}")

# 部分匹配
if len(probe) >= 2:
    entry2, cands2 = cat.find_app(probe[:2])
    print(f"RESULT fuzzy '{probe[:2]}' -> best={entry2['name'] if entry2 else None} cands={cands2[:5]}")

print("RESULT catalog: ALL PASS")
