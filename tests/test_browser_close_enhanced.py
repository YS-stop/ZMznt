"""浏览器关闭能力增强验证：

1. 工具注册：restore_browser_tab 已注册，工具总数 17
2. 纯逻辑：关键词扩展 / 域名提取 / 窗口标题剥离浏览器后缀
3. CDP 批量逻辑（mock）：按关键词批量关、重复站点只留一个、关闭其他标签
4. 模式路由：target=「全部」自动转 all；mode=others/duplicates 无 CDP 时优雅降级
5. 真机冒烟：_find_chromium_exe 能找到 Chrome/Edge；open_browser 带调试端口打开百度（可选，设 RUN_LIVE=1）
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.tools import AVAILABLE_TOOL_NAMES  # noqa: E402

names = AVAILABLE_TOOL_NAMES()
print("RESULT tool count:", len(names))
assert "restore_browser_tab" in names, "restore_browser_tab 未注册"
assert "close_browser_tab" in names
assert "open_browser" in names
print("RESULT registered restore_browser_tab: True")

import src.tools.browser_tools as bt  # noqa: E402

# ---------- 1. 关键词扩展 ----------
kws = bt._match_keywords_for("抖音")
print("RESULT keywords 抖音:", kws)
assert "抖音" in kws and "douyin" in kws

kws2 = bt._match_keywords_for("example.com")
assert kws2 == ["example.com"]
print("RESULT keywords plain-domain:", kws2)

# ---------- 2. 域名提取 / 标题剥离 ----------
assert bt._domain_of("https://www.taobao.com/item/123.htm") == "taobao.com"
assert bt._domain_of("https://item.taobao.com/a.htm") == "item.taobao.com"
assert bt._domain_of("about:blank") == ""
print("RESULT _domain_of: OK")

assert bt._strip_browser_suffix("百度一下 - Google Chrome") == "百度一下"
assert bt._strip_browser_suffix("知乎 - Microsoft Edge") == "知乎"
assert bt._strip_browser_suffix("普通标题") == "普通标题"
print("RESULT _strip_browser_suffix: OK")

# ---------- 3. CDP 批量逻辑（mock 标签列表 + 关闭动作） ----------
FAKE_TABS = [
    {"type": "page", "id": "t1", "title": "淘宝网 - 首页", "url": "https://www.taobao.com/"},
    {"type": "page", "id": "t2", "title": "淘宝 - 购物车", "url": "https://cart.taobao.com/cart.htm"},
    {"type": "page", "id": "t3", "title": "知乎 - 有问题就会有答案", "url": "https://www.zhihu.com/"},
    {"type": "page", "id": "t4", "title": "知乎 - 热榜", "url": "https://www.zhihu.com/hot"},
    {"type": "page", "id": "t5", "title": "新标签页", "url": "chrome://newtab/"},
    {"type": "service_worker", "id": "sw1", "title": "sw", "url": "https://www.taobao.com/sw.js"},
]

closed_ids: list[str] = []
orig_page_tabs = bt._cdp_page_tabs
orig_close_by_id = bt._cdp_close_tab_by_id
bt._cdp_page_tabs = lambda: [dict(t) for t in FAKE_TABS if t["type"] == "page"]  # type: ignore[assignment]
bt._cdp_close_tab_by_id = lambda tid: closed_ids.append(tid) or True  # type: ignore[assignment]
try:
    # 3a. site 模式批量关闭：「淘宝」应关掉 t1/t2，剩 3 个 page
    res = bt._cdp_close_tabs(bt._match_keywords_for("淘宝"))
    assert res is not None
    closed, remaining = res
    print("RESULT cdp site close:", closed, "| remaining:", remaining)
    assert closed_ids == ["t1", "t2"] and remaining == 3

    # 3b. duplicates：按站点主域去重（cart.taobao.com 与 taobao.com 视为同站点）
    #     知乎两个只留一个，taobao 两个只留一个 → 关 t2/t4，剩 3
    assert bt._site_key_of("https://cart.taobao.com/cart.htm") == "taobao.com"
    assert bt._site_key_of("https://docs.qq.com/a") == "qq.com"
    assert bt._site_key_of("https://www.example.com.cn/x") == "example.com.cn"
    closed_ids.clear()
    res = bt._cdp_close_duplicates()
    assert res is not None
    closed, remaining = res
    print("RESULT cdp duplicates close:", closed, "| remaining:", remaining)
    assert set(closed_ids) == {"t2", "t4"} and remaining == 3

    # 3c. others：mock 前台窗口标题为「知乎 - 热榜 - Google Chrome」→ 只保留 t4
    closed_ids.clear()
    import ctypes

    orig_fg = ctypes.windll.user32.GetForegroundWindow  # noqa: SLF001
    orig_len = ctypes.windll.user32.GetWindowTextLengthW  # noqa: SLF001
    orig_txt = ctypes.windll.user32.GetWindowTextW  # noqa: SLF001

    _fg_title = "知乎 - 热榜 - Google Chrome"
    ctypes.windll.user32.GetForegroundWindow = lambda: 12345  # noqa: SLF001
    ctypes.windll.user32.GetWindowTextLengthW = lambda h: len(_fg_title)  # noqa: SLF001
    ctypes.windll.user32.GetWindowTextW = lambda h, b, n: setattr(b, "value", _fg_title) or 0  # noqa: SLF001
    try:
        res = bt._cdp_close_others()
        assert res is not None
        closed, remaining, kept = res
        print("RESULT cdp others close:", closed, "| remaining:", remaining, "| kept:", kept)
        assert kept == "知乎 - 热榜"
        assert closed_ids == ["t1", "t2", "t3", "t5"] and remaining == 1
    finally:
        ctypes.windll.user32.GetForegroundWindow = orig_fg  # noqa: SLF001
        ctypes.windll.user32.GetWindowTextLengthW = orig_len  # noqa: SLF001
        ctypes.windll.user32.GetWindowTextW = orig_txt  # noqa: SLF001
finally:
    bt._cdp_page_tabs = orig_page_tabs  # type: ignore[assignment]
    bt._cdp_close_tab_by_id = orig_close_by_id  # type: ignore[assignment]

# ---------- 4. 模式路由 / 降级 ----------
tool = bt.CloseBrowserTabTool()

# 4a. target=「全部」→ 自动转 all（真机执行：只验证不抛异常且返回结构化文案）
r = tool._run(target="全部")
print("RESULT route 全部→all:", r.splitlines()[0])
assert isinstance(r, str) and r

# 4b. mode=others 无 CDP 时（若无调试端口浏览器在跑）应优雅提示
r2 = tool._run(target="", mode="others")
print("RESULT mode=others:", r2.splitlines()[0])
assert isinstance(r2, str) and r2

# 4c. mode=duplicates 同理
r3 = tool._run(target="", mode="duplicates")
print("RESULT mode=duplicates:", r3.splitlines()[0])
assert isinstance(r3, str) and r3

# 4d. mode=site 空 target 报错
r4 = tool._run(target="", mode="site")
assert "target 不能为空" in r4
print("RESULT site empty target guard: OK")

# ---------- 5. 真机探测 ----------
found = bt._find_chromium_exe()
print("RESULT chromium exe:", found)

if os.environ.get("RUN_LIVE") == "1":
    if not found:
        print("SKIP live open: 未找到 Chrome/Edge")
    else:
        opener = bt.OpenBrowserTool()
        out = opener._run("百度")
        print("RESULT live open_browser:", out.splitlines()[0])
        import time

        time.sleep(3)
        print("RESULT cdp ready after open:", bt._cdp_ready())
        # 恢复工具冒烟（恢复一次，无害）
        rest = bt.RestoreClosedTabTool()
        out2 = rest._run(1)
        print("RESULT restore_browser_tab:", out2.splitlines()[0])

print("ALL PASSED")
