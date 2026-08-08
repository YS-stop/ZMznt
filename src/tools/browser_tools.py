"""打开/关闭浏览器工具（LangChain BaseTool 子类）。

功能：
    * OpenBrowserTool：打开系统默认浏览器，访问指定 URL / 快捷站点 / 或执行「百度搜索关键词」。
    * CloseBrowserTabTool：关闭指定站点/关键词对应的浏览器标签页（三级策略：CDP 精确关 →
      窗口标题匹配 + 模拟 Ctrl+W → 仅报告找不到），target=「全部」时优雅关闭整个浏览器（WM_CLOSE）。

约定：
    1. 未知「非网址字符串」默认走百度搜索（URL = https://www.baidu.com/s?wd=xxx），让 LLM 说「帮我搜 xxx」也能一步到位。
    2. 快捷站点映射不用 LLM 记完整域名，传中文名称即可。
    3. 所有异常包装为 observation 返回，不向上抛。
    4. 关闭操作只针对浏览器窗口/标签，绝不 taskkill 杀进程（避免丢未保存数据）。
"""
from __future__ import annotations

import re
import sys
import time
import webbrowser
from pathlib import Path
from typing import ClassVar
from urllib.parse import quote_plus

from pydantic import BaseModel, Field

# 确保 import src.* 成功
_SRC_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from langchain_core.tools import BaseTool  # noqa: E402


# ============================================================
# 快捷站点映射：口语化名称 → 真实 URL
# ============================================================
_SITE_SHORTCUTS: dict[str, str] = {
    # —— 搜索引擎 ——
    "百度": "https://www.baidu.com",
    "baidu": "https://www.baidu.com",
    "必应": "https://cn.bing.com",
    "bing": "https://cn.bing.com",
    "搜狗": "https://www.sogou.com",
    "sogou": "https://www.sogou.com",
    "360搜索": "https://www.so.com",
    "so": "https://www.so.com",
    "谷歌": "https://www.google.com",
    "google": "https://www.google.com",
    "duckduckgo": "https://duckduckgo.com",
    # —— 社区 / 内容 ——
    "知乎": "https://www.zhihu.com",
    "zhihu": "https://www.zhihu.com",
    "B站": "https://www.bilibili.com",
    "bilibili": "https://www.bilibili.com",
    "哔哩哔哩": "https://www.bilibili.com",
    "微博": "https://weibo.com",
    "weibo": "https://weibo.com",
    "豆瓣": "https://www.douban.com",
    "douban": "https://www.douban.com",
    "小红书": "https://www.xiaohongshu.com",
    "xhs": "https://www.xiaohongshu.com",
    "抖音": "https://www.douyin.com",
    "douyin": "https://www.douyin.com",
    "快手": "https://www.kuaishou.com",
    "kuaishou": "https://www.kuaishou.com",
    # —— 电商 / 生活 ——
    "淘宝": "https://www.taobao.com",
    "taobao": "https://www.taobao.com",
    "天猫": "https://www.tmall.com",
    "tmall": "https://www.tmall.com",
    "京东": "https://www.jd.com",
    "jd": "https://www.jd.com",
    "拼多多": "https://www.pinduoduo.com",
    "pdd": "https://www.pinduoduo.com",
    "美团": "https://www.meituan.com",
    "meituan": "https://www.meituan.com",
    "饿了么": "https://www.ele.me",
    "eleme": "https://www.ele.me",
    # —— 邮箱 / 办公 ——
    "QQ邮箱": "https://mail.qq.com",
    "qqmail": "https://mail.qq.com",
    "163邮箱": "https://mail.163.com",
    "163mail": "https://mail.163.com",
    "126邮箱": "https://mail.126.com",
    "Gmail": "https://mail.google.com",
    "gmail": "https://mail.google.com",
    "Outlook": "https://outlook.live.com",
    "outlook": "https://outlook.live.com",
    "飞书": "https://www.feishu.cn",
    "feishu": "https://www.feishu.cn",
    "钉钉": "https://www.dingtalk.com",
    "dingtalk": "https://www.dingtalk.com",
    "企业微信": "https://work.weixin.qq.com",
    "wework": "https://work.weixin.qq.com",
    "语雀": "https://www.yuque.com",
    "yuque": "https://www.yuque.com",
    "印象笔记": "https://app.yinxiang.com",
    "yinxiang": "https://app.yinxiang.com",
    "有道云笔记": "https://note.youdao.com",
    "youdao_note": "https://note.youdao.com",
    "Notion": "https://www.notion.so",
    "notion": "https://www.notion.so",
    "石墨文档": "https://shimo.im",
    "shimo": "https://shimo.im",
    "腾讯文档": "https://docs.qq.com",
    "docs_qq": "https://docs.qq.com",
    "金山文档": "https://kdocs.cn",
    "kdocs": "https://kdocs.cn",
    # —— 开发 ——
    "GitHub": "https://github.com",
    "github": "https://github.com",
    "Gitee": "https://gitee.com",
    "gitee": "https://gitee.com",
    "GitLab": "https://gitlab.com",
    "gitlab": "https://gitlab.com",
    "掘金": "https://juejin.cn",
    "juejin": "https://juejin.cn",
    "博客园": "https://www.cnblogs.com",
    "cnblogs": "https://www.cnblogs.com",
    "思否": "https://segmentfault.com",
    "segmentfault": "https://segmentfault.com",
    "知乎专栏": "https://zhuanlan.zhihu.com",
    "CSDN": "https://www.csdn.net",
    "csdn": "https://www.csdn.net",
    "Docker Hub": "https://hub.docker.com",
    "dockerhub": "https://hub.docker.com",
    "PyPI": "https://pypi.org",
    "pypi": "https://pypi.org",
    "npm": "https://www.npmjs.com",
    "MDN": "https://developer.mozilla.org",
    "mdn": "https://developer.mozilla.org",
    "Stack Overflow": "https://stackoverflow.com",
    "stackoverflow": "https://stackoverflow.com",
    "LangChain 文档": "https://python.langchain.com",
    "langchain": "https://python.langchain.com",
    "LangGraph 文档": "https://langchain-ai.github.io/langgraph/",
    "langgraph": "https://langchain-ai.github.io/langgraph/",
    # —— 视频 / 娱乐 ——
    "爱奇艺": "https://www.iqiyi.com",
    "iqiyi": "https://www.iqiyi.com",
    "腾讯视频": "https://v.qq.com",
    "v_qq": "https://v.qq.com",
    "优酷": "https://www.youku.com",
    "youku": "https://www.youku.com",
    "芒果TV": "https://www.mgtv.com",
    "mgtv": "https://www.mgtv.com",
    "西瓜视频": "https://www.ixigua.com",
    "ixigua": "https://www.ixigua.com",
    "网易云音乐": "https://music.163.com",
    "netease_music": "https://music.163.com",
    "QQ音乐": "https://y.qq.com",
    "qq_music": "https://y.qq.com",
    # —— 盘 / 下载 ——
    "阿里云盘": "https://www.alipan.com",
    "alipan": "https://www.alipan.com",
    "百度网盘": "https://pan.baidu.com",
    "baidupan": "https://pan.baidu.com",
    "夸克网盘": "https://pan.quark.cn",
    "quarkpan": "https://pan.quark.cn",
    "迅雷云盘": "https://pan.xunlei.com",
    "xunleipan": "https://pan.xunlei.com",
}

# 常见域名后缀：用于快速判断一个字符串是「域名/URL」还是「纯关键词」
_DOMAIN_SUFFIXES = (
    ".com", ".cn", ".net", ".org", ".io", ".co", ".me", ".dev", ".app",
    ".top", ".xyz", ".tech", ".store", ".vip", ".cc", ".tv", ".info",
    ".edu", ".gov", ".ac.cn", ".com.cn", ".net.cn", ".org.cn",
)


def _looks_like_url(s: str) -> bool:
    """粗判是否是 URL / 域名（不是 100% 准确，但够用）。"""
    s = s.strip().lower()
    if not s:
        return False
    if s.startswith(("http://", "https://", "ftp://", "file://")):
        return True
    # 有没有包含 / 例如 bilibili.com/v/xxx
    if "/" in s and "." in s.split("/", 1)[0]:
        head = s.split("/", 1)[0]
        if any(head.endswith(suf) for suf in _DOMAIN_SUFFIXES):
            return True
    if ":" in s:  # 带端口号 localhost:8080
        return True
    return any(s.endswith(suf) for suf in _DOMAIN_SUFFIXES)


def resolve_target_to_url(raw_target: str) -> tuple[str, str]:
    """把用户输入的 target 解析为 (最终URL, 解析方式说明)。
    解析顺序：
        1. 完全匹配 快捷站点名称
        2. 看起来是 URL / 域名 → 如果无 http 前缀就补 https://
        3. 其他 → 当作关键词，走百度搜索 https://www.baidu.com/s?wd=xxx
    """
    t = (raw_target or "").strip()
    if not t:
        raise ValueError("target 不能为空，请提供要打开的网站、快捷名称或搜索关键词。")

    # 1. 快捷站点
    if t in _SITE_SHORTCUTS:
        return _SITE_SHORTCUTS[t], f"快捷站点「{t}」"

    # 2. 域名 / URL
    if _looks_like_url(t):
        url = t if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", t) else "https://" + t
        return url, f"自动补协议的 URL「{url}」"

    # 3. 纯关键词 → 百度搜索
    encoded = quote_plus(t)
    url = f"https://www.baidu.com/s?wd={encoded}"
    return url, f"百度搜索关键词「{t}」"


# ============================================================
# OpenBrowserTool
# ============================================================

class OpenBrowserArgs(BaseModel):
    target: str = Field(
        ...,
        min_length=1,
        description=(
            "【必填】浏览器目标，支持 3 种写法任意一种：\n"
            "  1. 快捷站点名（中文/英文都可）：例如「知乎」「GitHub」「B站」「百度」「淘宝」「Gmail」\n"
            "  2. 完整或部分 URL/域名：例如 https://www.zhihu.com 或 zhihu.com 或 127.0.0.1:8000\n"
            "  3. 普通关键词（不是站点也不是网址）：会自动打开「百度搜该关键词」的页面，例如「2026年奥运会赛程」"
        ),
    )
    new_tab: bool = Field(
        True,
        description="【选填】是否在新标签页打开（默认 True），False 表示复用当前浏览器窗口。",
    )
    autoraise: bool = Field(
        True,
        description="【选填】是否自动把浏览器窗口切到最前面（默认 True）。",
    )


class OpenBrowserTool(BaseTool):
    """打开系统默认浏览器并访问指定 URL。LLM 传中文快捷站点名即可，不需要记域名。
    若传普通关键词（非网址/非快捷站点）会自动跳转到百度搜索结果页，实现「搜 xxx」一步到位。
    """

    name: ClassVar[str] = "open_browser"
    description: ClassVar[str] = (
        "Tool Name: open_browser\n"
        "用途：打开系统默认浏览器，访问网站 / 打开百度搜索结果页。\n"
        "调用示例对应的 target 写法：\n"
        "  「打开知乎」→ target=知乎\n"
        "  「打开 GitHub 看我的仓库」→ target=GitHub\n"
        "  「打开 https://example.com/a?x=1」→ target=https://example.com/a?x=1\n"
        "  「打开 example.com」→ target=example.com（自动补 https://）\n"
        "  「帮我百度搜一下 2026 年奥运会赛程」→ target=2026 年奥运会赛程（自动跳百度搜索）\n"
        "常见快捷站点：百度、必应、知乎、B站、微博、豆瓣、小红书、抖音、淘宝、天猫、京东、QQ邮箱、163邮箱、"
        "Gmail、飞书、钉钉、企业微信、语雀、印象笔记、Notion、石墨文档、腾讯文档、GitHub、Gitee、掘金、CSDN、"
        "爱奇艺、腾讯视频、优酷、网易云音乐、QQ音乐、阿里云盘、百度网盘 ……（完整列表共 100+ 个）\n"
        "注意：不会读取浏览器页面内容，只负责『打开』。要获取网页资讯请使用后续的 search_news 工具。"
    )
    args_schema: type[BaseModel] = OpenBrowserArgs
    return_direct: ClassVar[bool] = False

    def _run(self, target: str, new_tab: bool = True, autoraise: bool = True) -> str:  # noqa: D401
        t0 = time.perf_counter_ns()
        try:
            url, method = resolve_target_to_url(target)
            # webbrowser.open 参数：
            #   new=2 → 新标签页；new=1 → 新窗口；new=0 → 当前
            new_val = 2 if new_tab else 0
            ok = webbrowser.open(url, new=new_val, autoraise=autoraise)
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            if ok:
                return (
                    f"✅ open_browser 成功\n"
                    f"  解析方式：{method}\n"
                    f"  最终 URL：{url}\n"
                    f"  新标签页：{new_tab} | 自动置顶：{autoraise}\n"
                    f"  耗时：{elapsed_ms} ms"
                )
            return (
                f"⚠️ open_browser 已提交但系统未确认成功（耗时 {elapsed_ms} ms）\n"
                f"  解析方式：{method}\n"
                f"  请手动检查浏览器是否打开：{url}"
            )
        except Exception as e:  # noqa: BLE001
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ open_browser 失败（{elapsed_ms} ms）：{type(e).__name__}: {e}"


# ============================================================
# CloseBrowserTabTool —— 关闭标签页 / 整个浏览器
# ============================================================

_CDP_PORTS = (9222, 9223)  # 常见 Chrome/Edge 远程调试端口
# 浏览器窗口标题里常见的浏览器标识（用于区分浏览器窗口和普通窗口）
_BROWSER_TITLE_MARKS = (
    "microsoft edge", "google chrome", "firefox", "brave", "opera",
    "360", "qq浏览器", "搜狗", "夸克", "浏览器", "browser",
)
_WM_CLOSE = 0x0010


def _match_keywords_for(target: str) -> list[str]:
    """把 target 扩展成匹配关键词列表：站点中文名 + 域名核心（如 douyin）。"""
    t = (target or "").strip()
    kws: list[str] = []
    if t:
        kws.append(t)
    url = _SITE_SHORTCUTS.get(t)
    if url:
        m = re.search(r"://(?:www\.)?([^./]+)", url)
        if m and m.group(1).lower() not in [k.lower() for k in kws]:
            kws.append(m.group(1))
    return kws


def _cdp_list_tabs() -> list[dict] | None:
    """尝试通过 CDP HTTP 端点列出标签页。浏览器没开调试端口时返回 None。"""
    try:
        import requests  # 项目已装
    except Exception:
        return None
    for port in _CDP_PORTS:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/json", timeout=1.5)
            if r.ok:
                return [t for t in r.json() if isinstance(t, dict)]
        except Exception:  # noqa: BLE001
            continue
    return None


def _cdp_close_tabs(keywords: list[str]) -> list[str] | None:
    """CDP 精确关闭匹配标签。返回已关闭标签标题列表；CDP 不可用返回 None。"""
    tabs = _cdp_list_tabs()
    if tabs is None:
        return None
    closed: list[str] = []
    import requests
    for tab in tabs:
        if tab.get("type") != "page":
            continue
        title = str(tab.get("title", ""))
        url = str(tab.get("url", ""))
        hay = f"{title} {url}".lower()
        if any(k.lower() in hay for k in keywords):
            try:
                # Chrome CDP 提供 HTTP 关闭端点，无需 websocket
                import urllib.parse
                tid = urllib.parse.quote(str(tab.get("id", "")), safe="")
                for port in _CDP_PORTS:
                    try:
                        requests.get(f"http://127.0.0.1:{port}/json/close/{tid}", timeout=1.5)
                        break
                    except Exception:  # noqa: BLE001
                        continue
                closed.append(title or url)
            except Exception:  # noqa: BLE001
                continue
    return closed


def _enum_visible_windows() -> list[tuple[int, str]]:
    """枚举所有可见顶层窗口：返回 [(hwnd, title)]。纯 ctypes，零依赖。"""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    results: list[tuple[int, str]] = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd: int, _lparam: int) -> bool:
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n > 0:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                results.append((hwnd, buf.value))
        return True

    user32.EnumWindows(WNDENUMPROC(_cb), 0)
    return results


def _is_browser_title(title: str) -> bool:
    low = title.lower()
    return any(mark in low for mark in _BROWSER_TITLE_MARKS)


def _foreground_and_ctrl_w(hwnd: int) -> bool:
    """把窗口置前台并模拟 Ctrl+W（关闭当前标签页）。返回是否按键已发出。"""
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    VK_MENU, VK_CONTROL, VK_W = 0x12, 0x11, 0x57
    KEYEVENTF_KEYUP = 0x0002
    SW_RESTORE = 9
    try:
        # 按一下 Alt 解除 Windows 的前台锁定限制（常见技巧）
        user32.keybd_event(VK_MENU, 0, 0, 0)
        user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.25)
        user32.keybd_event(VK_CONTROL, 0, 0, 0)
        user32.keybd_event(VK_W, 0, 0, 0)
        user32.keybd_event(VK_W, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def _close_window_gracefully(hwnd: int) -> bool:
    """给窗口发 WM_CLOSE（优雅关闭，等同点窗口右上角 ×）。"""
    import ctypes
    try:
        ctypes.windll.user32.PostMessageW(hwnd, _WM_CLOSE, 0, 0)  # type: ignore[attr-defined]
        return True
    except Exception:  # noqa: BLE001
        return False


class CloseBrowserTabArgs(BaseModel):
    target: str = Field(
        ...,
        min_length=1,
        description=(
            "【必填】要关闭的目标：\n"
            "  - 站点名/关键词：如「抖音」「知乎」「B站」，匹配标签页标题或网址后关闭对应标签页；\n"
            "  - 「全部」/「所有」/「浏览器」：关闭整个浏览器（等同点窗口右上角 ×，不是杀进程）。"
        ),
    )


class CloseBrowserTabTool(BaseTool):
    """关闭浏览器标签页 / 整个浏览器。

    三级策略：
        ① CDP 精确关（浏览器带 --remote-debugging-port 启动时，可按标题/URL 关任意后台标签）
        ② 枚举浏览器窗口标题匹配 → 置前台 → 模拟 Ctrl+W 关闭当前标签
        ③ 都没匹配到 → 列出当前打开的浏览器窗口标题，告诉用户没找到
    target=「全部」→ 给所有浏览器窗口发 WM_CLOSE 优雅关闭整个浏览器。
    """

    name: ClassVar[str] = "close_browser_tab"
    description: ClassVar[str] = (
        "Tool Name: close_browser_tab\n"
        "用途：关闭浏览器里指定的标签页，或关闭整个浏览器。\n"
        "典型场景：\n"
        "  - 「关闭抖音标签页」「把抖音关了」→ target=抖音\n"
        "  - 「关闭知乎那个页面」→ target=知乎\n"
        "  - 「关闭浏览器」「把浏览器都关了」→ target=全部\n"
        "匹配逻辑：站点中文名 + 域名（如 douyin）同时匹配标签页标题和网址。\n"
        "说明：\n"
        "  - 关闭整个浏览器用 WM_CLOSE（等同手动点 ×），不会杀进程，浏览器可正常恢复会话。\n"
        "  - 关单个标签时若目标标签不是窗口当前激活标签，且浏览器未开调试端口，可能关不掉——"
        "    此时会返回当前打开的窗口列表，请提示用户先切到该标签再试。\n"
        "  - 只处理浏览器窗口，不影响其他任何应用。"
    )
    args_schema: type[BaseModel] = CloseBrowserTabArgs
    return_direct: ClassVar[bool] = False

    def _run(self, target: str) -> str:  # noqa: D401
        t0 = time.perf_counter_ns()
        try:
            t = (target or "").strip()
            if not t:
                return "❌ close_browser_tab：target 不能为空"

            # ---------- 分支 1：关闭整个浏览器 ----------
            if t in ("全部", "所有", "所有浏览器", "浏览器", "全部关闭", "all", "*"):
                wins = [(h, ti) for h, ti in _enum_visible_windows() if _is_browser_title(ti)]
                if not wins:
                    return "⚠️ 没有找到任何正在运行的浏览器窗口（可能浏览器并未打开）。"
                n = 0
                for hwnd, _ti in wins:
                    if _close_window_gracefully(hwnd):
                        n += 1
                ms = (time.perf_counter_ns() - t0) // 1_000_000
                return (
                    f"✅ 已向 {n} 个浏览器窗口发送关闭请求（WM_CLOSE，等同手动点 ×，{ms} ms）\n"
                    f"  涉及窗口：{'；'.join(ti for _, ti in wins[:5])}"
                    + (" ……" if len(wins) > 5 else "")
                    + "\n  说明：浏览器可能提示「是否关闭所有标签页」，属正常确认流程。"
                )

            keywords = _match_keywords_for(t)

            # ---------- 分支 2：CDP 精确关闭 ----------
            cdp_res = _cdp_close_tabs(keywords)
            if cdp_res is not None:
                ms = (time.perf_counter_ns() - t0) // 1_000_000
                if cdp_res:
                    return (
                        f"✅ 已通过浏览器调试接口关闭 {len(cdp_res)} 个匹配「{t}」的标签页（{ms} ms）：\n"
                        + "\n".join(f"  ✂ {ti}" for ti in cdp_res)
                    )
                # CDP 可用但没匹配到 → 列出当前标签帮助定位
                tabs = _cdp_list_tabs() or []
                names = [str(x.get("title") or x.get("url")) for x in tabs if x.get("type") == "page"]
                return (
                    f"⚠️ 浏览器调试接口中没有找到匹配「{t}」的标签页（关键词：{'/'.join(keywords)}）。\n"
                    f"  当前打开的标签：{'；'.join(names[:10]) or '（无）'}"
                )

            # ---------- 分支 3：窗口标题匹配 + Ctrl+W ----------
            wins = _enum_visible_windows()
            matched = [
                (h, ti) for h, ti in wins
                if _is_browser_title(ti) and any(k in ti for k in keywords)
            ]
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            if not matched:
                browser_titles = [ti for _, ti in wins if _is_browser_title(ti)]
                return (
                    f"⚠️ 没有找到标题匹配「{t}」的浏览器窗口（{ms} ms）。\n"
                    f"  当前浏览器窗口：{'；'.join(browser_titles[:8]) or '（没有打开的浏览器窗口）'}\n"
                    f"  提示：若目标标签存在但不是激活标签，请先手动切到该标签；"
                    f"或以调试端口启动浏览器（--remote-debugging-port=9222）后可精确关闭任意标签。"
                )
            closed_titles: list[str] = []
            for hwnd, ti in matched:
                if _foreground_and_ctrl_w(hwnd):
                    closed_titles.append(ti)
                time.sleep(0.3)
            if closed_titles:
                return (
                    f"✅ 已关闭 {len(closed_titles)} 个匹配「{t}」的标签页（窗口匹配 + Ctrl+W，{ms} ms）：\n"
                    + "\n".join(f"  ✂ {ti}" for ti in closed_titles)
                    + "\n  说明：此方式关闭的是匹配窗口的当前激活标签页。"
                )
            return (
                f"⚠️ 找到了匹配「{t}」的窗口但按键模拟失败（可能被系统前台限制拦截）。\n"
                f"  请手动切换到该标签页后重试，或直接说「关闭浏览器」。"
            )
        except Exception as e:  # noqa: BLE001
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ close_browser_tab 失败（{ms} ms）：{type(e).__name__}: {e}"


__all__ = [
    "OpenBrowserTool",
    "CloseBrowserTabTool",
    "resolve_target_to_url",
    "_SITE_SHORTCUTS",
]
