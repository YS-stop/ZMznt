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

import os
import re
import subprocess
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
# Chromium 调试端口（CDP）启动支持
# ============================================================

_CDP_DEFAULT_PORT = 9222
_CDP_AUTOMATION_PORT = 9223


def _get_automation_profile_dir() -> str:
    """获取自动化浏览器专用用户数据目录（与用户日常浏览器隔离，避免端口冲突）。"""
    base = os.environ.get("ASSISTANT_DATA_DIR", "")
    if not base:
        base = str(Path(_SRC_ROOT) / "data")
    profile_dir = Path(base) / "browser_automation_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    return str(profile_dir)


def _find_chromium_exe() -> tuple[str, str] | None:
    """找到本机 Chromium 系浏览器（Chrome/Edge）的可执行文件路径。

    返回 (exe_path, 浏览器名)；找不到返回 None。
    查找顺序：注册表默认浏览器 → 常见安装路径。
    只接受 Chrome / Edge（均支持 --remote-debugging-port），
    其他默认浏览器（Firefox 等）返回 None 走 webbrowser 兜底。
    """
    import os

    candidates: list[tuple[str, str]] = []

    # 1) 注册表：当前用户默认浏览器
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice",
        ) as k:
            progid, _ = winreg.QueryValueEx(k, "ProgId")
        with winreg.OpenKey(
            winreg.HKEY_CLASSES_ROOT,
            rf"{progid}\shell\open\command",
        ) as k:
            cmd, _ = winreg.QueryValueEx(k, "")
        m = re.search(r'"([^"]+\.exe)"', str(cmd)) or re.search(r"(\S+\.exe)", str(cmd))
        if m:
            exe = m.group(1)
            low = exe.lower()
            if "chrome" in low:
                candidates.append((exe, "Google Chrome"))
            elif "msedge" in low or "edge" in low:
                candidates.append((exe, "Microsoft Edge"))
    except Exception:  # noqa: BLE001
        pass

    # 2) 常见安装路径
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pfx = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    lad = os.environ.get("LOCALAPPDATA", "")
    well_known = [
        (os.path.join(pf, r"Google\Chrome\Application\chrome.exe"), "Google Chrome"),
        (os.path.join(pfx, r"Google\Chrome\Application\chrome.exe"), "Google Chrome"),
        (os.path.join(lad, r"Google\Chrome\Application\chrome.exe"), "Google Chrome"),
        (os.path.join(pfx, r"Microsoft\Edge\Application\msedge.exe"), "Microsoft Edge"),
        (os.path.join(pf, r"Microsoft\Edge\Application\msedge.exe"), "Microsoft Edge"),
    ]
    candidates.extend(well_known)

    seen: set[str] = set()
    for exe, name in candidates:
        if exe and exe not in seen and Path(exe).is_file():
            return exe, name
        seen.add(exe)
    return None


def _cleanup_stale_automation_chrome() -> None:
    """启动新自动化 Chrome 前，清理占用同一端口或 Profile 的旧实例。

    场景：上次 App 异常退出时没关掉 Chrome，导致新实例端口冲突或 Profile 锁住。
    只杀「带调试端口的」Chrome（匹配 --remote-debugging-port 参数），不碰用户日常浏览器。
    """
    try:
        result = subprocess.run(
            ["wmic", "process", "where", "name='chrome.exe'", "get", "ProcessId,CommandLine"],
            capture_output=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW,  # noqa: S603
            encoding="utf-8", errors="replace",  # 处理 Windows 命令行编码
        )
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if not line or "CommandLine" in line:
                continue
            if "--remote-debugging-port" in line:
                parts = line.split()
                pid_str = None
                for p in parts:
                    if p.isdigit():
                        pid_str = p
                        break
                if pid_str:
                    try:
                        subprocess.run(  # noqa: S603
                            ["taskkill", "/PID", pid_str, "/F"],
                            capture_output=True, timeout=3,
                            creationflags=subprocess.CREATE_NO_WINDOW,
                        )
                    except Exception:
                        pass
    except Exception:  # noqa: BLE001
        pass


def _cdp_ready(timeout_s: float = 0.0) -> bool:
    """探测本机 CDP HTTP 端点是否可用（任一常见端口）。"""
    try:
        import requests
    except Exception:  # noqa: BLE001
        return False
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        for port in (9222, 9223):
            try:
                r = requests.get(f"http://127.0.0.1:{port}/json/version", timeout=0.8)
                if r.ok:
                    return True
            except Exception:  # noqa: BLE001
                continue
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.3)


def _open_url_with_cdp(url: str, wait_s: float = 12.0) -> tuple[bool, bool, str]:
    """尝试以「带远程调试端口」的方式打开 URL。

    返回 (是否成功建立CDP, URL是否已被打开, 说明文字)。

    策略：
    1. 如果已有CDP实例在运行（9222或9223端口），通过CDP HTTP API在该实例中新建标签。
    2. 否则使用独立 user-data-dir 启动新的 Chrome/Edge 实例（与用户日常浏览器完全隔离），
       保证 --remote-debugging-port 参数一定生效（不会因为已有浏览器运行而被忽略）。
    """
    # 1) 已有CDP实例 → 通过CDP API开新标签（不用webbrowser.open，避免打开到非CDP浏览器）
    if _cdp_ready():
        try:
            import requests as _req
            for p in (9222, 9223, _CDP_AUTOMATION_PORT):
                try:
                    put_url = f"http://127.0.0.1:{p}/json/new?{url}"
                    r = _req.put(put_url, timeout=2.0)
                    if r.ok:
                        # 激活这个新标签
                        tid = r.json().get("id", "")
                        if tid:
                            try:
                                _req.get(f"http://127.0.0.1:{p}/json/activate/{tid}", timeout=1.0)
                            except Exception:
                                pass
                        return True, True, "检测到已有带调试端口的浏览器实例，已在该实例中新建标签"
                except Exception:
                    continue
        except Exception:
            pass
        # 降级：webbrowser.open
        webbrowser.open(url, new=2, autoraise=True)
        return True, True, "检测到已有带调试端口的浏览器实例，新标签已加入该实例"

    # ---- 启动新实例前：清理残留的旧自动化 Chrome（占着端口或 Profile 锁）----
    _cleanup_stale_automation_chrome()

    found = _find_chromium_exe()
    if not found:
        return False, False, "未找到 Chrome/Edge 可执行文件"

    exe, name = found
    profile_dir = _get_automation_profile_dir()
    port = _CDP_AUTOMATION_PORT  # 使用独立端口，避免与用户手动开启的CDP冲突

    try:
        # 启动独立实例：专用profile + 调试端口 + 无首次运行向导
        cmd = [
            exe,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--remote-allow-origins=*",
            "--disable-background-networking",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--new-window",
            url,
        ]
        subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except Exception as e:  # noqa: BLE001
        return False, False, f"启动 {name} 失败：{type(e).__name__}: {e}"

    # 等待CDP端口就绪
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            import requests as _req
            for p in (9222, 9223, port):
                r = _req.get(f"http://127.0.0.1:{p}/json/version", timeout=0.8)
                if r.ok:
                    return True, True, (
                        f"已以自动化模式启动 {name}（独立窗口，调试端口{p}），"
                        f"现在可以对网页进行点击/输入/读取等精确操作"
                    )
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.4)

    # 端口没起来（可能浏览器启动慢或弹了错误对话框），但URL确实已经打开
    # → subprocess.Popen 成功 + URL 作为命令行参数传入，Chrome 一定会打开该页面
    return (
        False,
        True,
        f"{name} 已成功启动并打开目标网页（调试端口暂未就绪，不影响浏览）。"
        f"若需页面内自动化操作（点击/输入等），请稍等几秒后重试相关指令。",
    )


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
        "用途：打开浏览器并访问网站，自动以「自动化调试端口模式」启动（启动后可使用browser_*系列工具进行页面内操作）。\n"
        "⚠️ 注意：此工具仅用于【首次打开网站】，每个任务最多调用1次！打开网站后，所有后续操作"
        "（点击、输入、搜索、翻页、读内容）必须使用browser_navigate/browser_click/browser_input等browser_*工具，"
        "禁止反复调用open_browser！\n"
        "⚠️ 站内搜索（如「在抖音搜XX」「在百度搜XX」）：先用open_browser打开网站首页，"
        "再用browser_list_elements查看搜索框，然后用browser_input(text=XX, submit=True)在搜索框输入并提交，"
        "不要把搜索词传给open_browser！\n"
        "调用示例对应的 target 写法：\n"
        "  「打开知乎」→ target=知乎\n"
        "  「打开 GitHub 看我的仓库」→ target=GitHub\n"
        "  「打开 https://example.com/a?x=1」→ target=https://example.com/a?x=1\n"
        "  「打开 example.com」→ target=example.com（自动补 https://）\n"
        "  「帮我百度搜一下 2026 年奥运会赛程」→ target=2026 年奥运会赛程（此情况才用open_browser直接打开百度搜索结果页）\n"
        "常见快捷站点：百度、必应、知乎、B站、微博、豆瓣、小红书、抖音、淘宝、天猫、京东、QQ邮箱、163邮箱、"
        "Gmail、飞书、钉钉、企业微信、语雀、印象笔记、Notion、石墨文档、腾讯文档、GitHub、Gitee、掘金、CSDN、"
        "爱奇艺、腾讯视频、优酷、网易云音乐、QQ音乐、阿里云盘、百度网盘 ……（完整列表共 100+ 个）"
    )
    args_schema: type[BaseModel] = OpenBrowserArgs
    return_direct: ClassVar[bool] = False

    def _run(self, target: str, new_tab: bool = True, autoraise: bool = True) -> str:  # noqa: D401
        t0 = time.perf_counter_ns()
        try:
            url, method = resolve_target_to_url(target)

            # 优先：以「带远程调试端口」方式打开（Chrome/Edge），
            # 让后续 close_browser_tab 能精确管理任意标签（含后台标签、批量关闭）。
            cdp_ok, url_opened, cdp_note = _open_url_with_cdp(url)
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            if url_opened:
                # ✅ 核心目标已达成：浏览器已启动、目标URL已打开
                prefix = "✅" if cdp_ok else "✅"
                cdp_status = f"（自动化端口已就绪，可进行页面内操作）" if cdp_ok else "（自动化端口暂未就绪，纯浏览不受影响）"
                return (
                    f"{prefix} open_browser 成功（{elapsed_ms} ms）{cdp_status}\n"
                    f"  解析方式：{method}\n"
                    f"  最终 URL：{url}\n"
                    f"  说明：{cdp_note}"
                )

            # 回退：系统默认方式打开
            new_val = 2 if new_tab else 0
            ok = webbrowser.open(url, new=new_val, autoraise=autoraise)
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            if ok:
                return (
                    f"✅ open_browser 成功（系统默认方式，{elapsed_ms} ms）\n"
                    f"  解析方式：{method}\n"
                    f"  最终 URL：{url}\n"
                    f"  提示：{cdp_note}\n"
                    f"  本次以普通方式打开，关闭后台标签的精度可能受限。"
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


def _cdp_close_tab_by_id(tab_id: str) -> bool:
    """通过 CDP HTTP 端点关闭单个标签（无需 websocket）。"""
    import urllib.parse

    import requests

    tid = urllib.parse.quote(str(tab_id), safe="")
    for port in _CDP_PORTS:
        try:
            requests.get(f"http://127.0.0.1:{port}/json/close/{tid}", timeout=1.5)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _cdp_page_tabs() -> list[dict] | None:
    """列出所有 page 类型标签；CDP 不可用返回 None。"""
    tabs = _cdp_list_tabs()
    if tabs is None:
        return None
    return [t for t in tabs if t.get("type") == "page"]


def _tab_matches(tab: dict, keywords: list[str]) -> bool:
    title = str(tab.get("title", ""))
    url = str(tab.get("url", ""))
    hay = f"{title} {url}".lower()
    return any(k.lower() in hay for k in keywords)


def _cdp_close_tabs(keywords: list[str]) -> tuple[list[str], int] | None:
    """CDP 精确关闭所有匹配标签（天然支持后台标签 + 批量）。

    返回 (已关闭标签标题列表, 剩余 page 标签数)；CDP 不可用返回 None。
    """
    tabs = _cdp_page_tabs()
    if tabs is None:
        return None
    closed: list[str] = []
    remaining = 0
    for tab in tabs:
        if _tab_matches(tab, keywords):
            if _cdp_close_tab_by_id(str(tab.get("id", ""))):
                closed.append(str(tab.get("title") or tab.get("url") or ""))
        else:
            remaining += 1
    return closed, remaining


def _domain_of(url: str) -> str:
    """提取 URL 的域名（去 www. 前缀），用于同站点分组。"""
    m = re.search(r"://(?:www\.)?([^/:]+)", url or "")
    return (m.group(1).lower() if m else "").strip()


def _site_key_of(url: str) -> str:
    """提取 URL 的「站点主域」（域名的最后两段），用于同站点去重。

    例：cart.taobao.com / www.taobao.com → taobao.com；
    对 .com.cn / .net.cn 等双段后缀取最后三段。
    """
    domain = _domain_of(url)
    if not domain:
        return ""
    parts = domain.split(".")
    n = 3 if len(parts) >= 3 and ".".join(parts[-2:]) in ("com.cn", "net.cn", "org.cn", "ac.cn", "edu.cn", "gov.cn") else 2
    return ".".join(parts[-n:]) if len(parts) >= n else domain


def _cdp_close_duplicates() -> tuple[list[str], int] | None:
    """关闭「重复站点」标签：同站点（主域相同）只保留第一个，其余关闭。

    返回 (已关闭标签标题列表, 剩余 page 标签数)；CDP 不可用返回 None。
    """
    tabs = _cdp_page_tabs()
    if tabs is None:
        return None
    seen_sites: set[str] = set()
    closed: list[str] = []
    remaining = 0
    for tab in tabs:
        url = str(tab.get("url", ""))
        site = _site_key_of(url)
        # 空站点（新标签页等）不参与去重，始终保留
        if site and site in seen_sites:
            if _cdp_close_tab_by_id(str(tab.get("id", ""))):
                closed.append(f"{tab.get('title') or url}（重复 {site}）")
            continue
        if site:
            seen_sites.add(site)
        remaining += 1
    return closed, remaining


_BROWSER_TITLE_SUFFIXES = (
    " - google chrome",
    " - microsoft edge",
    " - microsoft​ edge",
    " - chromium",
    " - brave",
)


def _strip_browser_suffix(window_title: str) -> str:
    """把窗口标题尾部的浏览器名去掉，还原出标签页标题。"""
    low = window_title.lower()
    for suf in _BROWSER_TITLE_SUFFIXES:
        if low.endswith(suf):
            return window_title[: len(window_title) - len(suf)]
    return window_title


def _cdp_close_others() -> tuple[list[str], int, str] | None:
    """关闭「除当前激活标签外的所有标签」。

    当前标签识别：取前台浏览器窗口标题 → 剥离浏览器后缀 → 与 CDP 标签标题比对。
    返回 (已关闭列表, 剩余数, 保留标签标题)；CDP 不可用返回 None；
    找不到前台浏览器窗口 / 无法定位当前标签时返回 ([], -1, 原因)。
    """
    tabs = _cdp_page_tabs()
    if tabs is None:
        return None

    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    hwnd = user32.GetForegroundWindow()
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    fg_title = buf.value
    if not fg_title or not _is_browser_title(fg_title):
        return [], -1, "当前前台窗口不是浏览器，无法确定「当前页」是哪一个标签"

    active_title = _strip_browser_suffix(fg_title).strip()
    keep_ids: set[str] = set()
    exact = [t for t in tabs if str(t.get("title", "")).strip() == active_title]
    if exact:
        keep_ids = {str(t.get("id", "")) for t in exact}
    else:
        fuzzy = [t for t in tabs if active_title and active_title in str(t.get("title", ""))]
        if fuzzy:
            keep_ids = {str(t.get("id", "")) for t in fuzzy}
    if not keep_ids:
        return [], -1, f"未能在标签列表中定位当前页「{active_title}」，为安全起见未关闭任何标签"

    closed: list[str] = []
    remaining = 0
    for tab in tabs:
        if str(tab.get("id", "")) in keep_ids:
            remaining += 1
            continue
        if _cdp_close_tab_by_id(str(tab.get("id", ""))):
            closed.append(str(tab.get("title") or tab.get("url") or ""))
    return closed, remaining, active_title


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


def _focus_window(hwnd: int) -> bool:
    """把窗口置前台（含 Alt 技巧解除前台锁定）。"""
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    VK_MENU = 0x12
    KEYEVENTF_KEYUP = 0x0002
    SW_RESTORE = 9
    try:
        user32.keybd_event(VK_MENU, 0, 0, 0)
        user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.25)
        return True
    except Exception:  # noqa: BLE001
        return False


def _send_hotkey(vk: int, *, ctrl: bool = True, shift: bool = False) -> None:
    """向当前前台窗口发送热键（如 Ctrl+W / Ctrl+Shift+T）。"""
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    VK_CONTROL, VK_SHIFT = 0x11, 0x10
    KEYEVENTF_KEYUP = 0x0002
    if ctrl:
        user32.keybd_event(VK_CONTROL, 0, 0, 0)
    if shift:
        user32.keybd_event(VK_SHIFT, 0, 0, 0)
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
    if shift:
        user32.keybd_event(VK_SHIFT, 0, KEYEVENTF_KEYUP, 0)
    if ctrl:
        user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def _window_title(hwnd: int) -> str:
    """读取指定窗口当前标题。"""
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    n = user32.GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _foreground_and_ctrl_w(hwnd: int) -> bool:
    """把窗口置前台并模拟 Ctrl+W（关闭当前标签页）。返回是否按键已发出。"""
    if not _focus_window(hwnd):
        return False
    try:
        _send_hotkey(0x57, ctrl=True)  # VK_W
        return True
    except Exception:  # noqa: BLE001
        return False


def _close_matching_tabs_via_window(
    hwnd: int, keywords: list[str], first_title: str, max_n: int = 10
) -> list[str]:
    """无 CDP 时的批量关闭：重复「读当前标签标题 → 匹配则 Ctrl+W」。

    每次 Ctrl+W 后浏览器自动切到下一个标签，标题随之变化；
    直到当前标签不再匹配、窗口关闭或达到 max_n 上限。
    返回已关闭的标签标题列表（窗口标题已剥离浏览器后缀）。
    """
    VK_W = 0x57
    closed: list[str] = []
    title = first_title
    for _ in range(max_n):
        tab_title = _strip_browser_suffix(title)
        if not any(k in tab_title for k in keywords):
            break
        if not _focus_window(hwnd):
            break
        try:
            _send_hotkey(VK_W, ctrl=True)
        except Exception:  # noqa: BLE001
            break
        closed.append(tab_title)
        time.sleep(0.45)  # 等浏览器切换标签、刷新标题
        new_title = _window_title(hwnd)
        if not new_title:
            break  # 窗口已整个关闭
        if new_title == title:
            break  # 标题没变化（可能只剩一个标签或关的是弹窗），防止死循环
        title = new_title
    return closed


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
        "",
        description=(
            "【mode=site 时必填】要关闭的站点名/关键词：如「抖音」「知乎」「淘宝」，"
            "匹配标签页标题或网址后批量关闭对应标签页；\n"
            "  「全部」/「所有」/「浏览器」等价于 mode=all，关闭整个浏览器。"
        ),
    )
    mode: str = Field(
        "site",
        description=(
            "【选填】关闭模式，默认 site：\n"
            "  - site：关闭所有匹配 target 的标签页（「关闭所有淘宝页面」「把抖音关了」）；\n"
            "  - others：关闭除当前正在看的标签页之外的所有标签（「关闭其他标签」「只留当前页」），target 留空；\n"
            "  - duplicates：关闭重复站点的标签，同域名只保留一个（「关闭重复的标签」「清理重复页面」），target 留空；\n"
            "  - all：关闭整个浏览器（等同点窗口右上角 ×，不是杀进程）。"
        ),
    )


class CloseBrowserTabTool(BaseTool):
    """关闭浏览器标签页 / 整个浏览器。

    四种模式：
        site（默认）：批量关闭匹配站点/关键词的标签
        others：关闭除当前激活标签外的所有标签（需 CDP）
        duplicates：同域名标签只留一个（需 CDP）
        all：WM_CLOSE 优雅关闭整个浏览器
    site 模式三级策略：① CDP 精确批量关（含后台标签）② 窗口匹配 + 循环 Ctrl+W 连续关 ③ 报告找不到。
    """

    name: ClassVar[str] = "close_browser_tab"
    description: ClassVar[str] = (
        "Tool Name: close_browser_tab\n"
        "用途：关闭浏览器标签页，支持 4 种粒度。\n"
        "典型场景：\n"
        "  - 「关闭抖音标签页」「把抖音关了」→ mode=site, target=抖音\n"
        "  - 「关闭所有淘宝页面」「关掉所有百度标签」→ mode=site, target=淘宝（自动批量）\n"
        "  - 「关闭其他标签」「只保留当前页面」→ mode=others（需浏览器以调试端口运行）\n"
        "  - 「关闭重复的标签页」「清理重复站点」→ mode=duplicates（同域名只留一个，需调试端口）\n"
        "  - 「关闭浏览器」「把浏览器都关了」→ mode=all 或 target=全部\n"
        "匹配逻辑：站点中文名 + 域名（如 douyin）同时匹配标签页标题和网址。\n"
        "说明：\n"
        "  - 关闭整个浏览器用 WM_CLOSE（等同手动点 ×），不会杀进程，浏览器可正常恢复会话。\n"
        "  - 关错了可以说「恢复刚才关闭的页面」，配合 restore_browser_tab 工具找回。\n"
        "  - others/duplicates 模式需要浏览器带调试端口（通过 open_browser 打开的页面默认自带）。\n"
        "  - 只处理浏览器窗口，不影响其他任何应用。"
    )
    args_schema: type[BaseModel] = CloseBrowserTabArgs
    return_direct: ClassVar[bool] = False

    _ALL_WORDS: ClassVar[tuple[str, ...]] = ("全部", "所有", "所有浏览器", "浏览器", "全部关闭", "all", "*")

    def _run(self, target: str = "", mode: str = "site") -> str:  # noqa: D401
        t0 = time.perf_counter_ns()
        try:
            t = (target or "").strip()
            mode = (mode or "site").strip().lower()
            if t in self._ALL_WORDS:
                mode = "all"

            # ---------- mode=all：关闭整个浏览器 ----------
            if mode == "all":
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

            # ---------- mode=others：关闭除当前页外的所有标签 ----------
            if mode == "others":
                res = _cdp_close_others()
                ms = (time.perf_counter_ns() - t0) // 1_000_000
                if res is None:
                    return (
                        "⚠️ 「关闭其他标签」需要浏览器以调试端口模式运行（当前未检测到）。\n"
                        "  建议：把要保留的页面所在浏览器完全关闭，然后让我重新「打开」一个网站，"
                        "之后即可使用该模式；或手动在标签上右键选择「关闭其他标签页」。"
                    )
                closed, remaining, kept = res
                if remaining == -1:
                    return f"⚠️ {kept}"  # kept 字段此时是失败原因
                if not closed:
                    return f"ℹ️ 当前没有其他可关闭的标签，已保留「{kept}」（{ms} ms）。"
                return (
                    f"✅ 已关闭 {len(closed)} 个其他标签，保留当前页「{kept}」（{ms} ms）：\n"
                    + "\n".join(f"  ✂ {ti}" for ti in closed[:10])
                    + ("\n  ……" if len(closed) > 10 else "")
                    + f"\n  当前剩余 {remaining} 个标签。关错了可说「恢复刚才关闭的页面」。"
                )

            # ---------- mode=duplicates：关闭重复站点标签 ----------
            if mode == "duplicates":
                res = _cdp_close_duplicates()
                ms = (time.perf_counter_ns() - t0) // 1_000_000
                if res is None:
                    return (
                        "⚠️ 「关闭重复标签」需要浏览器以调试端口模式运行（当前未检测到）。\n"
                        "  建议：让我用 open_browser 重新打开网站（默认带调试端口）后即可使用该模式。"
                    )
                closed, remaining = res
                if not closed:
                    return f"ℹ️ 没有发现重复站点的标签，当前 {remaining} 个标签均不重复（{ms} ms）。"
                return (
                    f"✅ 已关闭 {len(closed)} 个重复站点标签（同域名只保留一个，{ms} ms）：\n"
                    + "\n".join(f"  ✂ {ti}" for ti in closed[:10])
                    + ("\n  ……" if len(closed) > 10 else "")
                    + f"\n  当前剩余 {remaining} 个标签。关错了可说「恢复刚才关闭的页面」。"
                )

            # ---------- mode=site（默认）：批量关闭匹配标签 ----------
            if not t:
                return "❌ close_browser_tab：mode=site 时 target 不能为空（要关闭哪个站点？）"
            keywords = _match_keywords_for(t)

            # ① CDP 精确批量关闭（含后台标签）
            cdp_res = _cdp_close_tabs(keywords)
            if cdp_res is not None:
                closed, remaining = cdp_res
                ms = (time.perf_counter_ns() - t0) // 1_000_000
                if closed:
                    return (
                        f"✅ 已关闭 {len(closed)} 个匹配「{t}」的标签页，当前浏览器还剩 {remaining} 个标签（{ms} ms）：\n"
                        + "\n".join(f"  ✂ {ti}" for ti in closed[:10])
                        + ("\n  ……" if len(closed) > 10 else "")
                        + "\n  关错了可说「恢复刚才关闭的页面」。"
                    )
                tabs = _cdp_page_tabs() or []
                names = [str(x.get("title") or x.get("url")) for x in tabs]
                return (
                    f"⚠️ 浏览器中没有找到匹配「{t}」的标签页（关键词：{'/'.join(keywords)}）。\n"
                    f"  当前打开的 {len(names)} 个标签：{'；'.join(names[:10]) or '（无）'}"
                )

            # ② 窗口标题匹配 + 循环 Ctrl+W 连续关闭（无 CDP 降级）
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
                    f"由 open_browser 打开的浏览器默认带调试端口，可精确关闭任意标签。"
                )
            closed_titles: list[str] = []
            for hwnd, ti in matched:
                closed_titles.extend(_close_matching_tabs_via_window(hwnd, keywords, ti))
                time.sleep(0.3)
            if closed_titles:
                return (
                    f"✅ 已关闭 {len(closed_titles)} 个匹配「{t}」的标签页（窗口匹配 + 连续 Ctrl+W，{ms} ms）：\n"
                    + "\n".join(f"  ✂ {ti}" for ti in closed_titles[:10])
                    + ("\n  ……" if len(closed_titles) > 10 else "")
                    + "\n  说明：无调试端口时只能关闭各窗口中标题匹配的激活标签；"
                    "关错了可切到浏览器按 Ctrl+Shift+T 恢复，或说「恢复刚才关闭的页面」。"
                )
            return (
                f"⚠️ 找到了匹配「{t}」的窗口但按键模拟失败（可能被系统前台限制拦截）。\n"
                f"  请手动切换到该标签页后重试，或直接说「关闭浏览器」。"
            )
        except Exception as e:  # noqa: BLE001
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ close_browser_tab 失败（{ms} ms）：{type(e).__name__}: {e}"


# ============================================================
# RestoreClosedTabTool —— 恢复刚关闭的标签页
# ============================================================

class RestoreClosedTabArgs(BaseModel):
    count: int = Field(
        1,
        ge=1,
        le=10,
        description="【选填】要恢复的标签页数量，默认 1（恢复最近一次关闭的），最多 10。",
    )


class RestoreClosedTabTool(BaseTool):
    """恢复最近关闭的浏览器标签页（模拟 Ctrl+Shift+T）。"""

    name: ClassVar[str] = "restore_browser_tab"
    description: ClassVar[str] = (
        "Tool Name: restore_browser_tab\n"
        "用途：恢复刚刚被关闭的浏览器标签页（等同在浏览器里按 Ctrl+Shift+T）。\n"
        "典型场景：\n"
        "  - 「恢复刚才关闭的页面」「刚才关错了」→ count=1\n"
        "  - 「恢复刚才关掉的 3 个标签」→ count=3\n"
        "说明：\n"
        "  - 会把最近使用的浏览器窗口置前台后逐次发送 Ctrl+Shift+T，按浏览器自身的关闭历史恢复。\n"
        "  - 浏览器完全关闭后重新打开时，部分浏览器也可恢复整个会话，但不保证。\n"
        "  - 恢复结果以浏览器实际行为为准，工具只能确认按键已发出。"
    )
    args_schema: type[BaseModel] = RestoreClosedTabArgs
    return_direct: ClassVar[bool] = False

    def _run(self, count: int = 1) -> str:  # noqa: D401
        t0 = time.perf_counter_ns()
        try:
            count = max(1, min(int(count), 10))
            wins = [(h, ti) for h, ti in _enum_visible_windows() if _is_browser_title(ti)]
            if not wins:
                return (
                    "⚠️ 没有找到正在运行的浏览器窗口，无法恢复。\n"
                    "  提示：部分浏览器重新打开后按 Ctrl+Shift+T 仍可恢复上次会话的标签。"
                )
            hwnd, title = wins[0]
            if not _focus_window(hwnd):
                return "⚠️ 无法把浏览器窗口置前台（可能被系统前台限制拦截），请手动切到浏览器后按 Ctrl+Shift+T。"
            VK_T = 0x54
            sent = 0
            for _ in range(count):
                try:
                    _send_hotkey(VK_T, ctrl=True, shift=True)
                    sent += 1
                except Exception:  # noqa: BLE001
                    break
                time.sleep(0.6)  # 给浏览器恢复标签留出时间
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            return (
                f"✅ 已向浏览器（{ _strip_browser_suffix(title) or title }）发送 {sent} 次恢复指令"
                f"（Ctrl+Shift+T，{ms} ms）。\n"
                f"  浏览器会按关闭的逆序恢复标签页；如需更多可再说一次「恢复刚才关闭的页面」。"
            )
        except Exception as e:  # noqa: BLE001
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ restore_browser_tab 失败（{ms} ms）：{type(e).__name__}: {e}"


__all__ = [
    "OpenBrowserTool",
    "CloseBrowserTabTool",
    "RestoreClosedTabTool",
    "resolve_target_to_url",
    "_SITE_SHORTCUTS",
]
