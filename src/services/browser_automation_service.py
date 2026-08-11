"""浏览器自动化服务：通过 Chrome DevTools Protocol (CDP) 控制带调试端口的浏览器。

核心能力：
    - CDP HTTP 端点探测与标签页管理
    - WebSocket 长连接发送 CDP 命令
    - DOM 可交互元素提取（按钮/链接/输入框等）
    - 页面截图（用于视觉定位）
    - 原子操作：点击/输入/滚动/导航/提取文本

安全边界：
    - 仅操作通过 open_browser 启动的带调试端口浏览器实例
    - 默认域名白名单：输入/提交等高危操作需校验
    - 所有异常包装返回，不崩溃主流程
"""
from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

import requests

try:
    import websocket
except ImportError:
    websocket = None  # type: ignore


# ============================================================
# 常量
# ============================================================

# 探测端口：与 browser_tools.py 的 _CDP_AUTOMATION_PORT(9223) 保持一致
CDP_PORTS = (9222, 9223)
CDP_HTTP_TIMEOUT = 3.0
CDP_WS_TIMEOUT = 10.0
MAX_WS_MSG_ID = 2**31 - 1

# 高危域名白名单（默认允许完整操作）
_TRUSTED_DOMAINS = {
    # 搜索引擎
    "baidu.com", "bing.com", "google.com", "sogou.com", "so.com", "duckduckgo.com",
    # 内容社区
    "zhihu.com", "bilibili.com", "weibo.com", "douban.com", "xiaohongshu.com",
    "douyin.com", "kuaishou.com", "juejin.cn", "csdn.net", "cnblogs.com",
    "segmentfault.com", "github.com", "gitee.com", "stackoverflow.com",
    # 电商
    "taobao.com", "tmall.com", "jd.com", "pinduoduo.com",
    # 视频/音乐
    "iqiyi.com", "v.qq.com", "youku.com", "mgtv.com", "ixigua.com",
    "music.163.com", "y.qq.com",
    # 办公
    "feishu.cn", "dingtalk.com", "work.weixin.qq.com", "yuque.com",
    "notion.so", "shimo.im", "docs.qq.com", "kdocs.cn",
    # 邮箱
    "mail.qq.com", "mail.163.com", "mail.126.com",
}

# 敏感操作关键词（检测到需二次确认）
_DANGEROUS_KEYWORDS = {
    "支付", "付款", "下单", "购买", "确认订单", "提交订单",
    "删除", "清空", "注销", "解绑", "转账", "汇款",
    "密码", "验证码", "身份证", "银行卡", "信用卡",
}


# ============================================================
# 数据结构
# ============================================================

@dataclass
class DOMElement:
    """可交互元素的结构化表示。"""
    index: int                       # 序号（从0开始，用于LLM定位）
    tag: str                         # 标签名 a/button/input/select/textarea 等
    text: str                        # 可见文本（截断到80字）
    aria_label: str                  # aria-label 属性
    placeholder: str                 # placeholder（输入框）
    element_id: str                  # id 属性
    class_name: str                  # class 属性
    element_type: str                # type 属性（input的type: text/submit/checkbox...）
    href: str                        # href（链接）
    role: str                        # role 属性
    rect: dict                       # 位置 {x, y, width, height}
    is_visible: bool                 # 是否可见
    is_enabled: bool                 # 是否可用（非disabled）

    def to_llm_desc(self) -> str:
        """给LLM看的精简描述。"""
        parts = [f"[{self.index}]"]
        # 类型标签
        type_label = self._type_label()
        parts.append(f"<{type_label}>")
        # 文本内容
        desc_text = self.text or self.aria_label or self.placeholder or self.element_id or self.class_name
        if desc_text:
            parts.append(f'文本="{desc_text[:60]}"')
        # 特殊属性
        if self.placeholder and self.placeholder != desc_text:
            parts.append(f'placeholder="{self.placeholder[:40]}"')
        if self.href and self.href.startswith("http"):
            parts.append(f'href="{self.href[:50]}"')
        # 位置（辅助判断方位）
        if self.rect.get("width", 0) > 0:
            parts.append(f'位置=({int(self.rect["x"])},{int(self.rect["y"])})')
        # 状态
        if not self.is_enabled:
            parts.append("[禁用]")
        if not self.is_visible:
            parts.append("[隐藏]")
        return " ".join(parts)

    def _type_label(self) -> str:
        """人类可读的元素类型名。"""
        t = self.tag.lower()
        if t == "a":
            return "链接"
        if t == "button" or (t == "input" and self.element_type in ("submit", "button", "reset")):
            return "按钮"
        if t in ("input", "textarea"):
            input_type = self.element_type or "text"
            type_map = {
                "text": "文本框", "search": "搜索框", "password": "密码框",
                "email": "邮箱框", "number": "数字框", "tel": "电话框",
                "checkbox": "复选框", "radio": "单选框",
                "submit": "提交按钮", "button": "按钮",
            }
            return type_map.get(input_type, f"输入框({input_type})")
        if t == "select":
            return "下拉选择"
        if self.role in ("button", "link"):
            return "按钮" if self.role == "button" else "链接"
        return t


@dataclass
class PageInfo:
    """页面基本信息。"""
    url: str = ""
    title: str = ""
    ready_state: str = ""
    elements: list[DOMElement] = field(default_factory=list)


@dataclass
class CDPConnection:
    """单个标签页的CDP WebSocket连接。"""
    tab_id: str
    ws_url: str
    ws: Optional[Any] = None
    msg_id: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    connected: bool = False


# ============================================================
# JS 脚本（注入页面执行）
# ============================================================

# 提取所有可交互元素的JS
_EXTRACT_INTERACTABLE_JS = r"""
(() => {
    const INTERACTABLE_SELECTORS = [
        'a[href]', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
        '[role="button"]', '[role="link"]', '[role="menuitem"]',
        '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
        '[role="switch"]', '[role="textbox"]', '[role="combobox"]',
        '[role="searchbox"]', '[role="search"]', '[role="option"]',
        '[role="spinbutton"]', '[role="slider"]', '[role="menuitemcheckbox"]',
        '[role="menuitemradio"]',
        '[onclick]', '[contenteditable="true"]',
        'summary', 'label',
        '[tabindex]:not([tabindex="-1"])',
        '[class*="search"] input', '[class*="search"] textarea',
        '[class*="Search"] input', '[class*="Search"] textarea',
    ];
    const results = [];
    const seen = new Set();
    const selector = INTERACTABLE_SELECTORS.join(',');
    let els = document.querySelectorAll(selector);
    // 去重（同一个元素可能被多个selector命中）
    const uniqueEls = [];
    for (const el of els) {
        if (!seen.has(el)) {
            seen.add(el);
            uniqueEls.push(el);
        }
    }
    let idx = 0;
    for (const el of uniqueEls) {
        // 跳过明显不可见/不可用的
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        const isVisible = (
            style.display !== 'none' &&
            style.visibility !== 'hidden' &&
            rect.width > 0 && rect.height > 0 &&
            rect.top < window.innerHeight + 500 && rect.bottom > -100 &&
            rect.left < window.innerWidth && rect.right > 0 &&
            style.opacity !== '0'
        );
        if (!isVisible) continue;
        // 跳过被遮挡的元素（粗略判断：中心点的elementFromPoint不是自己或子元素）
        const cx = rect.left + rect.width / 2 + window.scrollX;
        const cy = rect.top + rect.height / 2 + window.scrollY;
        let topEl = null;
        try { topEl = document.elementFromPoint(cx - window.scrollX, cy - window.scrollY); } catch(e) {}
        let isInteractable = true;
        if (topEl && topEl !== el && !el.contains(topEl) && !topEl.contains(el)) {
            // 被遮挡，检查是否仍可交互（浮层按钮/搜索框可能被自身标签遮挡）
            isInteractable = (el.tagName === 'A' || el.tagName === 'BUTTON' ||
                             el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' ||
                             el.getAttribute('role') === 'button' ||
                             el.getAttribute('role') === 'searchbox' ||
                             el.getAttribute('role') === 'textbox' ||
                             el.getAttribute('contenteditable') === 'true');
        }
        if (!isInteractable) continue;

        const getAttr = (name) => el.getAttribute(name) || '';
        const text = (el.innerText || el.textContent || '').trim().replace(/\s+/g, ' ');
        const ariaLabel = getAttr('aria-label') || getAttr('aria-labelledby') ?
            (() => { const id = getAttr('aria-labelledby');
              if (id) { const lab = document.getElementById(id); return lab ? (lab.innerText || '').trim() : ''; }
              return getAttr('aria-label');
            })() : '';
        const placeholder = getAttr('placeholder');
        // 跳过太小的纯装饰元素（宽或高<8px且无文本）
        if (rect.width < 8 || rect.height < 8) {
            if (!text && !ariaLabel && !placeholder && !getAttr('title')) continue;
        }
        results.push({
            index: idx++,
            tag: el.tagName.toLowerCase(),
            text: text.substring(0, 200),
            aria_label: ariaLabel.substring(0, 100),
            placeholder: placeholder,
            id: getAttr('id'),
            class_name: getAttr('class').substring(0, 80),
            type: getAttr('type'),
            href: getAttr('href'),
            role: getAttr('role'),
            rect: { x: rect.left, y: rect.top, width: rect.width, height: rect.height },
            is_visible: true,
            is_enabled: !el.disabled && !el.hasAttribute('disabled') && !el.hasAttribute('aria-disabled'),
            _xpath: getXPath(el),
        });
    }
    function getXPath(element) {
        if (element.id) return '//*[@id="' + element.id.replace(/"/g, '\\"') + '"]';
        if (element === document.body) return '/html/body';
        let path = '';
        let current = element;
        while (current && current.nodeType === 1 && current !== document.body) {
            let idx = 1;
            let sibling = current.previousSibling;
            while (sibling) {
                if (sibling.nodeType === 1 && sibling.tagName === current.tagName) idx++;
                sibling = sibling.previousSibling;
            }
            path = '/' + current.tagName.toLowerCase() + '[' + idx + ']' + path;
            current = current.parentNode;
        }
        return '/html/body' + path;
    }
    return {
        url: location.href,
        title: document.title,
        readyState: document.readyState,
        elements: results,
        total: results.length,
    };
})()
"""

# 滚动JS
_SCROLL_JS = {
    "down": "window.scrollBy(0, window.innerHeight * 0.8)",
    "up": "window.scrollBy(0, -window.innerHeight * 0.8)",
    "top": "window.scrollTo(0, 0)",
    "bottom": "window.scrollTo(0, document.body.scrollHeight)",
}


# ============================================================
# BrowserAutomationService 核心类
# ============================================================

class BrowserAutomationService:
    """浏览器自动化服务单例。"""

    _instance: Optional["BrowserAutomationService"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._connections: dict[str, CDPConnection] = {}
        self._active_tab_id: Optional[str] = None
        self._last_error: str = ""

    @classmethod
    def get_instance(cls) -> "BrowserAutomationService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ---------------- CDP 连接管理 ----------------

    def is_available(self) -> bool:
        """检测是否有可用的CDP浏览器实例。"""
        return self._get_cdp_base() is not None

    def _get_cdp_base(self) -> Optional[str]:
        """探测可用的CDP HTTP端点，返回base URL。"""
        for port in CDP_PORTS:
            try:
                r = requests.get(f"http://127.0.0.1:{port}/json/version", timeout=CDP_HTTP_TIMEOUT)
                if r.ok:
                    return f"http://127.0.0.1:{port}"
            except Exception:
                continue
        return None

    def list_tabs(self) -> list[dict]:
        """列出所有page类型标签页。"""
        base = self._get_cdp_base()
        if not base:
            return []
        try:
            r = requests.get(f"{base}/json", timeout=CDP_HTTP_TIMEOUT)
            if not r.ok:
                return []
            return [t for t in r.json() if isinstance(t, dict) and t.get("type") == "page"]
        except Exception:
            return []

    def _get_active_tab(self) -> Optional[dict]:
        """获取当前激活/最近的标签页。"""
        tabs = self.list_tabs()
        if not tabs:
            return None
        # 优先取非空URL且非新标签页
        for t in tabs:
            url = t.get("url", "")
            if url and not url.startswith("chrome://") and not url.startswith("edge://") and not url.startswith("about:"):
                return t
        return tabs[0]

    def _ensure_ws(self, tab: dict) -> Optional[CDPConnection]:
        """确保指定标签页有可用的WebSocket连接。"""
        tab_id = str(tab.get("id", ""))
        ws_url = str(tab.get("webSocketDebuggerUrl", ""))
        if not tab_id or not ws_url:
            self._last_error = "标签页缺少webSocketDebuggerUrl"
            return None

        # 已有连接且可用
        if tab_id in self._connections:
            conn = self._connections[tab_id]
            if conn.connected and conn.ws and conn.ws.connected:
                self._active_tab_id = tab_id
                return conn
            # 连接断开，重新建
            try:
                conn.ws.close()
            except Exception:
                pass
            del self._connections[tab_id]

        if websocket is None:
            self._last_error = "websocket-client未安装，请pip install websocket-client"
            return None

        # 新建连接
        try:
            ws = websocket.create_connection(
                ws_url,
                timeout=CDP_WS_TIMEOUT,
                enable_multithread=True,
            )
            conn = CDPConnection(tab_id=tab_id, ws_url=ws_url, ws=ws, connected=True)
            self._connections[tab_id] = conn
            self._active_tab_id = tab_id
            return conn
        except Exception as e:
            self._last_error = f"WebSocket连接失败：{type(e).__name__}: {e}"
            return None

    def _send_cdp(self, method: str, params: dict | None = None, timeout: float = 8.0) -> dict:
        """发送CDP命令并等待结果。"""
        tab = self._get_active_tab()
        if not tab:
            return {"ok": False, "error": "没有检测到可用的浏览器标签页（请先用open_browser打开网站）"}
        conn = self._ensure_ws(tab)
        if not conn:
            return {"ok": False, "error": self._last_error or "CDP连接失败"}

        with conn.lock:
            conn.msg_id = (conn.msg_id + 1) % MAX_WS_MSG_ID
            msg_id = conn.msg_id
            payload = json.dumps({"id": msg_id, "method": method, "params": params or {}})
            try:
                conn.ws.settimeout(timeout)
                conn.ws.send(payload)
                # 读取响应，跳过事件消息
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    try:
                        raw = conn.ws.recv()
                    except Exception as e:
                        conn.connected = False
                        return {"ok": False, "error": f"WebSocket接收失败：{e}"}
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if data.get("id") == msg_id:
                        if "error" in data:
                            return {"ok": False, "error": str(data["error"])}
                        return {"ok": True, "result": data.get("result", {})}
                return {"ok": False, "error": "CDP命令超时"}
            except Exception as e:
                conn.connected = False
                return {"ok": False, "error": f"CDP发送失败：{type(e).__name__}: {e}"}

    def _eval_js(self, expression: str, await_promise: bool = False, timeout: float = 8.0) -> dict:
        """执行JS表达式，返回结果值。"""
        res = self._send_cdp("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": await_promise,
            "timeout": int(timeout * 1000),
        }, timeout=timeout + 2)
        if not res.get("ok"):
            return res
        result = res["result"].get("result", {})
        if result.get("subtype") == "error":
            return {"ok": False, "error": result.get("description", str(result))}
        return {"ok": True, "value": result.get("value")}

    # ---------------- 页面信息获取 ----------------

    def get_page_info(self, extract_elements: bool = True, max_elements: int = 60) -> PageInfo:
        """获取当前页面信息和可交互元素列表。"""
        info = PageInfo()
        if not self.is_available():
            return info

        # 获取URL/Title
        nav = self._eval_js("({url: location.href, title: document.title, readyState: document.readyState})")
        if nav.get("ok") and isinstance(nav.get("value"), dict):
            v = nav["value"]
            info.url = v.get("url", "")
            info.title = v.get("title", "")
            info.ready_state = v.get("readyState", "")

        if extract_elements:
            # 等待页面稳定
            self._wait_for_ready(timeout=3.0)
            res = self._eval_js(_EXTRACT_INTERACTABLE_JS, timeout=5.0)
            if res.get("ok") and isinstance(res.get("value"), dict):
                data = res["value"]
                for item in data.get("elements", [])[:max_elements]:
                    info.elements.append(DOMElement(
                        index=item.get("index", 0),
                        tag=item.get("tag", ""),
                        text=item.get("text", ""),
                        aria_label=item.get("aria_label", ""),
                        placeholder=item.get("placeholder", ""),
                        element_id=item.get("id", ""),
                        class_name=item.get("class_name", ""),
                        element_type=item.get("type", ""),
                        href=item.get("href", ""),
                        role=item.get("role", ""),
                        rect=item.get("rect", {}),
                        is_visible=item.get("is_visible", True),
                        is_enabled=item.get("is_enabled", True),
                    ))
        return info

    def _wait_for_ready(self, timeout: float = 3.0) -> None:
        """等待页面加载完成。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            res = self._eval_js("document.readyState")
            if res.get("ok") and res.get("value") == "complete":
                return
            time.sleep(0.3)

    # ---------------- 截图 ----------------

    def capture_screenshot(self) -> Optional[bytes]:
        """截取当前页面完整视口截图，返回PNG字节；失败返回None。"""
        res = self._send_cdp("Page.captureScreenshot", {
            "format": "png",
            "captureBeyondViewport": False,
        }, timeout=10.0)
        if not res.get("ok"):
            return None
        data = res["result"].get("data", "")
        if not data:
            return None
        try:
            return base64.b64decode(data)
        except Exception:
            return None

    # ---------------- 原子操作 ----------------

    def navigate(self, url: str) -> dict:
        """页面导航（跳转到指定URL）。"""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        # Page.navigate
        res = self._send_cdp("Page.navigate", {"url": url})
        if not res.get("ok"):
            return res
        # 等待加载
        time.sleep(1.0)
        self._wait_for_ready(timeout=5.0)
        return {"ok": True, "message": f"已导航到 {url}"}

    def page_reload(self) -> dict:
        """刷新页面。"""
        res = self._send_cdp("Page.reload", {"ignoreCache": False})
        if res.get("ok"):
            time.sleep(0.8)
            self._wait_for_ready(timeout=5.0)
        return res

    def page_history(self, delta: int) -> dict:
        """前进/后退：delta=-1后退，delta=1前进。"""
        # 先启用Page域
        self._send_cdp("Page.enable")
        res = self._eval_js(f"history.go({delta})")
        time.sleep(0.8)
        self._wait_for_ready(timeout=4.0)
        if res.get("ok"):
            return {"ok": True, "message": "已导航" if delta != 0 else ""}
        return res

    def scroll_page(self, direction: str = "down") -> dict:
        """页面滚动：down/up/top/bottom。"""
        js = _SCROLL_JS.get(direction, _SCROLL_JS["down"])
        res = self._eval_js(js)
        if res.get("ok"):
            names = {"down": "向下翻页", "up": "向上翻页", "top": "回到顶部", "bottom": "滚到底部"}
            return {"ok": True, "message": names.get(direction, "滚动")}
        return res

    def click_element_by_index(self, index: int) -> dict:
        """按元素序号点击（序号来自get_page_info的DOMElement.index）。"""
        info = self.get_page_info(extract_elements=True)
        target = None
        for el in info.elements:
            if el.index == index:
                target = el
                break
        if not target:
            return {"ok": False, "error": f"未找到序号为 {index} 的元素，当前页面共 {len(info.elements)} 个可交互元素"}
        # 构建与_EXTRACT_INTERACTABLE_JS一致的选择器
        selectors = [
            'a[href]', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
            '[role="button"]', '[role="link"]', '[role="menuitem"]',
            '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
            '[role="switch"]', '[role="textbox"]', '[role="combobox"]',
            '[role="searchbox"]', '[role="search"]',
            '[onclick]', '[contenteditable="true"]',
            'summary', '[tabindex]:not([tabindex="-1"])',
        ]
        sel_str = ','.join(selectors)
        js = f"""
        (() => {{
            const els = document.querySelectorAll('{sel_str}');
            let visible = Array.from(els).filter(el => {{
                const r = el.getBoundingClientRect();
                const s = getComputedStyle(el);
                return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 0 && r.height > 0 && s.opacity !== '0';
            }});
            // 去重
            const seen = new Set();
            visible = visible.filter(el => {{ if(seen.has(el)) return false; seen.add(el); return true; }});
            const el = visible[{index}];
            if (!el) return {{ok: false, error: '元素不存在'}};
            if (el.disabled || el.getAttribute('aria-disabled') === 'true') return {{ok: false, error: '元素已禁用'}};
            el.scrollIntoView({{block: 'center', behavior: 'instant'}});
            el.click();
            el.focus();
            return {{ok: true, tag: el.tagName, text: (el.innerText||el.value||el.getAttribute('aria-label')||'').substring(0,50)}};
        }})()
        """
        res = self._eval_js(js, await_promise=False, timeout=5.0)
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error", "点击失败")}
        v = res.get("value", {})
        if isinstance(v, dict) and not v.get("ok", True):
            return {"ok": False, "error": v.get("error", "点击失败")}
        time.sleep(0.5)
        self._wait_for_ready(timeout=3.0)
        tag = v.get("tag", "") if isinstance(v, dict) else ""
        text = v.get("text", "") if isinstance(v, dict) else ""
        return {"ok": True, "message": f"已点击[{index}] {tag} {text[:30]}".strip()}

    def input_text_by_index(self, index: int, text: str, clear_first: bool = True) -> dict:
        """在指定输入框输入文本。"""
        selectors = [
            'a[href]', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
            '[role="button"]', '[role="link"]', '[role="textbox"]', '[role="combobox"]',
            '[role="searchbox"]', '[role="search"]',
            '[contenteditable="true"]', '[tabindex]:not([tabindex="-1"])',
        ]
        sel_str = ','.join(selectors)
        # 先聚焦并清空，再通过Input.insertText输入
        js_focus = f"""
        (() => {{
            const els = document.querySelectorAll('{sel_str}');
            let visible = Array.from(els).filter(el => {{
                const r = el.getBoundingClientRect();
                const s = getComputedStyle(el);
                return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 0 && r.height > 0 && s.opacity !== '0';
            }});
            const seen = new Set();
            visible = visible.filter(el => {{ if(seen.has(el)) return false; seen.add(el); return true; }});
            const el = visible[{index}];
            if (!el) return {{ok: false, error: '元素不存在'}};
            el.scrollIntoView({{block: 'center', behavior: 'instant'}});
            el.focus();
            el.click();
            if ({'true' if clear_first else 'false'}) {{
                if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {{
                    el.value = '';
                }} else if (el.isContentEditable || el.getAttribute('role') === 'textbox' || el.getAttribute('role') === 'searchbox') {{
                    el.innerText = '';
                    el.textContent = '';
                }}
                // 触发input/change事件
                el.dispatchEvent(new Event('input', {{bubbles: true}}));
                el.dispatchEvent(new Event('change', {{bubbles: true}}));
            }}
            return {{ok: true, tag: el.tagName, type: el.type || el.getAttribute('role') || ''}};
        }})()
        """
        res = self._eval_js(js_focus)
        if not res.get("ok"):
            return res
        v = res.get("value", {})
        if isinstance(v, dict) and not v.get("ok", True):
            return {"ok": False, "error": v.get("error", "聚焦失败")}

        # 通过CDP Input.insertText输入（能正确触发React/Vue的受控组件）
        text = str(text).replace("\r", "")
        res = self._send_cdp("Input.insertText", {"text": text})
        if not res.get("ok"):
            # fallback: JS设置value
            escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
            self._eval_js(f"""
                const el = document.activeElement;
                if (el) {{
                    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {{
                        el.value += '{escaped}';
                    }} else if (el.isContentEditable || el.getAttribute('role') === 'textbox' || el.getAttribute('role') === 'searchbox') {{
                        el.innerText += '{escaped}';
                    }}
                    el.dispatchEvent(new Event('input', {{bubbles: true}}));
                    el.dispatchEvent(new Event('change', {{bubbles: true}}));
                }}
            """)
        time.sleep(0.3)
        return {"ok": True, "message": f"已输入「{text[:30]}」"}

    def press_key(self, key: str, vk_code: int = 0) -> dict:
        """模拟按键（按下+弹起）。key: 按键名，如'Enter'；vk_code: Windows虚拟键码。"""
        key_args = {
            "type": "keyDown",
            "key": key,
            "code": key,
            "windowsVirtualKeyCode": vk_code,
            "nativeVirtualKeyCode": vk_code,
        }
        res = self._send_cdp("Input.dispatchKeyEvent", key_args)
        if not res.get("ok"):
            return res
        key_args["type"] = "keyUp"
        self._send_cdp("Input.dispatchKeyEvent", key_args)
        return {"ok": True, "message": f"已按键 {key}"}

    def press_enter(self) -> dict:
        """按回车键（常用于提交搜索框）。"""
        return self.press_key("Enter", 13)

    def extract_text(self) -> dict:
        """提取页面正文文本。"""
        js = r"""
        (() => {
            // 尝试取main/article内容，否则取body
            const candidates = [
                document.querySelector('main'),
                document.querySelector('article'),
                document.querySelector('#content'),
                document.querySelector('.content'),
                document.body,
            ].filter(Boolean);
            let text = '';
            for (const c of candidates) {
                const t = (c.innerText || '').trim();
                if (t.length > text.length) text = t;
            }
            // 清理多余空白
            text = text.replace(/\n{3,}/g, '\n\n').substring(0, 5000);
            return {url: location.href, title: document.title, text: text, length: text.length};
        })()
        """
        res = self._eval_js(js, timeout=5.0)
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error", "提取失败")}
        return {"ok": True, "result": res.get("value", {})}

    def list_tabs_info(self) -> list[dict]:
        """返回标签页精简信息列表。"""
        tabs = self.list_tabs()
        return [
            {
                "id": t.get("id", ""),
                "title": t.get("title", ""),
                "url": t.get("url", ""),
                "active": bool(t.get("active", False)),
            }
            for t in tabs if t.get("type") == "page"
        ]

    def activate_tab(self, tab_id: str) -> dict:
        """激活（切换到）指定标签页。"""
        base = self._get_cdp_base()
        if not base:
            return {"ok": False, "error": "CDP不可用"}
        tid = quote_plus(str(tab_id), safe="")
        try:
            r = requests.get(f"{base}/json/activate/{tid}", timeout=CDP_HTTP_TIMEOUT)
            if r.ok:
                self._active_tab_id = tab_id
                return {"ok": True, "message": "已切换标签页"}
            return {"ok": False, "error": f"激活失败：HTTP {r.status_code}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def close_tab(self, tab_id: str) -> dict:
        """关闭指定标签页。"""
        base = self._get_cdp_base()
        if not base:
            return {"ok": False, "error": "CDP不可用"}
        tid = quote_plus(str(tab_id), safe="")
        try:
            requests.get(f"{base}/json/close/{tid}", timeout=CDP_HTTP_TIMEOUT)
            if tab_id in self._connections:
                try:
                    self._connections[tab_id].ws.close()
                except Exception:
                    pass
                del self._connections[tab_id]
            return {"ok": True, "message": "已关闭标签页"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def switch_to_tab_by_index(self, index: int) -> dict:
        """按序号切换标签页。"""
        tabs = self.list_tabs()
        page_tabs = [t for t in tabs if t.get("type") == "page"]
        if index < 0 or index >= len(page_tabs):
            return {"ok": False, "error": f"标签序号{index}超出范围(0-{len(page_tabs)-1})"}
        return self.activate_tab(str(page_tabs[index].get("id", "")))

    # ---------------- 安全检查 ----------------

    def is_domain_trusted(self, url: str = "") -> bool:
        """检查当前域名是否在信任列表。"""
        if not url:
            info = self.get_page_info(extract_elements=False)
            url = info.url
        from urllib.parse import urlparse
        try:
            domain = urlparse(url).netloc.lower().replace("www.", "")
        except Exception:
            return False
        for trusted in _TRUSTED_DOMAINS:
            if domain == trusted or domain.endswith("." + trusted):
                return True
        return False

    def is_dangerous_operation(self, text: str) -> tuple[bool, str]:
        """检查操作文本是否涉及高危行为。返回(是否危险, 匹配到的关键词)。"""
        for kw in _DANGEROUS_KEYWORDS:
            if kw in text:
                return True, kw
        return False, ""

    # ---------------- 连接清理 ----------------

    def disconnect_all(self) -> None:
        """关闭所有WebSocket连接。"""
        for conn in list(self._connections.values()):
            try:
                if conn.ws:
                    conn.ws.close()
            except Exception:
                pass
        self._connections.clear()
        self._active_tab_id = None


# 便捷全局函数
def get_browser_automation() -> BrowserAutomationService:
    return BrowserAutomationService.get_instance()


def elements_to_llm_context(elements: list[DOMElement], max_elements: int = 50) -> str:
    """把元素列表格式化为LLM可读的上下文字符串。"""
    if not elements:
        return "（当前页面无可交互元素，可能页面未加载完成或需要滚动）"
    lines = ["当前页面可交互元素列表（按序号定位）："]
    for el in elements[:max_elements]:
        lines.append(el.to_llm_desc())
    if len(elements) > max_elements:
        lines.append(f"...（共{len(elements)}个，仅显示前{max_elements}个）")
    return "\n".join(lines)
