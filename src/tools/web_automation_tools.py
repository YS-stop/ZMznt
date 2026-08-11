"""网页自动化工具集（LangChain BaseTool）。

8 个核心工具（基础版）：
    1. browser_navigate     - 页面导航（打开指定URL）
    2. browser_refresh      - 刷新页面
    3. browser_go_back      - 后退
    4. browser_go_forward   - 前进
    5. browser_scroll       - 页面滚动（上下翻页/顶/底）
    6. browser_list_elements - 列出页面可交互元素
    7. browser_click        - 点击指定序号的元素
    8. browser_input        - 在输入框输入文本
    9. browser_extract_text - 提取页面正文文本
    10. browser_list_tabs   - 列出/切换标签页

视觉辅助定位：
    - 当DOM语义定位不确定时，截图调用 GLM-4.1V 视觉模型辅助定位图标/图片按钮
"""
from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel, Field

# 确保 import src.*
_SRC_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from langchain_core.tools import BaseTool  # noqa: E402

from src.services.browser_automation_service import (  # noqa: E402
    DOMElement,
    get_browser_automation,
    elements_to_llm_context,
)


# ============================================================
# 视觉模型辅助定位（GLM-4.1V）
# ============================================================

def _call_vl_model(prompt: str, image_bytes: bytes | None = None) -> Optional[str]:
    """调用视觉模型（优先智谱 GLM-4.1V，失败回退通义千问 Qwen-VL），返回模型输出文本；失败返回None。"""
    import requests as _req

    # 构造消息内容
    content: list[dict] = []
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    content.append({"type": "text", "text": prompt})

    def _do_call(api_key: str, base_url: str, model: str) -> Optional[str]:
        if not api_key:
            return None
        try:
            r = _req.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 500,
                    "temperature": 0.1,
                },
                timeout=30,
            )
            if r.status_code != 200:
                return None
            data = r.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        except Exception:
            return None

    # 1) 优先智谱 GLM-4.1V
    glm_key = os.getenv("GLM_API_KEY", "").strip()
    glm_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/").strip()
    glm_model = os.getenv("GLM_VL_MODEL", "glm-4.1v-flash").strip()
    result = _do_call(glm_key, glm_url, glm_model)
    if result:
        return result

    # 2) 回退：通义千问 Qwen-VL
    qwen_key = os.getenv("QWEN_API_KEY", "").strip()
    qwen_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
    return _do_call(qwen_key, qwen_url, "qwen-vl-max")


def vl_find_element_index(user_desc: str, elements: list[DOMElement], page_title: str = "") -> Optional[int]:
    """用视觉模型+DOM列表定位元素序号。返回元素index，失败返回None。"""
    if not elements:
        return None
    # 1) 先用LLM文本匹配（不依赖网络，速度快）
    idx = _text_match_element(user_desc, elements)
    if idx is not None:
        return idx

    # 2) 视觉辅助：截图+DOM上下文一起发给VL
    try:
        svc = get_browser_automation()
        img = svc.capture_screenshot()
    except Exception:
        img = None

    if img is None:
        return None

    # 构造元素列表文本
    elem_desc_lines = []
    for el in elements[:50]:
        rect = el.rect
        pos = f"({int(rect.get('x', 0))},{int(rect.get('y', 0))})" if rect else "?"
        text = (el.text or el.aria_label or el.placeholder or el.element_id or "")[:30]
        elem_desc_lines.append(f"序号{el.index}: <{el.tag}> 文本='{text}' 位置={pos}")
    elem_text = "\n".join(elem_desc_lines)

    prompt = f"""页面标题：{page_title}
用户想点击的元素描述：{user_desc}

以下是页面可交互元素列表（含坐标位置）：
{elem_text}

请根据截图和元素列表，判断用户描述对应的是哪个序号的元素。
只返回一个数字（元素序号），不要任何解释。如果完全不确定，返回-1。"""

    result = _call_vl_model(prompt, img)
    if not result:
        return None
    # 解析数字
    import re
    m = re.search(r'(-?\d+)', result)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(elements):
            return idx
    return None


def _text_match_element(user_desc: str, elements: list[DOMElement]) -> Optional[int]:
    """简单文本匹配定位（不联网，快速路径）。"""
    desc = user_desc.strip()
    if not desc:
        return None

    # 精确序号匹配："第3个"、"第三个"、"序号5"、"[3]"、"3号"
    import re
    num_match = re.search(r'(?:第|序号|\[|#)?\s*(\d{1,2})\s*(?:个|号|\])', desc)
    if num_match:
        idx = int(num_match.group(1))
        # 注意：用户说"第一个"通常对应index=0，但也可能对应index=1
        # 做兼容：如果是"第N个"，尝试N-1和N-1
        if idx == 0:
            idx = 0
        else:
            idx = idx - 1 if idx > 0 else 0
        for el in elements:
            if el.index == idx:
                return idx
        # 直接按列表顺序
        if 0 <= idx < len(elements):
            return elements[idx].index

    # 关键词匹配：找文本/aria/placeholer包含用户描述的元素
    desc_lower = desc.lower()
    # 去除常见修饰词
    for stop in ("点击", "点一下", "点", "那个", "这个", "按钮", "链接", "图标", "一下", "帮我"):
        desc_lower = desc_lower.replace(stop, "")
    desc_lower = desc_lower.strip()
    if not desc_lower:
        return None

    # 精确包含
    candidates = []
    for el in elements:
        haystack = f"{el.text} {el.aria_label} {el.placeholder} {el.element_id} {el.class_name}".lower()
        if desc_lower in haystack:
            # 优先文本完全匹配、可见、启用的
            score = 0
            if el.text.lower() == desc_lower:
                score += 10
            if desc_lower in el.text.lower():
                score += 5
            if el.is_visible:
                score += 2
            if el.is_enabled:
                score += 1
            candidates.append((score, el.index))

    if candidates:
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][1]

    # 模糊：分词匹配
    keywords = [k for k in re.split(r'[\s,，。、]+', desc_lower) if len(k) >= 1]
    if keywords:
        scored = []
        for el in elements:
            haystack = f"{el.text} {el.aria_label} {el.placeholder}".lower()
            hit = sum(1 for k in keywords if k in haystack)
            if hit > 0:
                scored.append((hit, el.is_visible, el.is_enabled, el.index))
        if scored:
            scored.sort(reverse=True)
            return scored[0][3]

    return None


# ============================================================
# 工具实现
# ============================================================

def _check_cdp_available(max_wait_s: float = 3.0) -> Optional[str]:
    """检查CDP是否可用，返回错误信息；可用返回None。

    如果刚打开浏览器，会自动等待CDP就绪（最多max_wait_s秒），避免Agent立即调用失败。
    """
    svc = get_browser_automation()
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        if svc.is_available():
            return None
        time.sleep(0.3)
    return (
        "⚠️ 当前没有检测到带调试端口的浏览器实例。\n"
        "如果刚调用过open_browser，可能浏览器还在启动中，请等待2秒后重试当前操作（不要再调用open_browser）。\n"
        "如果还没打开任何网站，才需要调用一次open_browser(target=网站名)打开目标站点。"
    )


class BrowserNavigateArgs(BaseModel):
    url: str = Field(..., min_length=1, description="要导航到的URL或网站名称（如'百度'、'zhihu.com'、'https://xxx'）")


class BrowserNavigateTool(BaseTool):
    name: ClassVar[str] = "browser_navigate"
    description: ClassVar[str] = (
        "Tool Name: browser_navigate\n"
        "用途：在当前浏览器标签页中导航到指定网址。\n"
        "参数url支持：完整URL(https://...)、域名(zhihu.com)、或快捷站点名(百度/知乎/B站/淘宝/GitHub等)。\n"
        "注意：必须先通过open_browser打开过浏览器，才能使用此工具。\n"
        "示例：browser_navigate(url='zhihu.com') / browser_navigate(url='https://www.bilibili.com')"
    )
    args_schema: ClassVar[type[BaseModel]] = BrowserNavigateArgs
    return_direct: ClassVar[bool] = False

    def _run(self, url: str) -> str:
        err = _check_cdp_available()
        if err:
            return err
        # 复用browser_tools的URL解析
        try:
            from src.tools.browser_tools import resolve_target_to_url
            final_url, method = resolve_target_to_url(url)
        except Exception as e:
            return f"❌ URL解析失败：{e}"
        svc = get_browser_automation()
        res = svc.navigate(final_url)
        if res.get("ok"):
            info = svc.get_page_info(extract_elements=False)
            return f"✅ 已导航到：{info.title or final_url}\n  URL：{info.url}\n  解析方式：{method}"
        return f"❌ 导航失败：{res.get('error', '未知错误')}"


class BrowserRefreshTool(BaseTool):
    name: ClassVar[str] = "browser_refresh"
    description: ClassVar[str] = (
        "Tool Name: browser_refresh\n"
        "用途：刷新当前浏览器页面（等同按F5）。无参数。\n"
        "注意：必须先通过open_browser打开过浏览器。"
    )
    args_schema: ClassVar[type[BaseModel]] = None
    return_direct: ClassVar[bool] = False

    def _run(self) -> str:
        err = _check_cdp_available()
        if err:
            return err
        svc = get_browser_automation()
        res = svc.page_reload()
        if res.get("ok"):
            time.sleep(1.0)
            svc._wait_for_ready(timeout=5.0)
            info = svc.get_page_info(extract_elements=False)
            return f"✅ 页面已刷新：{info.title or info.url}"
        return f"❌ 刷新失败：{res.get('error', '未知错误')}"


class BrowserGoBackTool(BaseTool):
    name: ClassVar[str] = "browser_go_back"
    description: ClassVar[str] = (
        "Tool Name: browser_go_back\n"
        "用途：浏览器后退一页（等同浏览器后退按钮）。无参数。"
    )
    args_schema: ClassVar[type[BaseModel]] = None
    return_direct: ClassVar[bool] = False

    def _run(self) -> str:
        err = _check_cdp_available()
        if err:
            return err
        svc = get_browser_automation()
        res = svc.page_history(-1)
        if res.get("ok"):
            info = svc.get_page_info(extract_elements=False)
            return f"✅ 已后退：{info.title or info.url}"
        return f"❌ 后退失败：{res.get('error', '未知错误')}"


class BrowserGoForwardTool(BaseTool):
    name: ClassVar[str] = "browser_go_forward"
    description: ClassVar[str] = (
        "Tool Name: browser_go_forward\n"
        "用途：浏览器前进一页（等同浏览器前进按钮）。无参数。"
    )
    args_schema: ClassVar[type[BaseModel]] = None
    return_direct: ClassVar[bool] = False

    def _run(self) -> str:
        err = _check_cdp_available()
        if err:
            return err
        svc = get_browser_automation()
        res = svc.page_history(1)
        if res.get("ok"):
            info = svc.get_page_info(extract_elements=False)
            return f"✅ 已前进：{info.title or info.url}"
        return f"❌ 前进失败：{res.get('error', '未知错误')}"


class BrowserScrollArgs(BaseModel):
    direction: str = Field(
        "down",
        description="滚动方向：down(向下翻页)/up(向上翻页)/top(回到顶部)/bottom(滚到底部)，默认down"
    )


class BrowserScrollTool(BaseTool):
    name: ClassVar[str] = "browser_scroll"
    description: ClassVar[str] = (
        "Tool Name: browser_scroll\n"
        "用途：滚动当前页面。\n"
        "参数direction可选：down(下翻一页，默认)/up(上翻一页)/top(顶部)/bottom(底部)。\n"
        "示例：'往下翻'→direction='down'；'回顶部'→direction='top'"
    )
    args_schema: ClassVar[type[BaseModel]] = BrowserScrollArgs
    return_direct: ClassVar[bool] = False

    def _run(self, direction: str = "down") -> str:
        err = _check_cdp_available()
        if err:
            return err
        d = (direction or "down").strip().lower()
        if d not in ("down", "up", "top", "bottom"):
            d = "down"
        svc = get_browser_automation()
        res = svc.scroll_page(d)
        if res.get("ok"):
            return f"✅ {res.get('message', '已滚动')}"
        return f"❌ 滚动失败：{res.get('error', '未知错误')}"


class BrowserListElementsTool(BaseTool):
    name: ClassVar[str] = "browser_list_elements"
    description: ClassVar[str] = (
        "Tool Name: browser_list_elements\n"
        "用途：列出当前页面所有可交互元素（按钮、链接、输入框等）及其序号。\n"
        "在用户说'页面上有什么'、'我能点什么'、'找不到XX按钮'时调用，"
        "返回元素列表后根据序号进行browser_click/browser_input操作。"
    )
    args_schema: ClassVar[type[BaseModel]] = None
    return_direct: ClassVar[bool] = False

    def _run(self) -> str:
        err = _check_cdp_available()
        if err:
            return err
        svc = get_browser_automation()
        info = svc.get_page_info(extract_elements=True, max_elements=60)
        if not info.elements:
            return (
                f"📄 当前页面：{info.title or '无标题'}\n"
                f"  URL：{info.url}\n"
                f"  未检测到可交互元素（可能页面未加载完成，或需要先滚动页面）。"
            )
        context = elements_to_llm_context(info.elements, max_elements=60)
        return (
            f"📄 当前页面：{info.title or '无标题'}\n"
            f"  URL：{info.url}\n"
            f"  共 {len(info.elements)} 个可交互元素\n\n"
            f"{context}\n\n"
            f"提示：根据序号调用 browser_click(element_index=N) 或 browser_input(element_index=N, text='xxx')"
        )


class BrowserClickArgs(BaseModel):
    element_index: int = Field(..., ge=0, description="要点击的元素序号（从browser_list_elements获取）")
    element_desc: str = Field("", description="用户对元素的自然语言描述（如'搜索按钮'、'蓝色登录按钮'），用于视觉辅助定位")


class BrowserClickTool(BaseTool):
    name: ClassVar[str] = "browser_click"
    description: ClassVar[str] = (
        "Tool Name: browser_click\n"
        "用途：点击页面上的元素。\n"
        "参数：element_index是元素序号（**强烈建议**先调用browser_list_elements获取序号再点击）；\n"
        "element_desc是用户的自然语言描述（可选，用于序号不准时视觉辅助定位）。\n"
        "注意：涉及提交订单/支付/删除/密码等高危操作会被拦截并请求用户确认。"
    )
    args_schema: ClassVar[type[BaseModel]] = BrowserClickArgs
    return_direct: ClassVar[bool] = False

    def _run(self, element_index: int, element_desc: str = "") -> str:
        err = _check_cdp_available()
        if err:
            return err
        svc = get_browser_automation()
        info = svc.get_page_info(extract_elements=True)

        # 如果传了desc但index不准，尝试文本/视觉定位
        target_idx = element_index
        if element_desc and element_desc.strip():
            found = vl_find_element_index(element_desc, info.elements, info.title)
            if found is not None:
                target_idx = found

        # 检查是否越界
        valid_indices = {el.index for el in info.elements}
        if target_idx not in valid_indices:
            # 尝试按列表位置取
            if 0 <= target_idx < len(info.elements):
                target_idx = info.elements[target_idx].index
            else:
                return (
                    f"⚠️ 序号{element_index}超出范围（当前页面共{len(info.elements)}个可交互元素，序号0-{len(info.elements)-1}）。\n"
                    f"请先调用 browser_list_elements 查看可用元素序号。"
                )

        # 高危操作检查
        target_el = next((e for e in info.elements if e.index == target_idx), None)
        if target_el:
            check_text = f"{target_el.text} {target_el.aria_label}"
            is_danger, kw = svc.is_dangerous_operation(check_text)
            if is_danger:
                return (
                    f"⚠️ 检测到高危操作（关键词：{kw}）！\n"
                    f"目标元素：[{target_idx}] {target_el.text[:50]}\n"
                    f"请确认后回复「确认点击{target_idx}」再执行，避免误操作。"
                )

        res = svc.click_element_by_index(target_idx)
        if res.get("ok"):
            time.sleep(1.0)
            svc._wait_for_ready(timeout=4.0)
            new_info = svc.get_page_info(extract_elements=False)
            return f"✅ {res.get('message', '点击成功')}\n  当前页面：{new_info.title or new_info.url}"
        return f"❌ 点击失败：{res.get('error', '未知错误')}"


class BrowserInputArgs(BaseModel):
    element_index: int = Field(-1, ge=-1, description="输入框序号（从browser_list_elements获取），-1表示自动查找搜索框")
    text: str = Field(..., min_length=1, description="要输入的文本内容")
    submit: bool = Field(False, description="输入后是否自动按回车提交（如搜索框输入后直接搜索）")
    element_desc: str = Field("", description="输入框的自然语言描述（如'搜索框'、'用户名输入框'）")


class BrowserInputTool(BaseTool):
    name: ClassVar[str] = "browser_input"
    description: ClassVar[str] = (
        "Tool Name: browser_input\n"
        "用途：在页面输入框中填写文本。\n"
        "参数：\n"
        "  - text【必填】：要输入的内容\n"
        "  - element_index：输入框序号（建议先list获取；不传时自动找页面上第一个文本/搜索输入框）\n"
        "  - submit：输入后是否自动按回车提交（默认False；搜索场景建议True）\n"
        "  - element_desc：输入框描述（辅助定位）\n"
        "示例：'在搜索框输入今天天气然后搜索'→text='今天天气',submit=True"
    )
    args_schema: ClassVar[type[BaseModel]] = BrowserInputArgs
    return_direct: ClassVar[bool] = False

    def _run(self, text: str, element_index: int = -1, submit: bool = False, element_desc: str = "") -> str:
        err = _check_cdp_available()
        if err:
            return err
        svc = get_browser_automation()
        info = svc.get_page_info(extract_elements=True)

        target_idx = element_index
        # 自动查找输入框
        if target_idx < 0 or element_desc:
            if element_desc:
                found = vl_find_element_index(element_desc or text, info.elements, info.title)
                if found is not None:
                    target_idx = found
            if target_idx < 0:
                # 智能查找输入框：按优先级打分
                # 1) 最高优先：type=search 或 role=searchbox
                # 2) 高优先：placeholder/aria/text 含"搜索"关键词
                # 3) 普通：可见的 text/email/url/tel 输入框 或 role=textbox
                # 4) 兜底：任意可见的 input/textarea/contenteditable
                best = None
                best_score = 0
                for el in info.elements:
                    score = 0
                    haystack = f"{el.text} {el.aria_label} {el.placeholder} {el.element_id} {el.class_name} {el.role}".lower()

                    # 类型得分
                    if el.element_type == "search" or el.role == "searchbox":
                        score += 100
                    if "搜索" in haystack or "search" in haystack or "搜" in haystack:
                        score += 80
                    if el.tag in ("input", "textarea") and el.element_type in ("text", "search", "email", "url", "tel", "", None):
                        score += 30
                    if el.role == "textbox" or el.role == "combobox":
                        score += 25
                    if el.tag in ("input", "textarea") and el.element_type not in ("checkbox", "radio", "password", "hidden", "submit", "button", "file"):
                        score += 10
                    if el.tag in ("input", "textarea"):
                        score += 5

                    # 可见/可用加分
                    if el.is_visible:
                        score += 3
                    if el.is_enabled:
                        score += 2

                    # 位置加分（搜索框通常在页面顶部）
                    if el.rect and el.rect.get("y", 1000) < 200:
                        score += 2

                    if score > best_score:
                        best_score = score
                        best = el
                if best is not None:
                    target_idx = best.index

        if target_idx < 0:
            return "⚠️ 未找到可用的输入框，请先调用 browser_list_elements 确认输入框序号。"

        # 检查是否是密码框等敏感输入
        target_el = next((e for e in info.elements if e.index == target_idx), None)
        if target_el and target_el.element_type == "password":
            return "⚠️ 禁止自动输入密码！为了账号安全，密码请手动输入。"

        # 域名信任检查（非信任域名提示但不阻止）
        if not svc.is_domain_trusted(info.url):
            domain_note = f"（注意：当前域名不在常用站点白名单，已仅执行输入操作）"
        else:
            domain_note = ""

        res = svc.input_text_by_index(target_idx, text, clear_first=True)
        if not res.get("ok"):
            return f"❌ 输入失败：{res.get('error', '未知错误')}"

        # 自动按回车
        if submit:
            time.sleep(0.3)
            svc.press_enter()
            time.sleep(1.0)
            svc._wait_for_ready(timeout=5.0)
            new_info = svc.get_page_info(extract_elements=False)
            return f"✅ 已输入「{text[:30]}」并提交回车\n  当前页面：{new_info.title or new_info.url} {domain_note}"

        return f"✅ {res.get('message', '输入成功')}{domain_note}"


class BrowserExtractTextTool(BaseTool):
    name: ClassVar[str] = "browser_extract_text"
    description: ClassVar[str] = (
        "Tool Name: browser_extract_text\n"
        "用途：提取当前页面的正文文本内容（自动找main/article区域，最长5000字）。\n"
        "在用户说'读一下页面内容'、'页面上说什么'、'提取文章内容'时调用。\n"
        "返回文本后由LLM总结后再回复用户，不要直接原样输出长文本。"
    )
    args_schema: ClassVar[type[BaseModel]] = None
    return_direct: ClassVar[bool] = False

    def _run(self) -> str:
        err = _check_cdp_available()
        if err:
            return err
        svc = get_browser_automation()
        res = svc.extract_text()
        if not res.get("ok"):
            return f"❌ 文本提取失败：{res.get('error', '未知错误')}"
        data = res.get("result", {})
        text = data.get("text", "")
        title = data.get("title", "")
        url = data.get("url", "")
        length = data.get("length", 0)
        if not text:
            return f"📄 {title}\n  URL：{url}\n  页面正文为空"
        return f"📄 页面：{title}\n  URL：{url}\n  正文长度：{length}字\n\n{text}"


class BrowserListTabsArgs(BaseModel):
    action: str = Field("list", description="操作类型：list(列出标签)/switch(切换标签)")
    tab_index: int = Field(-1, description="切换目标标签序号（action=switch时必填，从0开始）")


class BrowserListTabsTool(BaseTool):
    name: ClassVar[str] = "browser_list_tabs"
    description: ClassVar[str] = (
        "Tool Name: browser_list_tabs\n"
        "用途：列出当前浏览器所有标签页，或切换到指定序号的标签页。\n"
        "参数：\n"
        "  - action='list'（默认）：列出所有标签页及序号\n"
        "  - action='switch', tab_index=N：切换到第N个标签页\n"
        "示例：'现在开了几个标签'→action='list'；'切到第二个标签'→action='switch',tab_index=1"
    )
    args_schema: ClassVar[type[BaseModel]] = BrowserListTabsArgs
    return_direct: ClassVar[bool] = False

    def _run(self, action: str = "list", tab_index: int = -1) -> str:
        err = _check_cdp_available()
        if err:
            return err
        svc = get_browser_automation()
        tabs = svc.list_tabs_info()
        if not tabs:
            return "📑 当前没有打开的标签页。"

        if (action or "list").strip().lower() == "switch" and tab_index >= 0:
            res = svc.switch_to_tab_by_index(tab_index)
            if res.get("ok"):
                info = svc.get_page_info(extract_elements=False)
                return f"✅ 已切换到标签{tab_index}：{info.title or info.url}"
            return f"❌ 切换失败：{res.get('error', '未知错误')}"

        lines = [f"📑 当前共 {len(tabs)} 个标签页："]
        for i, t in enumerate(tabs):
            mark = "🟢" if t.get("active") else "  "
            title = t.get("title") or "(无标题)"
            url = t.get("url", "")
            if len(url) > 60:
                url = url[:57] + "..."
            lines.append(f"  {mark} [{i}] {title[:50]}\n      {url}")
        lines.append("\n提示：browser_list_tabs(action='switch', tab_index=N) 切换标签")
        return "\n".join(lines)


# ============================================================
# 工具导出
# ============================================================

def get_web_automation_tools() -> list[BaseTool]:
    """返回所有网页自动化工具实例。"""
    return [
        BrowserNavigateTool(),
        BrowserRefreshTool(),
        BrowserGoBackTool(),
        BrowserGoForwardTool(),
        BrowserScrollTool(),
        BrowserListElementsTool(),
        BrowserClickTool(),
        BrowserInputTool(),
        BrowserExtractTextTool(),
        BrowserListTabsTool(),
    ]


__all__ = [
    "get_web_automation_tools",
    "BrowserNavigateTool",
    "BrowserRefreshTool",
    "BrowserGoBackTool",
    "BrowserGoForwardTool",
    "BrowserScrollTool",
    "BrowserListElementsTool",
    "BrowserClickTool",
    "BrowserInputTool",
    "BrowserExtractTextTool",
    "BrowserListTabsTool",
]
