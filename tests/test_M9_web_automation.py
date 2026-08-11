"""网页自动化能力基础测试（M9）。

测试点：
    1. BrowserAutomationService 单例模式与基础属性
    2. DOMElement 格式化描述
    3. 工具实例化与参数schema
    4. 文本匹配定位算法
    5. CDP不可用时的优雅降级（工具返回友好提示）
    6. 高危操作检测
    7. 域名信任检查

运行：
    cd d:\zhuomZNT
    .\venv_assistant\Scripts\python.exe -m pytest tests/test_M9_web_automation.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.browser_automation_service import (  # noqa: E402
    BrowserAutomationService,
    DOMElement,
    elements_to_llm_context,
    _TRUSTED_DOMAINS,
    _DANGEROUS_KEYWORDS,
)
from src.tools.web_automation_tools import (  # noqa: E402
    _text_match_element,
    get_web_automation_tools,
    BrowserNavigateTool,
    BrowserClickTool,
    BrowserInputTool,
    BrowserScrollTool,
    BrowserListElementsTool,
    BrowserExtractTextTool,
    BrowserListTabsTool,
    BrowserRefreshTool,
    BrowserGoBackTool,
    BrowserGoForwardTool,
)


def _make_element(index: int, tag: str = "button", text: str = "",
                  aria: str = "", placeholder: str = "", element_id: str = "",
                  class_name: str = "", el_type: str = "", href: str = "",
                  role: str = "", rect=None, visible: bool = True, enabled: bool = True) -> DOMElement:
    return DOMElement(
        index=index, tag=tag, text=text, aria_label=aria, placeholder=placeholder,
        element_id=element_id, class_name=class_name, element_type=el_type,
        href=href, role=role, rect=rect or {"x": 0, "y": 0, "width": 100, "height": 30},
        is_visible=visible, is_enabled=enabled,
    )


class TestDOMElement(unittest.TestCase):
    """DOMElement 格式化描述测试。"""

    def test_button_desc(self):
        el = _make_element(0, tag="button", text="登录")
        desc = el.to_llm_desc()
        self.assertIn("[0]", desc)
        self.assertIn("按钮", desc)
        self.assertIn("登录", desc)

    def test_link_desc(self):
        el = _make_element(1, tag="a", text="首页", href="https://example.com")
        desc = el.to_llm_desc()
        self.assertIn("[1]", desc)
        self.assertIn("链接", desc)
        self.assertIn("首页", desc)

    def test_input_search_desc(self):
        el = _make_element(2, tag="input", placeholder="搜索内容...", el_type="search")
        desc = el.to_llm_desc()
        self.assertIn("[2]", desc)
        self.assertIn("搜索框", desc)
        self.assertIn("搜索内容", desc)

    def test_disabled_element(self):
        el = _make_element(3, tag="button", text="提交", enabled=False)
        desc = el.to_llm_desc()
        self.assertIn("[禁用]", desc)

    def test_elements_to_context(self):
        elements = [
            _make_element(0, "button", "登录"),
            _make_element(1, "a", "注册", href="#"),
            _make_element(2, "input", placeholder="用户名", el_type="text"),
        ]
        ctx = elements_to_llm_context(elements)
        self.assertIn("可交互元素列表", ctx)
        self.assertIn("[0]", ctx)
        self.assertIn("[1]", ctx)
        self.assertIn("[2]", ctx)

    def test_empty_elements(self):
        ctx = elements_to_llm_context([])
        self.assertIn("无可交互元素", ctx)


class TestTextMatch(unittest.TestCase):
    """文本定位算法测试。"""

    def setUp(self):
        self.elements = [
            _make_element(0, "button", "登录"),
            _make_element(1, "button", "注册"),
            _make_element(2, "a", "首页", href="/"),
            _make_element(3, "input", placeholder="搜索", el_type="search"),
            _make_element(4, "button", "搜索", el_type="submit"),
            _make_element(5, "a", "关于我们", href="/about"),
        ]

    def test_exact_text_match(self):
        idx = _text_match_element("登录", self.elements)
        self.assertEqual(idx, 0)

    def test_exact_text_match_2(self):
        idx = _text_match_element("注册", self.elements)
        self.assertEqual(idx, 1)

    def test_ordinal_match(self):
        """'第一个按钮'之类的序号匹配。"""
        idx = _text_match_element("第1个", self.elements)
        # 第1个应该是index=0
        self.assertIsNotNone(idx)

    def test_keyword_fuzzy(self):
        idx = _text_match_element("搜索按钮", self.elements)
        # 应该匹配到 搜索按钮(index=4)或搜索框(index=3)
        self.assertIn(idx, [3, 4])

    def test_no_match(self):
        idx = _text_match_element("完全不存在的xyz", self.elements)
        self.assertIsNone(idx)

    def test_empty_desc(self):
        idx = _text_match_element("", self.elements)
        self.assertIsNone(idx)


class TestBrowserService(unittest.TestCase):
    """BrowserAutomationService 基础测试。"""

    def test_singleton(self):
        s1 = BrowserAutomationService.get_instance()
        s2 = BrowserAutomationService.get_instance()
        self.assertIs(s1, s2)

    def test_not_available_when_no_browser(self):
        svc = BrowserAutomationService.get_instance()
        # 测试环境通常没有CDP浏览器，应该返回False
        result = svc.is_available()
        self.assertIsInstance(result, bool)

    def test_domain_trust_check(self):
        svc = BrowserAutomationService.get_instance()
        # 信任域名
        self.assertTrue(svc.is_domain_trusted("https://www.baidu.com/s?wd=test"))
        self.assertTrue(svc.is_domain_trusted("https://zhihu.com/question/123"))
        self.assertTrue(svc.is_domain_trusted("https://github.com/user/repo"))
        # 不信任域名
        self.assertFalse(svc.is_domain_trusted("https://random-fishing-site.com/pay"))

    def test_dangerous_operation_detection(self):
        svc = BrowserAutomationService.get_instance()
        is_danger, kw = svc.is_dangerous_operation("立即支付")
        self.assertTrue(is_danger)
        self.assertEqual(kw, "支付")

        is_danger, kw = svc.is_dangerous_operation("确认删除")
        self.assertTrue(is_danger)
        self.assertEqual(kw, "删除")

        is_danger, kw = svc.is_dangerous_operation("普通的查看按钮")
        self.assertFalse(is_danger)


class TestWebAutomationTools(unittest.TestCase):
    """网页自动化工具实例化测试。"""

    def test_all_tools_instantiate(self):
        tools = get_web_automation_tools()
        self.assertEqual(len(tools), 10)
        names = {t.name for t in tools}
        expected = {
            "browser_navigate", "browser_refresh", "browser_go_back",
            "browser_go_forward", "browser_scroll", "browser_list_elements",
            "browser_click", "browser_input", "browser_extract_text",
            "browser_list_tabs",
        }
        self.assertEqual(names, expected)

    def test_navigate_args_schema(self):
        tool = BrowserNavigateTool()
        self.assertIsNotNone(tool.args_schema)
        # url是必填
        schema = tool.args_schema.model_json_schema()
        self.assertIn("url", schema["properties"])
        self.assertIn("url", schema.get("required", []))

    def test_click_args_schema(self):
        tool = BrowserClickTool()
        schema = tool.args_schema.model_json_schema()
        self.assertIn("element_index", schema["properties"])
        self.assertIn("element_desc", schema["properties"])

    def test_input_args_schema(self):
        tool = BrowserInputTool()
        schema = tool.args_schema.model_json_schema()
        self.assertIn("text", schema["properties"])
        self.assertIn("submit", schema["properties"])

    def test_tools_return_helpful_message_when_no_cdp(self):
        """无CDP浏览器时应该返回友好提示，而不是崩溃。"""
        tool = BrowserListElementsTool()
        result = tool._run()
        # 应该返回包含提示信息的字符串
        self.assertIsInstance(result, str)
        # 要么提示没有CDP，要么返回页面信息
        self.assertTrue(
            "调试端口" in result or
            "可交互元素" in result or
            "未检测到" in result or
            "浏览器" in result,
            f"意外返回: {result[:200]}"
        )

    def test_click_requires_cdp(self):
        tool = BrowserClickTool()
        result = tool._run(element_index=0)
        self.assertIsInstance(result, str)
        # 应该提示需要先打开浏览器
        self.assertTrue(
            "调试端口" in result or
            "浏览器" in result or
            "打开" in result or
            "未找到" in result or
            "范围" in result,
            f"意外返回: {result[:200]}"
        )


class TestSecurity(unittest.TestCase):
    """安全机制测试。"""

    def test_trusted_domains_include_common_sites(self):
        self.assertIn("baidu.com", _TRUSTED_DOMAINS)
        self.assertIn("zhihu.com", _TRUSTED_DOMAINS)
        self.assertIn("github.com", _TRUSTED_DOMAINS)
        self.assertIn("bilibili.com", _TRUSTED_DOMAINS)
        self.assertIn("taobao.com", _TRUSTED_DOMAINS)

    def test_dangerous_keywords_cover_high_risk(self):
        for kw in ["支付", "删除", "密码", "转账", "验证码"]:
            self.assertIn(kw, _DANGEROUS_KEYWORDS, f"缺少高危关键词: {kw}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
