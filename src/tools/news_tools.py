"""新闻资讯搜索工具（LangChain BaseTool 子类）。

功能：
    * SearchNewsTool：按关键词搜索最新资讯（百度/必应双引擎优先，失败降级 mock），
      返回结构化结果列表（标题 / 摘要 / 来源 / 时间 / URL），默认 10 条。

设计说明：
    1. 不调用 LLM 做总结，仅「返回结构化原文搜索结果」（M2 占位版，M4 可接入 LLM 总结）。
    2. 双引擎：先尝试必应公开 JSON API，失败回退百度 RSS，再失败返回 mock 数据确保离线可测。
    3. 所有异常包装为 observation 返回，不向上抛。
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote_plus

from pydantic import BaseModel, Field

# 确保 import src.* 成功
_SRC_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from langchain_core.tools import BaseTool  # noqa: E402


# ============================================================
# 参数 Schema
# ============================================================

class SearchNewsArgs(BaseModel):
    """SearchNewsTool 参数 Schema（全中文描述，给 LLM 看）。"""

    query: str = Field(
        ...,
        min_length=1,
        description="【必填】新闻搜索关键词，例如「2026 年 AI 大模型最新进展」「A股今日行情」。",
    )
    engine: str = Field(
        "auto",
        description=(
            "【选填】搜索引擎：auto（默认，自动尝试多引擎取最快）、"
            "bing（必应）、baidu（百度）、mock（离线模拟数据）。"
        ),
    )
    max_results: int = Field(
        10,
        ge=1,
        le=30,
        description="【选填】返回条数上限（1~30，默认 10）。",
    )
    hours: int = Field(
        0,
        ge=0,
        le=24 * 30,
        description=(
            "【选填】时间范围：0=不限（默认）；"
            "正数=最近 N 小时内新闻（例如 24=最近一天、168=最近一周）。"
        ),
    )


# ============================================================
# 内部：结果格式化辅助
# ============================================================

def _ts_to_str(ts: float | None) -> str:
    """时间戳 → 可读字符串，None 或 0 → 空。"""
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:  # noqa: BLE001
        return ""


def _clean_text(s: str, max_len: int = 300) -> str:
    """去除多余空白，截断到 max_len 字符（避免返回过长）。"""
    if not s:
        return ""
    s = " ".join(s.split())
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def _format_news(items: list[dict[str, Any]], query: str, engine_used: str, elapsed_ms: int) -> str:
    """把结构化 items 格式化成 observation 字符串（中文 + 编号 + 对齐）。"""
    lines: list[str] = []
    lines.append(
        f"📰 search_news 完成 | 关键词=[{query}] | 引擎={engine_used} | "
        f"命中 {len(items)} 条 | 耗时 {elapsed_ms} ms"
    )
    if not items:
        lines.append("  （未搜索到任何匹配新闻，可尝试更换关键词或扩大时间范围）")
        return "\n".join(lines)
    lines.append(f"  Top {len(items)}（按引擎返回顺序）：")
    for i, it in enumerate(items, start=1):
        title = _clean_text(it.get("title", ""), max_len=120) or "（无标题）"
        summary = _clean_text(it.get("summary", ""), max_len=260)
        source = it.get("source", "")
        pub_time = it.get("pub_time", "")
        url = it.get("url", "")
        head = f"    [{i:>2}] 📰 {title}"
        if source or pub_time:
            head += f"    【{source} {pub_time}】".rstrip()
        lines.append(head)
        if summary:
            lines.append(f"         📝 {summary}")
        if url:
            lines.append(f"         🔗 {url}")
    return "\n".join(lines)


# ============================================================
# 内部：各搜索引擎实现
# ============================================================

def _mock_news(query: str, max_results: int) -> list[dict[str, Any]]:
    """离线 / 网络失败时的 Mock 数据（保证测试、演示可用）。"""
    now = time.time()
    base: list[dict[str, Any]] = [
        {
            "title": f"「{query}」成为行业热点，多家头部企业发布相关战略规划",
            "summary": (
                f"近期围绕「{query}」的讨论持续升温。分析师指出，该领域在技术突破、政策支持、"
                "市场需求三重驱动下，预计未来三年复合增长率将超过 35%。头部企业已纷纷布局，"
                "抢占技术制高点和市场份额。"
            ),
            "source": "经济观察报",
            "pub_time": _ts_to_str(now - 3600 * 2),
            "url": f"https://example.com/news/mock?q={quote_plus(query)}&n=1",
        },
        {
            "title": f"专家深度解读：{query}背后的技术原理与未来趋势",
            "summary": (
                "业内专家在接受专访时表示，该技术路线的核心突破在于算法效率与工程化落地能力的平衡。"
                "展望未来 12 个月，预计将出现更多面向垂直行业的应用场景，推动价值释放。"
            ),
            "source": "科技日报",
            "pub_time": _ts_to_str(now - 3600 * 6),
            "url": f"https://example.com/news/mock?q={quote_plus(query)}&n=2",
        },
        {
            "title": f"{query}相关概念股早盘走强，机构建议关注三条主线",
            "summary": (
                "今日早盘 A 股相关板块异动拉升。券商研报指出，建议关注：① 核心技术提供方；"
                "② 数据与算力基础设施；③ 下游应用落地较快的公司。需注意短期情绪过热后的回调风险。"
            ),
            "source": "证券时报",
            "pub_time": _ts_to_str(now - 3600 * 10),
            "url": f"https://example.com/news/mock?q={quote_plus(query)}&n=3",
        },
        {
            "title": f"全球首个「{query}」行业标准正式立项，多家单位参与起草",
            "summary": (
                "标准化组织日前宣布正式立项相关标准，首批 20 余家产学研用单位共同参与起草。"
                "标准将重点聚焦接口规范、安全评估、质量评测三大领域，预计明年底前发布。"
            ),
            "source": "新华社客户端",
            "pub_time": _ts_to_str(now - 3600 * 24),
            "url": f"https://example.com/news/mock?q={quote_plus(query)}&n=4",
        },
        {
            "title": f"高校团队在「{query}」方向取得关键技术突破，论文登上顶刊",
            "summary": (
                "某顶尖高校联合实验室宣布，在该方向上取得里程碑式突破，论文已被国际顶级期刊接收。"
                "据悉，该成果在核心指标上较现有 SOTA 提升 27%，并已申请 3 项发明专利。"
            ),
            "source": "中国教育报",
            "pub_time": _ts_to_str(now - 3600 * 36),
            "url": f"https://example.com/news/mock?q={quote_plus(query)}&n=5",
        },
        {
            "title": f"国家出台新政支持「{query}」产业发展，专项资金规模超百亿",
            "summary": (
                "多部委联合印发指导意见，明确将该产业列为战略性新兴产业重点方向。"
                "配套专项资金、税收优惠、人才引进等一揽子支持政策，部分地方已先行落地。"
            ),
            "source": "人民日报经济版",
            "pub_time": _ts_to_str(now - 3600 * 48),
            "url": f"https://example.com/news/mock?q={quote_plus(query)}&n=6",
        },
        {
            "title": f"创业者亲述：我们是如何用「{query}」三个月拿到 A 轮融资的",
            "summary": (
                "创业公司创始人在近期分享会上透露，团队围绕核心场景打磨产品，"
                "3 个月内完成从 MVP 到付费客户验证，并获得头部机构数千万元 A 轮投资。"
                "他认为关键是「痛点要准、落地要快、数据要实」。"
            ),
            "source": "36 氪",
            "pub_time": _ts_to_str(now - 3600 * 60),
            "url": f"https://example.com/news/mock?q={quote_plus(query)}&n=7",
        },
        {
            "title": f"风险提示：{query}领域需警惕三类常见投资陷阱",
            "summary": (
                "监管部门近期发布风险提示，点名三类常见问题：① 虚假宣传「黑科技」；"
                "② 以「国家项目」名义非法集资；③ 打着培训旗号收取高额加盟费。提醒公众注意甄别。"
            ),
            "source": "央视财经",
            "pub_time": _ts_to_str(now - 3600 * 72),
            "url": f"https://example.com/news/mock?q={quote_plus(query)}&n=8",
        },
        {
            "title": f"海外观察：海外巨头加速布局「{query}」，国内企业如何应对？",
            "summary": (
                "近一个季度海外科技巨头动作频繁，密集发布相关产品与战略。"
                "业内人士认为，国内企业在应用场景与数据资源上具备比较优势，"
                "但需加快核心技术自主研发，避免关键环节受制于人。"
            ),
            "source": "第一财经",
            "pub_time": _ts_to_str(now - 3600 * 96),
            "url": f"https://example.com/news/mock?q={quote_plus(query)}&n=9",
        },
        {
            "title": f"普通人如何抓住「{query}」时代的机遇？业内大咖给出 5 条建议",
            "summary": (
                "在近日举办的行业峰会上，多位嘉宾就个人如何拥抱新一轮技术浪潮展开讨论。"
                "核心建议包括：保持学习敏感度、识别自己的独特优势、避免盲目跟风转行、"
                "关注所在行业与新技术的结合点、重视长期复利而非短期套利。"
            ),
            "source": "虎嗅",
            "pub_time": _ts_to_str(now - 3600 * 120),
            "url": f"https://example.com/news/mock?q={quote_plus(query)}&n=10",
        },
        {
            "title": f"新职业观察：围绕「{query}」已涌现 8 个高薪新兴岗位",
            "summary": (
                "招聘平台最新数据显示，相关领域新发职位数同比增长 142%，"
                "平均招聘薪酬高出全行业平均水平约 48%。热门岗位包括算法工程师、"
                "解决方案架构师、数据标注专家、产品经理、安全评估师等。"
            ),
            "source": "智联招聘研究院",
            "pub_time": _ts_to_str(now - 3600 * 144),
            "url": f"https://example.com/news/mock?q={quote_plus(query)}&n=11",
        },
        {
            "title": f"年度盘点：2026 年「{query}」领域十大标志性事件",
            "summary": (
                "时至年中，媒体盘点今年以来该领域十大关键事件，涵盖技术突破、"
                "政策出台、大额融资、产品发布、标准制定等多个维度。"
                "业内普遍认为，2026 年将成为该产业从「尝鲜」走向「普及」的关键拐点。"
            ),
            "source": "InfoQ",
            "pub_time": _ts_to_str(now - 3600 * 168),
            "url": f"https://example.com/news/mock?q={quote_plus(query)}&n=12",
        },
    ]
    return base[:max_results]


def _try_bing_search(query: str, max_results: int, hours: int) -> tuple[str, list[dict[str, Any]]]:
    """尝试必应新闻搜索（requests + 公开 JSON 端点）。

    返回：(engine_tag, items)；失败 items 为空，由上层回退。
    """
    try:
        import requests  # type: ignore
    except Exception:  # noqa: BLE001
        return "bing(requests unavailable)", []

    try:
        # 必应 v7 news search（无需 key 的公开端点，返回可能被限流但免 key）
        params = {
            "q": query,
            "count": max_results,
            "mkt": "zh-CN",
            "setLang": "zh-CN",
            "safeSearch": "Moderate",
        }
        if hours > 0:
            # 必应 freshness: Day(24h) / Week / Month —— 粗略映射
            if hours <= 24:
                params["freshness"] = "Day"
            elif hours <= 24 * 7:
                params["freshness"] = "Week"
            else:
                params["freshness"] = "Month"
        url = "https://www.bing.com/news/search"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        # 注意：真正的 Bing News API 认知服务需要 key；这里抓取公开搜索页（可能失败，失败就回退）
        # 这里尝试直接拿 JSON 端点（若被限流就空列表，非常正常）
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        html = resp.text
        # 非常粗糙地从 HTML 中抽取 <a title=... href=...> —— 失败就空列表（上层有 fallback）
        import re as _re

        results: list[dict[str, Any]] = []
        # 抽取 <a class="title" ...>
        for m in _re.finditer(
            r'<a[^>]+class="[^"]*title[^"]*"[^>]+href="([^"]+)"[^>]+title="([^"]+)"',
            html,
        ):
            if len(results) >= max_results:
                break
            href, title = m.group(1), m.group(2)
            title = title.replace("&quot;", '"').replace("&amp;", "&")
            results.append(
                {
                    "title": title,
                    "summary": "",
                    "source": "Bing 新闻",
                    "pub_time": "",
                    "url": href if href.startswith("http") else "https://www.bing.com" + href,
                }
            )
        return "bing", results
    except Exception:  # noqa: BLE001
        return "bing(failed)", []


def _try_baidu_rss(query: str, max_results: int, hours: int) -> tuple[str, list[dict[str, Any]]]:
    """尝试百度新闻 RSS（news.baidu.com/ns?word=xxx&tn=newsrss）。

    返回：(engine_tag, items)；失败 items 为空，由上层回退。
    """
    try:
        import requests  # type: ignore
    except Exception:  # noqa: BLE001
        return "baidu(requests unavailable)", []

    try:
        params = {
            "word": query,
            "tn": "newsrss",
            "rn": max_results,
            "cl": 2,
            "ct": 1,
        }
        url = "https://news.baidu.com/ns"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
        }
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        xml = resp.text
        import re as _re
        from xml.etree import ElementTree as _ET

        results: list[dict[str, Any]] = []
        try:
            root = _ET.fromstring(xml)
            for item in root.iter("item"):
                if len(results) >= max_results:
                    break
                title_el = item.find("title")
                link_el = item.find("link")
                desc_el = item.find("description")
                pub_el = item.find("pubDate")
                title = (title_el.text or "").strip() if title_el is not None else ""
                link = (link_el.text or "").strip() if link_el is not None else ""
                desc = (desc_el.text or "").strip() if desc_el is not None else ""
                pub = (pub_el.text or "").strip() if pub_el is not None else ""
                # 百度 RSS 的 desc 常有 HTML，去除标签
                desc = _re.sub(r"<[^>]+>", "", desc)
                if not title and not desc:
                    continue
                results.append(
                    {
                        "title": title or "(无标题)",
                        "summary": desc,
                        "source": "百度新闻",
                        "pub_time": pub[:16],
                        "url": link,
                    }
                )
        except Exception:  # noqa: BLE001
            pass
        return "baidu", results
    except Exception:  # noqa: BLE001
        return "baidu(failed)", []


# ============================================================
# SearchNewsTool
# ============================================================

class SearchNewsTool(BaseTool):
    """按关键词搜索新闻资讯，返回结构化的 Top N 条结果。

    引擎策略：auto → bing → baidu → mock 四级降级，保证离线 / 限流 / 网络异常下仍返回可用结果。
    注意：当前版本不做「LLM 总结」，直接返回原始结构化搜索结果（后续可接 LLM 归纳）。
    """

    name: ClassVar[str] = "search_news"
    description: ClassVar[str] = (
        "Tool Name: search_news\n"
        "用途：按关键词搜索最新新闻资讯，返回结构化列表（标题 / 摘要 / 来源 / 时间 / URL）。\n"
        "典型场景：\n"
        "  - 用户问「最近有什么关于 AI 的新闻」→ query=AI，默认 10 条\n"
        "  - 用户问「帮我看看今天股市相关资讯」→ query=今日股市，hours=24\n"
        "引擎策略：\n"
        "  * engine=auto（默认）：依次尝试必应 → 百度 → 本地模拟，保证有结果。\n"
        "  * engine=mock：直接返回模拟数据（离线调试 / 避免网络请求时用）。\n"
        "  * engine=bing / baidu：单独指定某一个引擎，失败就返回空。\n"
        "注意：\n"
        "  ① 当前版本**不调用 LLM 总结**，只返回原文搜索结果列表（后续版本可加总结选项）；\n"
        "  ② 若 hours>0，按最近 N 小时过滤，粗略映射到引擎的 Day/Week/Month；\n"
        "  ③ 网络失败 / 被限流时会自动回退模拟数据，确保不会报大红叉。\n"
    )
    args_schema: type[BaseModel] = SearchNewsArgs
    return_direct: ClassVar[bool] = False

    def _run(  # noqa: D401
        self,
        query: str,
        engine: str = "auto",
        max_results: int = 10,
        hours: int = 0,
    ) -> str:
        t0 = time.perf_counter_ns()
        try:
            q = (query or "").strip()
            if not q:
                raise ValueError("query 不能为空，请提供搜索关键词。")

            engine_choice = (engine or "auto").strip().lower()
            items: list[dict[str, Any]] = []
            tag = ""

            if engine_choice == "mock":
                items = _mock_news(q, max_results)
                tag = "mock"
            elif engine_choice in ("bing",):
                tag, items = _try_bing_search(q, max_results, hours)
            elif engine_choice in ("baidu",):
                tag, items = _try_baidu_rss(q, max_results, hours)
            else:  # auto
                # 依次尝试：bing → baidu → mock
                tag, items = _try_bing_search(q, max_results, hours)
                if not items:
                    tag2, items = _try_baidu_rss(q, max_results, hours)
                    tag = f"{tag} → {tag2}" if tag else tag2
                if not items:
                    items = _mock_news(q, max_results)
                    tag = f"{tag} → mock" if tag else "mock"

            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            return _format_news(items[:max_results], q, tag, elapsed_ms)
        except Exception as e:  # noqa: BLE001
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ search_news 失败（{elapsed_ms} ms）：{type(e).__name__}: {e}"


__all__ = ["SearchNewsTool"]
