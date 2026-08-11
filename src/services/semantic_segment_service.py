"""M8 语义级智能断句服务：LLM 判断语音识别文本的语义完整性。

核心功能：
    1. 收尾词检测（关键词快速路径，不调LLM）
    2. 语义完整性判断（LLM 判断，用于高级版持续监听）
    3. 结果缓存（短时间内相同文本不重复调用LLM）

设计原则：
    - 快速路径优先：收尾词命中立即返回 COMPLETE，零延迟
    - LLM 调用超时短（3s）：超时默认返回 INCOMPLETE 继续等，宁可不截错
    - 结果带置信度：上层可根据置信度决定是否提交
    - 线程安全：可从监听线程安全调用

收尾词列表（中文常用指令结束语）：
    执行、开始、去吧、就这样、好了、行、可以、确定、确认、OK、好的、
    吧、呀、呢、啊（语气词结尾通常是完整句）
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class SegmentJudgment(str, Enum):
    """断句判定结果。"""
    COMPLETE = "complete"       # 语义完整，可以提交
    INCOMPLETE = "incomplete"   # 语义不完整，继续听
    UNCERTAIN = "uncertain"     # 不确定（保守策略：继续听短时间再判）


@dataclass
class SegmentResult:
    """断句判断结果。"""
    judgment: SegmentJudgment
    confidence: float = 0.0     # 0~1，1=最确定
    reason: str = ""            # 判定理由（调试用）
    is_fast_path: bool = False  # 是否走了快速路径（收尾词）


# ---------------- 快速路径：收尾词/语气词检测 ----------------

# 强收尾词（明确指令结束，立即提交）
_STRONG_FINISH_WORDS = (
    "执行吧", "开始吧", "去吧", "就这样", "好了", "可以了",
    "确定", "确认", "执行", "开始", "去做", "马上", "立刻",
    "ok", "okay",
)

# 语气词结尾（通常是完整问句/祈使句）
_FINAL_PARTICLES = ("吧", "呀", "呢", "啊", "吗", "嘛", "呗", "啦", "喽")

# 明显的完整句式标记（至少需要宾语/补语，不能是单个动词）
_COMPLETE_PATTERNS = [
    r"^帮我[^，。？！]{2,}[吧呀啊]$",      # 帮我xxx吧/呀/啊（至少2个中间字）
    r"^打开[^，。？！吧呀啊]{1,}[吧呀啊]?$",  # 打开xxx（xxx至少1字），如"打开抖音"
    r"^关闭[^，。？！吧呀啊]{1,}[吧呀啊]?$",  # 关闭xxx
    r"^搜索[^，。？！吧呀啊]{1,}[吧呀啊]?$",  # 搜索xxx
    r"^创建[^，。？！吧呀啊]{1,}[吧呀啊]?$",  # 创建xxx
    r"^删除[^，。？！吧呀啊]{1,}[吧呀啊]?$",  # 删除xxx
    r"^看看[^，。？！吧呀啊]{1,}[吧呀啊]?$",  # 看看xxx
    r"^音量(?:调到|调至|设置为|设为)[^，。？！吧呀啊]{1,}[%％度]?$",  # 音量调到xx
    r"^把[^，。？！]{2,}(?:打开|关闭|删掉|删除|调到|设为)[^，。？！]*$",
    r"^在[^，。？！]{2,}(?:创建|新建|建|写|找|打开)[^，。？！]{1,}$",
    r"^搜(?:索|一下)[^，。？！]{1,}$",
    r"^今[天海]?[有什发]?[^，。？！]{1,}$",  # 今天有什么/今天热搜
]


class SemanticSegmenter:
    """语义断句器：快速路径 + LLM 慢速路径。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, Tuple[SegmentResult, float]] = {}  # text -> (result, timestamp)
        self._cache_ttl_sec: float = 5.0  # 缓存5秒（同一句话不会短时间内重复判）
        self._llm = None
        self._llm_failed: bool = False

    # ---------------- 对外 API ----------------

    def judge(self, text: str, min_chars: int = 2) -> SegmentResult:
        """判断文本语义是否完整。

        Args:
            text: 当前ASR识别文本（增量累积的）
            min_chars: 低于此字符数直接判 INCOMPLETE（避免单字误触发）

        Returns:
            SegmentResult
        """
        clean = (text or "").strip()
        if not clean:
            return SegmentResult(SegmentJudgment.INCOMPLETE, 0.0, "空文本")

        # 过短直接不完整
        if len(clean) < min_chars:
            return SegmentResult(SegmentJudgment.INCOMPLETE, 0.2, f"文本过短({len(clean)}字)")

        # 查缓存
        with self._lock:
            cached = self._cache.get(clean)
            if cached is not None:
                res, ts = cached
                if time.time() - ts < self._cache_ttl_sec:
                    return res

        # 1. 快速路径：强收尾词命中
        res = self._fast_path_check(clean)
        if res is not None:
            self._cache_put(clean, res)
            return res

        # 2. 慢速路径：LLM判断（如果LLM可用）
        res = self._llm_check(clean)
        if res is not None:
            self._cache_put(clean, res)
            return res

        # 兜底：不确定
        res = SegmentResult(SegmentJudgment.UNCERTAIN, 0.3, "快速路径未命中，LLM不可用，保守不确定")
        self._cache_put(clean, res)
        return res

    def reset_cache(self) -> None:
        """清空缓存（新的一轮监听开始时调用）。"""
        with self._lock:
            self._cache.clear()

    # ---------------- 内部：快速路径 ----------------

    def _fast_path_check(self, text: str) -> Optional[SegmentResult]:
        """收尾词/句式快速检测，不调LLM。"""
        low = text.lower().rstrip("。！？!?，, ")

        # 强收尾词（结尾匹配优先）
        for word in _STRONG_FINISH_WORDS:
            if low.endswith(word) or low.endswith(word + "。") or low.endswith(word + "！"):
                return SegmentResult(
                    SegmentJudgment.COMPLETE,
                    0.85,
                    f"命中强收尾词「{word}」",
                    is_fast_path=True,
                )

        # 语气词结尾 + 一定长度
        for particle in _FINAL_PARTICLES:
            if low.endswith(particle) and len(low) >= 4:
                # 再检查是否有明显不完整的开头
                if not self._looks_obviously_incomplete(low):
                    return SegmentResult(
                        SegmentJudgment.COMPLETE,
                        0.65,
                        f"语气词「{particle}」结尾且句式完整",
                        is_fast_path=True,
                    )

        # 完整句式正则匹配
        for pat in _COMPLETE_PATTERNS:
            if re.search(pat, low):
                return SegmentResult(
                    SegmentJudgment.COMPLETE,
                    0.7,
                    f"命中完整句式「{pat[:20]}...」",
                    is_fast_path=True,
                )

        # 明显不完整的开头
        if self._looks_obviously_incomplete(low):
            return SegmentResult(
                SegmentJudgment.INCOMPLETE,
                0.75,
                "句式明显不完整（如「帮我把」「打开一」等）",
                is_fast_path=True,
            )

        return None  # 快速路径未命中，走LLM

    @staticmethod
    def _looks_obviously_incomplete(text: str) -> bool:
        """判断文本是否明显没说完（如结尾是量词/介词/助词/数词）。"""
        # 以这些字结尾通常不完整（还在构思宾语/补语）
        bad_endings = (
            "的", "把", "被", "给", "让", "在", "和", "与", "跟",
            "一", "个", "几", "些", "下", "点",
            "了", "着",  # 注意："好了/行了/可以了"是例外，下面特判
            "我", "你", "他",
        )
        t = text.rstrip("。！？!?，, ")
        if len(t) <= 2:
            return False  # 太短不判断（如"打开"虽然2字，但可能是打开浏览器省略说法）
        # 例外：常见完整短语
        complete_exceptions = ("好了", "行了", "可以了", "确定", "确认", "执行", "开始", "去吧", "就这样")
        for exc in complete_exceptions:
            if t.endswith(exc):
                return False
        last_char = t[-1]
        if last_char in bad_endings:
            return True
        # 结尾是"个"+数词/指示词（如"一个"、"这个"）
        if len(t) >= 2 and t[-1] == "个" and t[-2] in "一这那每某":
            return True
        return False

    # ---------------- 内部：LLM 慢速路径 ----------------

    def _ensure_llm(self) -> bool:
        """懒加载LLM客户端（第一次调用时才初始化，避免启动慢）。"""
        if self._llm is not None:
            return True
        if self._llm_failed:
            return False
        try:
            from src.infra.llm_client import get_main_llm
            self._llm = get_main_llm(temperature=0.0, timeout=3)
            return True
        except Exception:
            self._llm_failed = True
            return False

    def _llm_check(self, text: str) -> Optional[SegmentResult]:
        """调用LLM判断语义完整性。超时或失败返回None。"""
        if not self._ensure_llm():
            return None

        prompt = (
            "你是一个语音指令断句助手。用户正在用语音向桌面助手发出指令，ASR实时识别的文字是：\n"
            f"「{text}」\n\n"
            "请判断这句话作为一个桌面操作指令，语义是否已经完整（用户是否已经说完）。\n"
            "只回答一个词：\n"
            "- COMPLETE：语义完整（如「打开抖音」「帮我创建一个文件」「音量调到50」），可以提交执行\n"
            "- INCOMPLETE：语义不完整（如「帮我把」「打开一个」「音量调到」），需要继续听\n"
            "- UNCERTAIN：不确定（保守策略，倾向继续听）\n"
            "不要解释，不要加标点，只返回这三个词之一。"
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            # 3秒超时（在帧循环里不能卡太久）
            msgs = [
                SystemMessage(content="你是精准的中文语音指令断句判断器。只输出COMPLETE/INCOMPLETE/UNCERTAIN三个词之一。"),
                HumanMessage(content=prompt),
            ]
            resp = self._llm.invoke(msgs)
            ans = str(getattr(resp, "content", resp) or "").strip().upper()

            if "COMPLETE" in ans and "INCOMPLETE" not in ans:
                return SegmentResult(SegmentJudgment.COMPLETE, 0.7, f"LLM判定完整：{ans[:40]}")
            if "INCOMPLETE" in ans:
                return SegmentResult(SegmentJudgment.INCOMPLETE, 0.7, f"LLM判定不完整：{ans[:40]}")
            return SegmentResult(SegmentJudgment.UNCERTAIN, 0.4, f"LLM不确定：{ans[:40]}")
        except Exception as e:
            # LLM调用失败不影响主流程，下次还可以重试（_llm_failed不设True，只这一次超时）
            return SegmentResult(SegmentJudgment.UNCERTAIN, 0.2, f"LLM调用异常：{type(e).__name__}")

    def _cache_put(self, text: str, result: SegmentResult) -> None:
        with self._lock:
            self._cache[text] = (result, time.time())
            # 简单防内存膨胀：超过50条就清掉最早的
            if len(self._cache) > 50:
                # 按时间排序保留最近20条
                items = sorted(self._cache.items(), key=lambda x: x[1][1])
                self._cache = dict(items[-20:])


# 模块级单例
_SEGMENTER: Optional[SemanticSegmenter] = None


def get_semantic_segmenter() -> SemanticSegmenter:
    global _SEGMENTER
    if _SEGMENTER is None:
        _SEGMENTER = SemanticSegmenter()
    return _SEGMENTER


def judge_speech_complete(text: str, min_chars: int = 2) -> SegmentResult:
    """便捷函数：判断语音文本是否语义完整。"""
    return get_semantic_segmenter().judge(text, min_chars=min_chars)


__all__ = [
    "SemanticSegmenter",
    "SegmentJudgment",
    "SegmentResult",
    "get_semantic_segmenter",
    "judge_speech_complete",
]
