"""M4 阶段验收测试：TTS 语音合成 + 播报（Edge-TTS 在线 + CosyVoice 占位）。

验收标准（10 个用例）：
    ✅ T1  tts_service.get_tts_info() 引擎报告：engine=edge-tts（环境已装包）且 note 不是空
    ✅ T2  别名归一化：_normalize_voice → default/xiaoxiao/yunxi/news/cantonese/en 正确映射到 Edge-TTS ShortName
    ✅ T3  语速换算：_speed_to_rate(0.8/1.0/1.5) → '-20%' / '+0%' / '+50%'
    ✅ T4  synthesize_to_file 合成 1 句短中文：有外网 → 生成 mp3 >200 字节；无外网/失败 → 返回中文错误，不抛异常
    ✅ T5  play_audio(不存在的文件) → False + 中文提示，不抛异常
    ✅ T6  VoiceOutputTool(play_now=False) 只合成不播放：成功/失败都返回结构化 observation，不抛异常
    ✅ T7  VoiceOutputTool(text="") 空文本 → 返回中文错误提示，不抛异常
    ✅ T8  工具注册：TOOLS['voice_output'] 存在，get_all_tools() 共 8 个
    ✅ T9  tts_service.speak 综合返回 dict 含 ok/engine/text_len/audio_path/audio_size_bytes/play_ok 7 个键
    ✅ T10 get_diagnostics 无语法错误（pytest 运行过程验证）

注意：Edge-TTS 需要外网，如果环境断网，T4/T6/T9 会走到降级分支，不要求成功合成，但要求不抛异常。
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# T1 ~ T3：tts_service 基础设施
# ---------------------------------------------------------------------------

def test_T1_tts_service_info():
    from src.services.tts_service import get_tts_info
    info = get_tts_info()
    print("  [T1] tts_info =", info)
    assert isinstance(info, dict), "get_tts_info 必须返回 dict"
    assert "engine" in info and info["engine"] in (
        "edge-tts",
        "text-mock",
    ), "engine 必须是 edge-tts 或 text-mock"
    assert info.get("note", "").strip() != "", "note 不能为空，要告诉用户引擎状态"


def test_T2_voice_alias_normalization():
    from src.services.tts_service import _normalize_voice
    cases = {
        "default": "zh-CN-XiaoxiaoNeural",
        "xiaoxiao": "zh-CN-XiaoxiaoNeural",
        "female": "zh-CN-XiaoxiaoNeural",
        "girl": "zh-CN-XiaoxiaoNeural",
        "yunxi": "zh-CN-YunxiNeural",
        "male": "zh-CN-YunxiNeural",
        "boy": "zh-CN-YunxiNeural",
        "news": "zh-CN-YunjianNeural",
        "cantonese": "zh-HK-HiuGaaiNeural",
        "yue": "zh-HK-HiuGaaiNeural",
        "hk": "zh-HK-HiuGaaiNeural",
        "en": "en-US-AriaNeural",
        "english": "en-US-AriaNeural",
        "aria": "en-US-AriaNeural",
        None: "zh-CN-XiaoxiaoNeural",
        "zh-CN-YunjianNeural": "zh-CN-YunjianNeural",  # 直接传 ShortName 原样返回
    }
    for alias, expect in cases.items():
        got = _normalize_voice(alias)
        print(f"  [T2] alias={alias!r} → {got}  (expect {expect})")
        assert got == expect, f"别名 {alias} 错误：期望 {expect}，实际 {got}"


def test_T3_speed_to_rate():
    from src.services.tts_service import _speed_to_rate
    cases = [(0.8, "-20%"), (1.0, "+0%"), (1.5, "+50%"), (0.49, "-50%"), (2.01, "+100%"), (1.15, "+15%")]
    for s, expect in cases:
        got = _speed_to_rate(s)
        print(f"  [T3] speed={s} → rate={got}  (expect {expect})")
        assert got == expect, f"speed {s} 换算错误：期望 {expect}，实际 {got}"


# ---------------------------------------------------------------------------
# T4：真实合成（有外网就真合成，没网也只降级返回中文错误，不抛异常）
# ---------------------------------------------------------------------------

def test_T4_synthesize_to_file_works_or_graceful_degrade(tmp_path: Path):
    from src.services.tts_service import tts_synthesize_to_file
    target = tmp_path / "m4_hello.mp3"
    ok, ret = tts_synthesize_to_file(
        "你好，这里是桌面语音助手 M4 阶段验收，现在测试短句语音合成。",
        save_path=target,
        voice="default",
        speed=1.0,
    )
    print(f"  [T4] ok={ok}, ret={ret}")
    # 不要求一定成功（可能外网断），但必须 (ok=True & 文件存在>200字节) 或 (ok=False & ret 是中文错误串)
    if ok:
        assert isinstance(ret, Path), f"成功时 ret 必须是 Path，实际 {type(ret)}：{ret}"
        p = Path(ret)
        assert p.exists(), f"合成成功但文件不存在：{p}"
        size = p.stat().st_size
        print(f"  [T4] 合成成功！大小 {size} 字节，路径：{p}")
        assert size > 200, f"mp3 太小（{size} 字节），疑似合成失败但误报成功"
    else:
        msg = str(ret)
        # 必须中文错误，且不能出现 Python 异常未捕获
        print(f"  [T4] 合成失败（可能无外网，可接受）：{msg}")
        assert isinstance(msg, str) and len(msg) > 6, f"失败提示太短或非中文：{msg!r}"
        # 不能是空字符串
        assert msg.strip() != "", "合成失败不能返回空串"


# ---------------------------------------------------------------------------
# T5：play_audio 路径不存在
# ---------------------------------------------------------------------------

def test_T5_play_audio_missing_path_returns_error_no_exception(tmp_path: Path):
    from src.services.tts_service import play_audio
    bad = tmp_path / "not_exist_12345.mp3"
    ok, note = play_audio(bad, backend="auto")
    print(f"  [T5] ok={ok}, note={note}")
    assert ok is False, "播放不存在的文件必须返回 False"
    assert "不存在" in note or "不存在" in note, f"提示语没包含不存在：{note}"


# ---------------------------------------------------------------------------
# T6 ~ T7：VoiceOutputTool
# ---------------------------------------------------------------------------

def test_T6_voice_output_tool_play_now_false(tmp_path: Path):
    from src.tools.voice_tools import VoiceOutputTool
    tool = VoiceOutputTool()
    # 只合成不播放（这样即使没装播放库也不影响）
    out = tmp_path / "tool_out.mp3"
    text = "欢迎使用桌面语音助手，当前为 M4 阶段 voice_output 工具集成测试。"
    obs = tool.run(
        {
            "text": text,
            "voice": "xiaoxiao",
            "speed": 1.0,
            "save_path": str(out),
            "play_now": False,
        }
    )
    print(f"  [T6]  observation 前 300 字符：\n      {obs[:300]}\n      ……")
    assert isinstance(obs, str), f"voice_output 必须返回 str observation，实际 {type(obs)}"
    # 不允许 Python 原始异常冒泡（"Traceback" 这种关键字）
    assert "Traceback" not in obs, f"observation 里有 Traceback 异常未捕获：\n{obs}"
    # 结构上必须返回含 🔊 voice_output（成功或失败都应该有结构化 head）
    assert "voice_output" in obs, f"observation 必须含 voice_output 标题：\n{obs}"


def test_T7_voice_output_tool_empty_text_returns_graceful_error():
    from src.tools.voice_tools import VoiceOutputTool
    tool = VoiceOutputTool()
    obs = tool.run({"text": "   \t\n "})  # 全是空白→视为空
    print(f"  [T7] empty-text obs: {obs}")
    assert isinstance(obs, str)
    assert "❌" in obs or "不能为空" in obs or "空" in obs, f"空文本必须报错但未报：{obs}"


# ---------------------------------------------------------------------------
# T8：工具注册（保持 8 个）
# ---------------------------------------------------------------------------

def test_T8_tool_registry_has_voice_output_and_count_8():
    from src.tools import AVAILABLE_TOOL_NAMES, TOOL_MAP, get_all_tools
    tools = get_all_tools()
    names = [t.name for t in tools]
    tool_map = TOOL_MAP()                   # lazy callable
    avail_names = set(AVAILABLE_TOOL_NAMES())
    m4_required_8 = {
        "create_file", "search_files", "delete_file", "recognize_file",
        "open_browser", "search_news", "voice_input", "voice_output",
    }
    print(f"  [T8] 已注册工具数={len(tools)}，名单：{names}")
    assert "voice_output" in tool_map, "TOOL_MAP 缺少 voice_output 工具"
    assert "voice_output" in avail_names, "AVAILABLE_TOOL_NAMES 缺少 voice_output"
    # M6 起加了 4 个系统工具，总数 ≥8 即可，不要求严格 ==8
    assert m4_required_8.issubset(avail_names), (
        f"缺少 M4 必需工具：缺少={sorted(m4_required_8 - avail_names)}，实际={sorted(avail_names)}"
    )
    assert len(tools) >= 8, (
        f"工具总数必须 ≥8（create/search/delete/recognize/open_browser/search_news/voice_input/voice_output + M6 系统工具），实际 {len(tools)}"
    )


# ---------------------------------------------------------------------------
# T9：speak 返回 dict 结构完整
# ---------------------------------------------------------------------------

def test_T9_speak_returns_well_formed_dict(tmp_path: Path):
    from src.services.tts_service import get_tts_service
    svc = get_tts_service()
    target = tmp_path / "speak_demo.mp3"
    res = svc.speak(
        "语音服务 speak 接口综合测试，合成并返回结构化结果。",
        voice="default",
        speed=1.0,
        save_path=target,
        play_now=False,
    )
    print("  [T9] speak() 结果 keys:", sorted(res.keys()))
    for k, v in res.items():
        print(f"       {k}: {v!r}" if k != "audio_path" or not res.get("ok") else f"       {k}: {v} (size={res.get('audio_size_bytes')})")
    for key in ("ok", "engine", "text_len", "audio_path", "audio_size_bytes", "play_ok", "play_note"):
        assert key in res, f"speak 结果缺少必要键 {key}，现有关键字：{list(res.keys())}"
    assert isinstance(res["ok"], bool)
    assert isinstance(res["text_len"], int) and res["text_len"] > 0
    assert isinstance(res["audio_size_bytes"], int) and res["audio_size_bytes"] >= 0
    assert isinstance(res["play_ok"], bool)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
