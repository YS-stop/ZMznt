"""Qwen 云端探活脚本：最小化请求（只消耗几个 token），验证 Key/模型/URL/网络是否全通。

如果连这个最小请求都 401/403，说明 Key 或额度问题，不需要跑完整 e2e 浪费时间。

运行方式：
    & "d:\zhuomZNT\venv_assistant\Scripts\Activate.ps1"; python d:\zhuomZNT\tests\probe_qwen_live.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env", override=False)


def mask_key(s: str) -> str:
    if not s:
        return "<empty>"
    return s[:4] + "*" * max(4, len(s) - 8) + s[-4:] if len(s) > 8 else "*" * len(s)


print("=" * 70)
print("【Qwen 云端探活】最小请求 只费几个 token 验证链路")
print("=" * 70)
KEY = os.environ.get("QWEN_API_KEY", "")
BASE = os.environ.get("QWEN_BASE_URL", "")
MODEL = os.environ.get("QWEN_MODEL", "")
print(f"  QWEN_API_KEY  : {mask_key(KEY)} (len={len(KEY)})")
print(f"  QWEN_BASE_URL : {BASE}")
print(f"  QWEN_MODEL    : {MODEL}")
print("=" * 70)

if not KEY or not BASE or not MODEL:
    print("❌ 配置不完整：请确保 .env 中 QWEN_API_KEY / QWEN_BASE_URL / QWEN_MODEL 全部填写！")
    sys.exit(2)

print("[1/2] 初始化 get_qwen_llm()……")
from src.infra.llm_client import get_qwen_llm  # noqa: E402
llm = get_qwen_llm()
print("      类型:", type(llm).__module__ + "." + type(llm).__name__)
print(f"      配置的 model: {getattr(llm, 'model_name', getattr(llm, 'model', '?'))}")

print("[2/2] 发起最小 LLM 请求……（只要求回复 10 个字内，省 token）")
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
t0 = time.perf_counter_ns()

SYS = SystemMessage(content="你是一个极简小助手，只用 10 个汉字以内回答，不要说多余的话。")
H = HumanMessage(content="你好，告诉我今天是周几？")
try:
    resp = llm.invoke([SYS, H])
    ms = (time.perf_counter_ns() - t0) // 1_000_000
    content = (resp.content or "").strip()
    print(f"      ✅ 调用成功！耗时 {ms} ms")
    print(f"      模型回答（原样）：{content!r}")
    # 其他元信息：Usage 能拿到就打印（token 消耗）
    usage = getattr(resp, "response_metadata", {}).get("token_usage")
    if usage:
        print(f"      Token 消耗：{usage}")
    id_ = getattr(resp, "id", None)
    if id_:
        print(f"      request_id/completion_id：{id_}")
    print("\n🎉 探活通过！Key/模型/网络 全通，现在可以跑完整 e2e（3 工具端到端）。")
    sys.exit(0)

except Exception as e:  # noqa: BLE001
    ms = (time.perf_counter_ns() - t0) // 1_000_000
    print(f"      ❌ 调用失败！耗时 {ms} ms")
    print(f"      异常类型: {type(e).__module__}.{type(e).__name__}")
    msg = str(e)
    # 关键错误码：
    #   401 AuthenticationError → Key 不对 / 没传 / 格式错
    #   403 PermissionDeniedError → 额度用完 / 账号被禁 / use free tier only
    #   404 NotFoundError → 模型名不对 / base_url 不对
    #   429 RateLimitError → 频率超限（免费用户 QPS/TPS 不够）
    #   ConnectionError / SSL → 网络 / 代理 / 墙
    import traceback
    try:
        status_code = getattr(e, "status_code", None)
        body = getattr(e, "body", None)
        print(f"      HTTP 状态码: {status_code}")
        if body:
            print(f"      错误 body: {body!r}")
    except Exception:  # noqa: BLE001
        pass
    print(f"      异常 msg: {msg[:800]}")
    # 用户友好建议
    if "401" in msg or "AuthenticationError" in type(e).__name__:
        print("\n💡 建议：401 = API Key 错/无效。请复制 .env QWEN_API_KEY 的完整值到 DashScope 控制台核对是否正确，不要有前后空格。")
    elif "403" in msg or "PermissionDenied" in type(e).__name__ or "quota" in msg.lower() or "Allocate" in msg:
        print("\n💡 建议：403 = 额度/权限问题。请去 DashScope 控制台：")
        print("   a) 顶部『财务中心』→ 充值几块钱")
        print("   b) 或者 右上角设置 → 账户 → 关闭『仅使用免费额度』开关")
        print("   c) 或者 换更便宜的模型：.env QWEN_MODEL=qwen-turbo（qwen-turbo 价格低 ~50%，免费额度可能还有剩）")
    elif "404" in msg or "NotFound" in type(e).__name__:
        print("\n💡 建议：404 = 模型名或 URL 错。请核对：")
        print("   .env QWEN_BASE_URL 必须是 https://dashscope.aliyuncs.com/compatible-mode/v1（结尾不要有 /chat/completions）")
        print("   .env QWEN_MODEL 可用值：qwen-plus / qwen-turbo / qwen-max / qwen2.5-72b-instruct ……")
    elif "429" in msg or "RateLimit" in type(e).__name__:
        print("\n💡 建议：429 = 频率超限。免费账号一般 60次/分钟，等 30 秒再试；或升级付费计划。")
    elif "SSL" in msg or "EOF" in msg or "ConnectionError" in msg or "Timeout" in type(e).__name__:
        print("\n💡 建议：网络/SSL/代理问题。请检查：")
        print("   a) 是否开了 clash/v2ray 等全局代理？关掉再试，或设置环境变量 NO_PROXY=.aliyuncs.com")
        print("   b) 用浏览器打开 https://dashscope.aliyuncs.com 看能不能打开")
    else:
        print("\n💡 建议：未知错误，已打印堆栈前 800 字，把完整异常贴到 DashScope 工单群里问。")
    print("\n----- 完整堆栈（调试用）-----")
    traceback.print_exc(limit=5)
    sys.exit(1)
