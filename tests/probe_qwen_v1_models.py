"""零 token 探活：调 DashScope OpenAI 兼容 GET /v1/models，列出账号可用模型。

不花任何 token，只验证：
  1. API Key 是否真有效（401=Key错，403=Key无权限/use free tier only关？）
  2. Base_URL 是否拼对了（/v1/models 必须 200 才能说明 /chat/completions 也拼对了）
  3. 我们想调用的 qwen-plus / qwen-turbo / qwen-max，真的在账号可用模型列表里吗？

（OpenAI 兼容协议里 /v1/models 是 GET 方法，无 body，0 token 消耗。）
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
print("【零 token 探活】GET /v1/models 列账号可用模型")
print("=" * 70)
KEY = os.environ.get("QWEN_API_KEY", "").strip()
BASE = os.environ.get("QWEN_BASE_URL", "").strip().rstrip("/")
MODEL = os.environ.get("QWEN_MODEL", "").strip()
print(f"  QWEN_API_KEY  : {mask_key(KEY)} (len={len(KEY)})")
print(f"  QWEN_BASE_URL : {BASE!r}")
print(f"  QWEN_MODEL    : {MODEL!r}")
URL = f"{BASE}/models"  # 兼容模式的 /v1 下面一般路径是 /models 而不是 /v1/models，因为 BASE 已经带 /compatible-mode/v1
print(f"  GET URL       : {URL}")
print("=" * 70)

if not KEY or not BASE:
    print("❌ 配置不完整，请先填写 .env 的 QWEN_API_KEY / QWEN_BASE_URL")
    sys.exit(2)

print("[1/2] 发送 HTTP 请求 …（超时 20 秒）")
import requests  # noqa: E402

headers = {
    "Authorization": f"Bearer {KEY}",
    "Accept": "application/json",
    "User-Agent": "desktop-voice-assistant-m1-e2e/1.0",
}

t0 = time.perf_counter_ns()
try:
    r = requests.get(URL, headers=headers, timeout=20)
    ms = (time.perf_counter_ns() - t0) // 1_000_000
    print(f"      HTTP 状态码: {r.status_code}  耗时: {ms} ms")
    print(f"      Content-Type: {r.headers.get('Content-Type')}")

    if r.status_code == 200:
        data = r.json()
        models = data.get("data", []) or []
        ids = sorted({str(m.get("id", "")).strip() for m in models if m.get("id")})
        print("\n✅ /v1/models 成功！账号可用模型列表（按字母排序，去重）：")
        for i, mid in enumerate(ids, start=1):
            mark = "  ← 🎯 当前 .env 配置的模型在这里！" if mid == MODEL else ""
            print(f"  [{i:>2}] {mid}{mark}")

        if MODEL not in ids:
            print(f"\n⚠️  你的 .env 配置 MODEL={MODEL!r} 不在可用列表里！")
            # 推荐几个：取包含 qwen-plus / qwen-turbo / qwen-max / qwen3 的
            recs = [m for m in ids if any(k in m.lower() for k in ("qwen-plus", "qwen-turbo", "qwen-max", "qwen3"))]
            if recs:
                print("💡 推荐可用的替代（在可用列表里）：")
                for m in recs[:8]:
                    print(f"     - QWEN_MODEL={m}")
            print("请复制一个确确实实在上面列表里的 id，替换 .env 的 QWEN_MODEL= 行！")

        # 403 还会 200？一般不会，如果 200 说明 Key 权限没问题，那 AllocationQuota.FreeTierOnly 才是开关问题。
        if MODEL in ids:
            print("\n🎉 关键结论：")
            print("   ✅ API Key 有效（/v1/models 200 OK）")
            print("   ✅ 配置的 MODEL 确实在账号可用列表里")
            print("   ❓ 之前 /chat/completions 返回 403 AllocationQuota.FreeTierOnly 的唯一原因 =")
            print("      DashScope 控制台 → 账户设置页 默认开了『仅使用免费额度』开关")
            print("      👉 去控制台关掉它就能用余额/购买的 token 额度了：")
            print("         https://dashscope.console.aliyun.com/")
            print("         登录 → 右上角头像 → 账户管理/API-KEY 管理 / 计费设置")
            print("         找『仅使用免费额度 / use free tier only』的开关，关掉它")
        sys.exit(0)

    # —— 非 200 错误码分支 ——
    print(f"\n❌  HTTP {r.status_code} 失败")
    try:
        err = r.json()
        print("响应 JSON body:")
        import json
        print(json.dumps(err, ensure_ascii=False, indent=2)[:1500])
    except Exception:
        print("响应纯文本:")
        print((r.text or "")[:1500])

    # 用户友好建议
    if r.status_code == 401:
        print("\n💡 401 = Authorization 失败。请核对：")
        print("   a) .env QWEN_API_KEY 复制是否完整（前后没有空格/换行）")
        print("   b) Key 是否在 https://dashscope.console.aliyun.com/apiKey 创建且状态『有效』")
        print("   c) 是不是把 DASHSCOPE_API_KEY 抄错了？两个 Key 可以相同，但必须是 DashScope 的有效 Key")
    elif r.status_code == 403:
        print("\n💡 403 = Key 或 IP 或模型访问 权限不足。")
        print("   a) 控制台 → 权限管理 → 你这个 Key 有没有开通『百炼/通义千问 / OpenAI 兼容模式』权限")
        print("   b) 开了 IP 白名单？需要把本机 IP 加进去")
    elif r.status_code == 404:
        print("\n💡 404 = URL 拼错。请核对：")
        print("   .env QWEN_BASE_URL 正确值必须是 https://dashscope.aliyuncs.com/compatible-mode/v1")
        print("   （不要带 /chat/completions 后缀！LangChain ChatOpenAI 内部会追加 /chat/completions）")
    elif r.status_code == 429:
        print("\n💡 429 = 频率限制。等 10 秒再试一次。")
    elif r.status_code >= 500:
        print("\n💡 5xx = DashScope 内部服务故障，过几分钟再试，或看阿里云官方健康状态页。")
    sys.exit(1)

except Exception as e:  # noqa: BLE001
    ms = (time.perf_counter_ns() - t0) // 1_000_000
    print(f"\n❌  网络/连接异常（{ms} ms）: {type(e).__name__}: {e}")
    print("\n💡 建议：")
    print("   a) 开了 Clash/V2ray 全局代理？临时关掉，或 cmd set NO_PROXY=.aliyuncs.com")
    print("   b) 浏览器打开 https://dashscope.aliyuncs.com 看看是否能正常打开（返回 JSON/401 都算网络通）")
    print("   c) 公司网络/校园网可能有防火墙，开手机热点再试")
    import traceback
    traceback.print_exc(limit=4)
    sys.exit(3)
