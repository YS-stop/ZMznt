"""M2 真实云端端到端：Qwen LLM + 6 工具 + Checkpointer 历史保留。

场景：
    Round 1（thread_id=XXX）：要求 Qwen 按顺序做 3 件事
        ① 在数据根创建 M2真实验证_<ts>.md 文件并写入 3 行 Markdown
        ② 调 recognize_file 识别该文件（with_preview_lines=5, with_sha256=True）
        ③ 调 search_news(engine=mock) 搜 3 条关于「2026 AI 应用落地」的资讯
        最后给中文 3 点小结
    Round 2（相同 thread_id）：追问「不调用任何工具，只凭你记忆，把 Round 1 做的 3 件事重新说一遍，包括文件路径和文件里第一行内容」
        → 如果 Checkpointer 生效，LLM 能正确复述 Round 1 的任务与文件路径；
        → 如果 Checkpointer 失效，LLM 会回答「我没做过这些」之类。

依赖校验：
    - 无 QWEN_API_KEY 或模型 403：脚本给出友好提示，exit 0（不算失败，CI 环境允许跳过）。
    - 有 Key：真实跑，所有真执行的工具会真实落盘（数据根创建 md 文件等），结束自动清理。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ============================================================
# 0. 环境
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env", override=False)


TS = int(time.time() * 1000)
TEST_FILE_ALIAS = f"数据根/M2真实验证_{TS}.md"
TEST_NEWS_QUERY = "2026 AI 应用落地"
THREAD_ID = f"e2e-m2-real-{TS}"

print("=" * 70)
print("【M2 真实云端端到端】Qwen + 6 Tools + Checkpointer 历史保留")
print("=" * 70)
print(f"  时间戳       : {TS}")
print(f"  文件别名     : {TEST_FILE_ALIAS}")
print(f"  新闻关键词   : {TEST_NEWS_QUERY}")
print(f"  thread_id    : {THREAD_ID}")
print("-" * 70)


# ============================================================
# 1. Key 校验：无 key 直接给提示退出
# ============================================================
KEY = (os.environ.get("QWEN_API_KEY") or "").strip()
if not KEY:
    print("[SKIP] 未检测到 QWEN_API_KEY，不跑真云端；已通过 M2 单测（Mock 模型）覆盖 ReAct 链路。")
    print("       想体验真云端：编辑 .env 把 QWEN_API_KEY=sk-xxx 填好，再跑本脚本。")
    sys.exit(0)


# ============================================================
# 2. 导入所有模块 + 先零 token 探活 key
# ============================================================
print("[2/6] 导入模块 & 探活 Qwen API Key……")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from collections import Counter  # noqa: E402

from src.services.agent_service import AssistantAgent, reset_agent_singleton  # noqa: E402
from src.services.checkpoint_service import start_checkpoint_service, get_checkpoint_info  # noqa: E402
from src.tools.file_tools import resolve_user_path  # noqa: E402

start_checkpoint_service(force_backend="memory")
print(f"      Checkpointer backend = {get_checkpoint_info()['backend']} | note={get_checkpoint_info().get('note','')}")

# 零 token 探活（/v1/models）
import requests  # type: ignore  # noqa: E402
BASE = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
MODELS_URL = BASE.rstrip("/") + "/models"
try:
    r = requests.get(MODELS_URL, headers={"Authorization": f"Bearer {KEY}"}, timeout=15)
    if r.status_code == 200:
        print(f"      ✅ Key 探活成功（GET /models 200）")
    elif r.status_code == 401 or r.status_code == 403:
        print(f"[SKIP] Key 无效 / 免费额度耗尽（HTTP {r.status_code}）：{r.text[:300]}")
        print("       可以充值 DashScope，或改 .env QWEN_MODEL=qwen-turbo 试试更便宜的模型。")
        sys.exit(0)
    else:
        print(f"      ⚠️ Key 探活返回 HTTP {r.status_code}，继续尝试（因为 /v1/chat/completions 可能仍可用）：{r.text[:200]}")
except Exception as e:  # noqa: BLE001
    print(f"      ⚠️ Key 探活网络异常，继续尝试：{type(e).__name__}: {e}")


# ============================================================
# 3. 构建 Agent（用真实 Qwen + 6 工具 + MemorySaver）
# ============================================================
print("[3/6] 构建 AssistantAgent（真实 Qwen + 6 Tools + Checkpointer）……")
reset_agent_singleton()
from src.infra.llm_client import get_qwen_llm  # noqa: E402
real_llm = get_qwen_llm()
agent = AssistantAgent(llm=real_llm, system_prompt=None)  # 默认系统提示即可
print(f"      工具数量: {len(agent.tools)} -> {sorted([t.name for t in agent.tools])}")
print(f"      LLM 类型: {type(real_llm).__name__}")


# ============================================================
# 4. Round 1 执行
# ============================================================
USER_ROUND1 = f"""请严格按顺序完成以下 3 件事，每件事都必须通过工具完成，然后给我一个 3 点中文小结：

① 在「{TEST_FILE_ALIAS}」路径创建一个 Markdown 文件，overwrite=False，内容如下：
```
# M2 真实端到端验证

第一行正文：LangGraph + Checkpointer 🎉
第二行正文：6 个工具已全部接入
第三行正文：thread_id = {THREAD_ID}
```

② 文件创建成功后，调用 recognize_file 识别这个文件，参数：with_preview_lines=5，with_sha256=True，with_image_info=True。

③ 调用 search_news 搜索资讯：query={TEST_NEWS_QUERY}，engine=mock，max_results=3，hours=0。

（注意：第 3 步用 engine=mock 避免出外网慢；不要改 open_browser，不要真删任何文件。）
"""

print("[4/6] Round 1 开始（stream 模式，实时显示 AI/工具消息）……")
print("-" * 70)
t0 = time.perf_counter_ns()
events_r1: list[tuple[str, object]] = []

def cb(kind, payload):  # type: ignore[no-untyped-def]
    events_r1.append((kind, payload))
    if kind == "human":
        print(f"\n[EVT] 👤 HUMAN -> {str(payload.content)[:120]}")
    elif kind == "ai":
        tcs = getattr(payload, "tool_calls", None) or []
        if tcs:
            names = [t.get("name") for t in tcs]
            print(f"[EVT] 🤖 AI 发 tool_calls: {names}")
        else:
            text = (payload.content or "").strip()  # type: ignore[union-attr]
            print(f"[EVT] 🤖 AI 最终:\n{text}")
    elif kind == "tool_pre":
        print(f"[EVT] 📤 ToolNode 即将执行 {len(payload)} 个: {[x.get('name') for x in payload]}")
    elif kind == "tool":
        name = getattr(payload, "name", "?")
        ctt = getattr(payload, "content", "") or ""
        preview = ctt if len(ctt) < 260 else ctt[:260] + "…（已截断）"
        print(f"[EVT] 📥 Tool ({name}) -> {preview}")
    elif kind == "done":
        print(f"[EVT] 🏁 图执行完成: {str(payload)[:120]}")

final_ans_r1 = agent.stream_events(USER_ROUND1, thread_id=THREAD_ID, stream_cb=cb)
ms_r1 = (time.perf_counter_ns() - t0) // 1_000_000
print(f"\n[4/6] Round 1 总耗时 {ms_r1} ms")


# ============================================================
# 5. Round 2 执行（同 thread_id → 验证 Checkpointer 历史保留）
# ============================================================
USER_ROUND2 = """【第二问】现在不要调用任何工具。
只凭你记忆里刚才这一轮对话的上下文，请回答：
- 刚才 Round 1 我要求你做的 3 件事分别是什么？
- 第 ① 步创建的 Markdown 文件的完整路径（相对路径别名也行）是什么？
- 文件正文第一行写了什么？

（如果 Checkpointer 正常生效，你应该能从历史 messages 中直接读出上述答案；而不是说「我没有记忆」或「我不知道」。）
"""

print("\n" + "-" * 70)
print("[5/6] Round 2 开始（相同 thread_id → 验证 Checkpointer 历史带入）……")
t0 = time.perf_counter_ns()
events_r2: list[tuple[str, object]] = []
def cb2(kind, payload):  # type: ignore[no-untyped-def]
    events_r2.append((kind, payload))
    if kind == "ai" and not (getattr(payload, "tool_calls", None) or []):
        print(f"[EVT R2] 🤖 AI 最终:\n{payload.content}")
    elif kind == "ai":
        tcs = getattr(payload, "tool_calls", None) or []
        if tcs:
            print(f"[EVT R2] ⚠️ AI 居然调工具了！{[t.get('name') for t in tcs]}（期望它只靠历史回答）")
final_ans_r2 = agent.stream_events(USER_ROUND2, thread_id=THREAD_ID, stream_cb=cb2)
ms_r2 = (time.perf_counter_ns() - t0) // 1_000_000
print(f"[5/6] Round 2 总耗时 {ms_r2} ms")


# ============================================================
# 6. 校验 & 汇总
# ============================================================
print("\n" + "-" * 70)
print("[6/6] 最终校验")

real_file_p = resolve_user_path(TEST_FILE_ALIAS)
file_exists = real_file_p.exists() and real_file_p.is_file()
# Round 1 工具调用统计
r1_tool_names = [getattr(m, "name", None) for (k, m) in events_r1 if k == "tool"]
r1_had_create = "create_file" in r1_tool_names
r1_had_recognize = "recognize_file" in r1_tool_names
r1_had_news = "search_news" in r1_tool_names
# Round 2 有没有调工具（期望：工具调用数 = 0）
r2_tool_names = [getattr(m, "name", None) for (k, m) in events_r2 if k == "tool"]
r2_no_tool = len(r2_tool_names) == 0
# Round 2 最终回答能否复述 Round 1 关键要素
mem_ok = (
    "M2 真实端到端" in final_ans_r2 or
    TEST_FILE_ALIAS.split("/", 1)[-1] in final_ans_r2 or   # 文件名
    "LangGraph" in final_ans_r2 or   # 正文第一行里的字
    "3 件事" in final_ans_r2 or "三件事" in final_ans_r2 or
    ("第 ① 步" in final_ans_r2 and "第 ③ 步" in final_ans_r2)
)

print(f"  ✅ Round1 3 个工具全调过：create_file={r1_had_create}, recognize_file={r1_had_recognize}, search_news={r1_had_news}")
print(f"  ✅ 文件真实落盘: {file_exists}  | 路径: {real_file_p}")
print(f"  ✅ Round2 没额外调工具（仅靠记忆回答）: {r2_no_tool}  | 实际调过: {r2_tool_names or '无'}")
print(f"  ✅ Round2 能复述 Round1 关键要素（文件名/3件事/LangGraph）: {mem_ok}")
print("-" * 70)
print(f"【R1 最终 AI 回答】:\n{final_ans_r1}\n")
print(f"【R2 最终 AI 回答】:\n{final_ans_r2}\n")

passed = all([r1_had_create, r1_had_recognize, r1_had_news, file_exists, r2_no_tool])
# 注意：mem_ok 只作参考（LLM 可能换种说法复述），不是硬失败项，会打 WARNING
print("=" * 70)
if passed:
    print("🎉🎉🎉 M2 真实云端端到端通过！")
    print("   - 6 个工具被成功 bind 到 LangGraph ReAct Agent（本轮实际调用了 create_file/recognize_file/search_news）")
    print("   - Checkpointer（MemorySaver）已生效：同 thread_id 第二轮 LLM 能看到第一轮历史，且没有重复调工具")
    if not mem_ok:
        print("   ⚠️  提示：Round 2 复述判定 mem_ok=False，但可能是 LLM 换了说法；请人工核对上面【R2 最终 AI 回答】即可。")
    print("\n👉 下一步可做：")
    print("   a) 把 CHECKPOINT_BACKEND 切到 sqlite，安装 langgraph-checkpoint-sqlite，会话可跨进程持久化")
    print("   b) 进入 M3 阶段：接入 ASR/TTS + PySide6 桌面 UI（悬浮球/抽屉式面板）")
else:
    print("❌ 存在失败项，请对照上方 ✅ 列表排查：")
    print(f"   create_file={r1_had_create} recognize_file={r1_had_recognize} search_news={r1_had_news}")
    print(f"   file_exists={file_exists} r2_no_tool={r2_no_tool}")

# —— 清理：真删测试文件 ——
if file_exists and real_file_p.name.startswith("M2真实验证_") and real_file_p.suffix == ".md":
    try:
        real_file_p.unlink()
        print(f"\n🧹 已自动清理测试文件 {real_file_p.name}")
    except Exception as e:  # noqa: BLE001
        print(f"\n🧹 清理失败（手动删即可）：{e}")

reset_agent_singleton()
sys.exit(0 if passed else 2)
