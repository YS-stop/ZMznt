import re

_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_MD_ITALIC = re.compile(r'(?<!\*)\*(?!\*)(.+?)\*(?!\*)')
_MD_UNDERLINE = re.compile(r'__(.+?)__')
_MD_HEADING = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_MD_CODE_BLOCK = re.compile(r'```[\s\S]*?```')
_MD_INLINE_CODE = re.compile(r'`([^`]+)`')
_TOOL_REF = re.compile(r'\[([a-zA-Z_][\w]*)\]')
_ARROW = re.compile(r'[→←↑↓↔⇒⇐⇑⇓➤➜▶◀]')
_EMOJI = re.compile(
    '['
    '\U00002700-\U000027BF\U0001F300-\U0001F9FF'
    '\U00002600-\U000026FF\U0001FA00-\U0001FA6F'
    '\U0001FA70-\U0001FAFF\u200d\ufe0f'
    ']',
    flags=re.UNICODE,
)
_EXCESSIVE_NL = re.compile(r'\n{3,}')
_LIST_MK = re.compile(r'^(\s*)[-*•]\s+', re.MULTILINE)
_ORPHAN = re.compile(r'[*_`]{2,}')


def clean(raw, for_tts=False):
    t = raw
    t = _MD_CODE_BLOCK.sub(' ', t)
    t = _MD_BOLD.sub(r'\1', t)
    t = _MD_ITALIC.sub(r'\1', t)
    t = _MD_UNDERLINE.sub(r'\1', t)
    t = _MD_HEADING.sub('', t)
    t = _MD_INLINE_CODE.sub(r'\1', t)
    t = _TOOL_REF.sub('' if for_tts else r'\1', t)
    t = _ARROW.sub(',' if for_tts else '→', t)
    t = _EMOJI.sub('', t)
    t = _LIST_MK.sub(r'\1', t)
    t = _ORPHAN.sub('', t)
    t = _EXCESSIVE_NL.sub('\n\n', t)
    t = re.sub(r'[ \t]+', ' ', t)
    return t.strip()


raw = (
    "\u26a0\ufe0f **安全提醒**：出于隐私保护原则，我**不能**接收、存储或传输输入您的手机号、验证码、密码等任何敏感信息**。"
    "这是系统级安全设计，也是对您账户安全的必要保障。\n\n"
    "- **为什么不能代输？**\n"
    "- 浏览器登录框可能被恶意网站仿冒（钓鱼风险）；\n"
    "- 手机号+验证码=账户实际控制权，AI代输等于主动交出凭证；\n"
    "- 即使是本地运行的助手，也不应接触明文身份凭证（符合最小权限原则）。\n\n"
    "\u2705 **安全又便捷的替代方案**：\n"
    "1. **自己操作（推荐）**：\n"
    "   \u2192 点击网页右上角「登录」\u2192 选择「手机号登录」\n"
    "   \u2192 您手动输入手机号 \u2192 点击「获取验证码」\u2192 输入收到的6位短信码 \u2192 完成登录。\n"
    "   （整个过程只需10秒，且全程在您掌控中）\n\n"
    "2. **如需辅助**：\n"
    "   - 我可帮您**自动点击「获取验证码」按钮**（调用 `browser_click`）；\n"
    "   - 或**自动填写您口述的验证码**（您说数字，我立即即填，不留存）；\n"
    "   - 或**跳转到抖音App下载页**，扫码一键登录（更安全）。\n\n"
    "请告诉我您希望采用哪种方式？我会严格按您的指令执行 \u2705"
)

tts = clean(raw, for_tts=True)
print("=== TTS 播报版 ===")
print(tts)
print()

ui = clean(raw, for_tts=False)
print("=== UI 展示版 ===")
print(ui)
print()

print("=== 噪音残留检查（TTS版）===")
noise_list = ["**", "*", "\u2705", "\u26a0\ufe0f", "\u2192", "[browser_click]", "`"]
for n in noise_list:
    found = n in tts
    status = " 仍存在" if found else " 已清除"
    print(f"  {repr(n)}: {status}")
