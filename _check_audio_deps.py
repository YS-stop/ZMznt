"""快速校验 4 个音频依赖是否 import 成功。"""
try:
    import sounddevice as sd
    print(f"  [OK] sounddevice {sd.__version__}")
except Exception as e:
    print(f"  [FAIL] sounddevice: {type(e).__name__}: {e}")

try:
    import numpy as np
    print(f"  [OK] numpy {np.__version__}")
except Exception as e:
    print(f"  [FAIL] numpy: {type(e).__name__}: {e}")

try:
    import soundfile as sf
    print(f"  [OK] soundfile {sf.__version__}")
except Exception as e:
    print(f"  [FAIL] soundfile: {type(e).__name__}: {e}")

try:
    import pyaudio
    pa = pyaudio.PyAudio()
    n = pa.get_device_count()
    print(f"  [OK] pyaudio installed, audio devices={n}")
except Exception as e:
    print(f"  [SKIP] pyaudio 未装（sounddevice 优先，可忽略）: {type(e).__name__}: {e}")
