"""pycaw API 探究：为什么 AudioDevice 没有 Activate。"""
from pycaw.pycaw import AudioUtilities
import inspect

d = AudioUtilities.GetSpeakers()
print(f"  类型: {type(d)}  ->  {type(d).__mro__}")
print(f"  public methods: {[m for m in dir(d) if not m.startswith('_')]}")
print()
print("  GetSpeakers 源码:")
try:
    print(inspect.getsource(AudioUtilities.GetSpeakers))
except Exception as e:
    print(f"  (inspect 失败：{e})")
    print("  直接用 AudioUtilities 方法列表：")
    print([m for m in dir(AudioUtilities) if not m.startswith("_")])

print()
print("  试试用 AudioUtilities.GetAllDevices()：")
try:
    devs = AudioUtilities.GetAllDevices()
    print(f"  设备数: {len(devs)}")
    if devs:
        print(f"  第一个：{devs[0]}  类型={type(devs[0])}  public attrs: {[m for m in dir(devs[0]) if not m.startswith('_')][:20]}")
except Exception as e:
    print(f"  GetAllDevices 失败：{e}")

print()
print("  试试 IAudioEndpointVolume 接口直接？pycaw 文档常用方法：")
try:
    # 最新 pycaw 正确做法：用 AudioUtilities.GetAllDevices + 每个设备 property
    # 或者官方推荐 AudioUtilities.GetSpeakers 返回的就是 IMMDevice
    # 但需要用 cast 到 POINTER，用 Activate 可能是 dev.Activate() 要走 comtypes
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import IAudioEndpointVolume
    # 新版可能：直接 IMMDevice.Activate 走属性名不同？试试叫 QueryInterface 或 ActivateInterface？
    print("  dir(d) 全部属性（含下划线），搜含 Activate 的：")
    for m in dir(d):
        if "act" in m.lower() or "interf" in m.lower() or "query" in m.lower() or "cast" in m.lower():
            print(f"    - {m}")
except Exception as e:
    print(f"  失败：{e}")
