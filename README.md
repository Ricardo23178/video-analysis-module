# 本地视频分析模块

把本地视频转成「模型可读的证据材料」的独立 Python 模块：FFprobe 读取媒体信息 → 均匀抽帧 → 提取音轨 → 解析字幕或可选 Whisper 转写 → 输出统一的分析清单 `analysis.json`。

**为什么它独立存在：** 模块最初是为一个视频解析工具开发的「下载后处理」环节，但设计上不依赖该工具的下载器、Codex、yt-dlp 或任何 MCP 服务——只要有一个本地视频文件就能用，标准库即可完成媒体调度，`faster-whisper` 为可选依赖。

## 核心能力

| 能力 | 实现方式 |
| --- | --- |
| 媒体信息 | `ffprobe` 读取时长、分辨率、编码、音轨延迟 |
| 画面采样 | `ffmpeg` 按时间均匀抽帧，每帧保留 `time_seconds` |
| 音轨提取 | 统一转 16 kHz 单声道 WAV，供转写使用 |
| 字幕解析 | SRT / VTT / ASS 统一经 FFmpeg 转换后解析，保留时间戳 |
| 语音转写 | 可选 `faster-whisper`；依赖、模型或结果为空时明确降级 |
| 统一清单 | `analysis.json`（`schema_version=1`）描述所有产物的路径与状态 |

## 四个设计决策

这些不是随手写的，每一个都对应一类具体的失败场景：

**1. 契约优先，而不是返回裸数据。**
`analysis.json` 带 `schema_version`，字段固定为 `status` / `degraded_modalities` / `errors` / `media` / `audio` / `frames[]` / `transcript`。调用方只依赖契约，不管内部换了哪种抽帧策略都不会被影响。对应的验证在 `tests/test_analysis_contract.py`——**测的是输出契约，不是实现细节**。

**2. 源文件不可变。**
分析一个视频不应该改动它。测试里读取分析前后的源文件字节并断言相等，把这条约束钉死在回归测试里，而不是写在文档里靠自觉。

**3. 部分失败不等于整体失败。**
转写不可用、字幕缺失、音轨异常，都不抛异常中断流程，而是降级：`status` 标记可用性，`degraded_modalities` 记录具体缺失的模态，`errors` 给出原因。调用方能拿到「画面可用但转写缺失」这种中间态，而不是一个笼统的报错。

**4. 输出目录必须尚不存在。**
重复分析同一个视频时，宁可让调用方显式换目录，也不静默覆盖上一次的结果——误覆盖比报错更难被发现。

另外，`video_analysis.py` 里有一段 Windows 下 CUDA DLL 的加载处理：`faster-whisper` 依赖 NVIDIA wheel 提供的 `cublas` / `cudnn` / `cuda_nvrtc`，这些 DLL 在 Windows 上不会被自动发现。代码里用 `os.add_dll_directory` 注册并保留句柄（避免被 GC 后目录失效），同时把路径追加进 `PATH`，让子进程也能找到。

## 快速开始

依赖：Python 3.11+、完整 FFmpeg（`ffmpeg` 与 `ffprobe` 都在 PATH 中）。

```bash
python video_analysis.py "D:/video/source.mp4" --output "D:/analysis/task001"
```

作为库调用：

```python
from pathlib import Path
from video_analysis import analyze_video

result = analyze_video(
    "D:/video/source.mp4",
    "D:/analysis/task001",   # 必须是不存在的目录
    max_frames=12,
)
print(result["status"], result["analysis_json"])
```

可选参数：`subtitle_path` 指定字幕、`whisper_model="small"` 指定转写模型、`transcribe=False` 关闭转写（已有字幕仍会读取）。

## 测试

```bash
python -m unittest discover -s tests -p test_analysis_contract.py -v
```

测试用 `ffmpeg` 的 `lavfi` 现场合成一段测试视频（纯色画面 + 正弦音轨）和一份 SRT，**不访问任何视频平台、不下载模型、不依赖外部素材**。缺少 FFmpeg 时跳过并给出原因。

```
Ran 1 test in 2.096s
OK
```

## 协作说明

- 本模块由 **朱思兴** 独立开发，以 MIT 许可分发。
- 上游下载器（抖音 / B站链接解析）由协作者提供；两者集成的完整版本见 [RanT711/video-parser](https://github.com/RanT711/video-parser)。
- 集成后的形式为：下载器保存并校验原视频 → 本模块生成分析材料 → 由调用方的大模型读取材料生成摘要。

## 许可

MIT，见 [LICENSE](LICENSE)。
