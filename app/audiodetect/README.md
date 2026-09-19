# audiodetect：OpenVela 端侧应用

本目录是 Audio Sentinel v10 的实际 OpenVela/NuttX 应用源码。赛事 manifest 将它映射到 OpenVela 工作区的 `apps/audiodetect`。

## 当前实现

- 五类音频事件：background、cough、glass_break、baby_cry、dog_bark。
- 92维时域/MFCC/Delta特征。
- TinyMLP `92→64→5`，6,277参数。
- Float32与INT8权重推理；实时UI路径使用INT8。
- 三秒麦克风采集，每段分析三个一秒窗口。
- 静音门限 `2.0e-5`、事件门限 `0.80`。
- LVGL触摸界面、自动启动偏好、事件计数和最近三条事件。
- WAV命令行推理、噪声演示、关键词路径和可选HTTP/MQTT上报代码。

## 主要文件

| 文件 | 作用 |
|---|---|
| `audiodetect_main.c` | 命令入口、WAV推理、静音门限、告警和子命令分发 |
| `audiodetect_ui.c` | LVGL界面、麦克风三秒采集和后台工作线程 |
| `audio_classifier.c` | 三个一秒窗口的INT8分类与事件选择 |
| `feature_extractor.c` | 92维C特征提取 |
| `model_infer.c` | Float32 TinyMLP前向推理 |
| `model_infer_int8.c` | INT8权重存储、运行时反量化推理 |
| `model_weights*.h` | 当前冻结模型生成的C权重 |
| `wav_reader.c` | 16 kHz、16-bit、单声道PCM WAV读取 |
| `keyword_detect.c` | MFCC+DTW关键词路径及MFCC统计复用 |
| `alarm_report.c` | 可选HTTP/MQTT告警代码 |
| `Kconfig`、`Makefile`、`CMakeLists.txt` | OpenVela构建集成 |

## 运行

启用以下配置：

```text
CONFIG_AE_AUDIODETECT=y
CONFIG_AE_AUDIODETECT_UI=y
CONFIG_AE_AUDIODETECT_UI_AUTOSTART=y
CONFIG_GRAPHICS_LVGL=y
```

命令示例：

```text
nsh> audiodetect ui
nsh> audiodetect /data/test.wav --int8 --debug
nsh> audiodetect robust /data/test.wav
nsh> audiodetect keyword /data/keyword.wav
```

实时UI使用 `arecord -D default -r16000 -f16 -c1 -d3 /data/aed_live.wav`。它不是无间隙流式PCM管线；边界和未验证项请见仓库根目录README。

