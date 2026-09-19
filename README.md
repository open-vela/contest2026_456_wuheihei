# Audio Sentinel：基于 OpenVela 的离线音频事件检测系统

> 2026 首届 openvela AI 硬件开发者大赛 · 队伍「呜嘿嘿」<br>
> 团队成员：李炳霖、吴安琪<br>
> 选题方向：AI 硬件产品创新（端侧离线音频智能）

Audio Sentinel 面向家庭安全、看护和隐私敏感场景，在 OpenVela 设备本地完成麦克风采集、音频特征提取、轻量模型推理和可视化告警。系统识别 `background`、`cough`、`glass_break`、`baby_cry`、`dog_bark` 五类声音；默认推理不依赖云端，也不上传原始音频。

本仓库内容对应 v11 版本的应用源码、模型与评测结果。

## 一、作品亮点

- **真实数据训练**：使用 ESC-50 中四类目标事件，并从其余类别抽取背景样本。
- **严格五折验证**：按 ESC-50 官方折划分，同一源录音切出的窗口不会跨训练集和测试集。
- **轻量端侧模型**：92 维时域/MFCC/Delta 特征，TinyMLP `92→64→5`，共 6,277 个参数。
- **INT8 权重存储**：估算从 25,844 字节降至 7,029 字节，缩减 72.80%。
- **OpenVela 真板集成**：已接入 R528S3 音频、LCD、触摸、LVGL 和开机自启动。
- **本地交互**：实时显示状态、类别、置信度、能量、窗口数、告警数和最近事件。
- **鲁棒性策略**：训练阶段使用混合增强（时间平移、音量缩放、时间拉伸、高斯噪声，并混合同训练折真实背景）；设备端增加静音门限和 0.80 事件阈值。
- **强噪声鲁棒性**：10 dB / 5 dB 下宏平均召回率为 80.57% / 74.37%，强噪声下五类事件均可检出。
- **结果随源码提交**：训练脚本、模型、混淆矩阵、各类别指标和 C 推理测试结果均位于仓库内。

## 二、系统架构

```text
Python 训练端
ESC-50 → 1 s 分窗 → 92维特征 → 五折训练/验证 → NPZ模型 → Float/INT8 C头文件
                                                        │
                                                        ▼
OpenVela R528S3 端
麦克风 → arecord 3 s → 三个1 s窗口 → C特征提取 → INT8 TinyMLP
                                                      │
                                   静音门限 + 0.80事件阈值
                                                      │
                                                      ▼
                              LVGL界面 / 本地事件计数 / 最近告警
```

三个运行端严格区分：

| 运行端 | 职责 | 本仓库中的位置 |
|---|---|---|
| Python 训练与评测 | 数据准备、五折训练、噪声鲁棒性、模型导出 | `training/` |
| Linux 主机 C 模拟器 | 编译 C 特征、三秒决策逻辑和 INT8 权重 | `simulator/`、`scripts/build_robust_c_simulator.sh` |
| 真实 OpenVela 开发板 | 麦克风、LCD、触摸、LVGL 和自动启动 | `app/audiodetect/`、`board/` |

主机 C 模拟器与开发板共用同一套 92 维 C 特征实现。模型指标来自 ESC-50 离线五折评测，五折评测与模拟器验证使用同一模型。开发板端已完成代码集成与镜像构建，真机未做准确率、延迟、内存与功耗量化。

## 三、仓库结构

```text
contest2026_456_wuheihei/
├── app/audiodetect/                 # OpenVela/NuttX 应用源码与权重头文件
├── board/r528s3-gemini-s1/          # 当前真板 defconfig、rcS 和冻结配置
├── training/                        # 当前 Python 数据、训练、评测与导出源码
├── simulator/                        # 主机 C 模拟器 INT8 推理与生成权重
├── model/                            # 项目模型 NPZ
├── results/
│   └── python_cv/                    # 五折指标、混淆矩阵与噪声鲁棒性结果
├── docs/                             # 实验口径、功能边界与提交材料
├── scripts/                          # 板级 overlay、C 模拟器与仓库自检脚本
├── logs/                             # 官方格式 AI Coding 日志目录
└── contest2026_456_wuheihei.xml      # repo manifest，映射应用到 apps/audiodetect
```

## 四、拉取完整 OpenVela 工程

```bash
repo init -u https://github.com/open-vela/contest2026_456_wuheihei \
  -b dev-ai-contest-2026 -m contest2026_456_wuheihei.xml
repo sync -c -j8
```

同步完成后，队伍仓库位于工作区的 `contest2026_456_wuheihei/`，应用通过 manifest 映射到外层 `apps/audiodetect/`。

可先执行仓库完整性检查：

```bash
cd contest2026_456_wuheihei
./scripts/verify_submission.sh
```

## 五、Python 训练与交叉验证

### 5.1 环境

```bash
cd contest2026_456_wuheihei/training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 5.2 准备 ESC-50

将 ESC-50 放到 `training/data/datasets/ESC-50/`，目录中应包含 `audio/` 和 `meta/esc50.csv`。数据集本身不提交到本仓库。

```bash
python import_esc50.py \
  --esc50_dir data/datasets/ESC-50 \
  --output_dir data/raw_esc50_real \
  --background_per_class 5 --seed 42 --overwrite

python prepare_data.py \
  --raw_dir data/raw_esc50_real \
  --output_dir data/processed_esc50_real --overwrite
```

### 5.3 构建当前 92 维特征与增强缓存

```bash
python extract_features_mfcc.py \
  --labels data/processed_esc50_real/labels.csv \
  --out results/esc50_mfcc92_features.npz --include-delta

python build_robust_augmented_cache.py \
  --base-cache results/esc50_mfcc92_features.npz \
  --feature-mode mfcc92 --profile hybrid --copies 4 --seed 2026 \
  --out results/esc50_mfcc92_robust_augmented_features.npz
```

### 5.4 重新训练和五折验证

```bash
python train_edge_mfcc.py \
  --clean-cache results/esc50_mfcc92_features.npz \
  --aug-cache results/esc50_mfcc92_robust_augmented_features.npz \
  --models-dir models/edge_mfcc_cv \
  --final-model models/tiny_mlp_mfcc92_robust_simulator.npz \
  --report-dir results/python_cv \
  --epochs 120 --hidden-dim 64 --class-weight-power 0.5 \
  --aug-fraction 0.5 --event-aug-fraction 0.75 --background-aug-fraction 0.25 \
  --batch-size 64 --lr 0.003 --weight-decay 0.0001 --seed 42
```

模型 SHA256：

```text
c96333aaae66d324b8b8cec48ffb0ec981ac4e975d3cf9a7d4a2fdba4307c97e
```

## 六、Linux 主机 C 模拟器

主机 C 模拟器复用 `app/audiodetect/` 中的 C 特征提取与三秒决策逻辑，配合 `simulator/` 的 INT8 推理实现，编译出可独立运行的验证程序。

```bash
python training/export_simulator_int8.py \
  --model model/tiny_mlp_mfcc92_robust_simulator.npz

./scripts/build_robust_c_simulator.sh
./out/robust-simulator/audiodetect_robust_sim /path/to/16k_mono_16bit.wav
```

该入口用于确认 C 推理链路和门限行为；其样本不属于独立测试集，模型指标以 `results/python_cv/metrics.json` 为准。

## 七、真实 OpenVela 开发板构建

目标硬件为润芯微 OpenVela 适配开发板，构建配置对应 R528S3 Gemini S1。

在 `repo sync` 后的 OpenVela 工作区根目录执行：

```bash
./contest2026_456_wuheihei/scripts/apply_board_overlay.sh .

./nuttx/tools/configure.sh -e \
  vendor/allwinnertech/boards/r528/r528s3-gemini-s1/configs/nsh

make -C nuttx EXTRAFLAGS="-Wno-cpp -Wno-deprecated-declarations" -j16
```

当前冻结镜像信息：

- 生成时间：2026-09-19 09:57:09（UTC-04:00）
- 大小：113,143,844 字节
- SHA256：`e4aead1e04c0633e9c9017b6023acfc21d13c3cd4d40d574007dae32f5181db4`

镜像体积为整个 OpenVela 固件大小，编译产物不随仓库提交。

### 7.1 运行方式

启用 `CONFIG_AE_AUDIODETECT_UI_AUTOSTART` 后，板级 `rcS.nsh` 会在硬件服务初始化后等待 3 秒并执行：

```text
audiodetect ui &
```

也可以从 NSH 手动运行：

```text
nsh> audiodetect ui
nsh> audiodetect /data/test.wav --int8 --debug
```

## 八、模型设计

### 8.1 特征

| 特征组 | 维度 | 内容 |
|---|---:|---|
| 时域统计 | 40 | 能量、过零率、平均幅值、峰值、能量变化 × 8种统计量 |
| MFCC | 26 | 13维 MFCC 的均值与标准差 |
| MFCC Delta | 26 | 13维一阶差分的均值与标准差 |
| 合计 | 92 | Python 与 C 端保持相同布局 |

- 采样率：16 kHz
- PCM：16-bit 单声道
- 模型窗口：1 秒
- 帧长：25 ms
- 帧移：10 ms

### 8.2 TinyMLP

```text
92维输入 → z-score → 64维全连接 + ReLU → 5维输出 → Softmax
```

| 指标 | 数值 |
|---|---:|
| 参数量 | 6,277 |
| Float32 参数及归一化数据 | 25,844 B |
| INT8 权重存储估算 | 7,029 B |
| 存储缩减 | 72.80% |

INT8 路径采用权重对称量化与运行时反量化，收益集中在存储空间，不涉及纯整数 MAC 加速。

## 九、可复现指标

评测以源录音为统计单位：390 条源录音切分为 1,950 个一秒窗口，同一源录音的窗口保持在同一折。background 有 230 条源录音，其余四类各 40 条，因此同时给出总体准确率与宏平均召回率。

训练使用混合增强：时间平移、音量缩放、时间拉伸和高斯噪声，并混合同训练折真实背景；背景只从训练折内取，测试折不参与。事件类使用 75% 增强样本，background 使用 25%，增强副本数为 4。

### 9.1 五折总体结果

| 条件 | Float 总体准确率 | Float 宏平均召回率 | INT8 总体准确率 | INT8 宏平均召回率 |
|---|---:|---:|---:|---:|
| Clean | 82.56% | 78.39% | 82.56% | 78.39% |
| 20 dB | 80.00% | 78.76% | 80.00% | 79.17% |
| 10 dB | 79.23% | 80.57% | 79.23% | 80.15% |
| 5 dB | 74.36% | 74.37% | 74.87% | 74.54% |

INT8 与 Float 结果基本一致，未观察到明显精度损失。

### 9.2 Clean 条件各类别召回率

| 类别 | 样本数 | Float Recall | INT8 Recall |
|---|---:|---:|---:|
| background | 230 | 86.96% | 86.96% |
| cough | 40 | 55.00% | 55.00% |
| glass_break | 40 | 85.00% | 85.00% |
| baby_cry | 40 | 85.00% | 85.00% |
| dog_bark | 40 | 80.00% | 80.00% |

### 9.3 5 dB 条件各类别召回率

| 类别 | Float Recall | INT8 Recall |
|---|---:|---:|
| background | 74.35% | 75.22% |
| cough | 47.50% | 47.50% |
| glass_break | 82.50% | 82.50% |
| baby_cry | 90.00% | 87.50% |
| dog_bark | 77.50% | 80.00% |

5 dB 下五类事件均可检出，其中 cough 召回率最低（47.50%）。真机端准确率、延迟、内存和功耗未测量。

## 十、OpenVela 端当前能力

### 已实现

- 16 kHz/16-bit/单声道麦克风三秒分段采集。
- 三个一秒窗口的 C 特征与 INT8 推理。
- 静音能量门限 `2.0e-5` 和事件阈值 `0.80`。
- 五类事件输出。
- LVGL 的 START、STOP、AUTO、状态、类别、置信度、能量和计数显示。
- 最近 3 条事件摘要。
- 自动启动偏好持久化和开机启动。
- 命令行 WAV 推理、噪声演示、关键词路径和可选 HTTP/MQTT 代码路径。

### 未实现或未验证

- 当前通过 `arecord` 分段写 WAV，不是无间隙 PCM 环形缓冲。
- 不支持重叠滑窗、多标签和事件起止时间定位。
- HTTP/MQTT 和关键词识别尚未接入实时 UI 工作线程。
- 没有持久化事件数据库或音频证据归档。
- 没有真机准确率、端到端延迟、峰值内存、功耗和长期稳定性数据。

更完整的边界见 [`docs/OpenVela端功能边界.md`](docs/OpenVela端功能边界.md)。

## 十一、AI Coding 使用说明

AI 工具参与了需求拆解、训练与交叉验证脚本完善、C/Python 一致性排查、板端问题定位、结果复核和提交材料整理；相关改动经过源码检查与运行验证。

日志由赛事官方归集工具导出，目录规则见 [`logs/README.md`](logs/README.md)。

## 十二、许可证和数据来源

- 项目源码采用 Apache License 2.0，见 [`LICENSE`](LICENSE)。
- ESC-50 数据不随本仓库分发，使用时遵循其 CC BY-NC-SA 3.0 许可。
- 数据和第三方说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 十三、证据索引

- [`results/python_cv/metrics.json`](results/python_cv/metrics.json)：五折指标、混淆矩阵与噪声鲁棒性原始结果。
- [`model/tiny_mlp_mfcc92_robust_simulator.npz`](model/tiny_mlp_mfcc92_robust_simulator.npz)：项目模型 NPZ。
- [`app/audiodetect/model_weights.h`](app/audiodetect/model_weights.h)、[`model_weights_int8.h`](app/audiodetect/model_weights_int8.h)：开发板权重头文件，由 `training/export_to_c.py` 从项目模型生成。
- [`simulator/`](simulator/)：主机 C 模拟器 INT8 推理实现与生成权重。
- [`docs/阶段2_复现实验与指标.md`](docs/阶段2_复现实验与指标.md)：完整实验口径和解释。
- [`docs/OpenVela端功能边界.md`](docs/OpenVela端功能边界.md)：OpenVela 端能力边界。
