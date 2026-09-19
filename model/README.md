# 冻结模型

`tiny_mlp_mfcc92_edge_robust.npz` 是 v10 部署模型：

- 架构：92→64→5
- 参数：6,277
- SHA256：`5382765b0aeb0c0aa9f96f7f78b26728acfcd20efd35ac7b045637889b50df62`

OpenVela使用的C权重位于 `app/audiodetect/model_weights.h` 和 `app/audiodetect/model_weights_int8.h`。

`tiny_mlp_mfcc92_robust_simulator.npz` 是鲁棒性增强后的模拟器专用模型，结构、参数量与 INT8 方案相同：

- 架构：92→64→5
- 参数：6,277
- SHA256：`c96333aaae66d324b8b8cec48ffb0ec981ac4e975d3cf9a7d4a2fdba4307c97e`

其 INT8 C 权重由 `training/export_simulator_int8.py` 生成到 `simulator/generated/model_weights_int8.h`，只用于主机 C 模拟器，不覆盖开发板权重。
