# 项目模型

`tiny_mlp_mfcc92_robust_simulator.npz` 是项目当前模型：

- 架构：92→64→5
- 参数：6,277
- SHA256：`c96333aaae66d324b8b8cec48ffb0ec981ac4e975d3cf9a7d4a2fdba4307c97e`

模型采用泄漏安全的混合增强训练（时间平移、音量缩放、时间拉伸、高斯噪声，并混合同训练折真实背景），
五折指标见 `results/simulator_robust_hybrid/metrics.json`。

主机 C 模拟器使用的 INT8 C 权重由 `training/export_simulator_int8.py` 生成到
`simulator/generated/model_weights_int8.h`。

`tiny_mlp_mfcc92_edge_robust.npz` 为早期版本模型，保留用于对照。
