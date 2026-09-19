# Python 训练与评测源码

本目录对应 v10 的当前训练管线，依赖 Python 3.10+ 和 NumPy。

核心流程：

1. `import_esc50.py`：选择cough、glass_break、baby_cry、dog_bark，并抽取其余类别作为background。
2. `prepare_data.py`：转换为16 kHz单声道一秒窗口，同时保留源录音路径。
3. `extract_features_mfcc.py`：生成92维特征缓存。
4. `build_robust_augmented_cache.py`：生成鲁棒性增强缓存。
5. `train_edge_mfcc.py`：按官方折、按源录音分组完成五折训练和噪声评测。
6. `export_to_c.py`、`quantize_int8.py`：导出Float和INT8 C权重。

数据集和特征缓存不提交；完整命令见仓库根目录README。

