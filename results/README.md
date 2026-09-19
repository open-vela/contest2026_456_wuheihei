# 结果目录

- `python_cv/`：v10 部署模型的严格五折交叉验证和噪声鲁棒性结果，统计单位为源录音。
- `simulator_robust_hybrid/`：鲁棒性增强模拟器模型的五折与噪声鲁棒性结果。
- `c_simulator/`：当前 C 推理链路的便利样本冒烟测试，不作为独立准确率。
- `evidence_checksums.json`：阶段 1–2 证据文件校验值。

准确率来自Python五折评测；本目录不包含真机准确率、延迟、内存或功耗结果。
