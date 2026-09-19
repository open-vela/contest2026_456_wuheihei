---
name: openvela-audio-validation
description: Validate and reproduce the Audio Sentinel v11 training evidence, model checksums, C host simulator, and OpenVela deployment boundary. Use when asked to audit, reproduce, or explain this repository. Do not use it to claim new board measurements or to change the board deployment.
---

# OpenVela Audio Validation

Use this workflow for prompts such as “复现 Audio Sentinel v11”, “核对 OpenVela 音频模型”, or “验证音频事件部署证据”.

## Scope

Validate only the files and evidence already present in this repository. Keep the three execution layers separate:

1. Python training and evaluation.
2. Native C host simulator inference.
3. OpenVela development-board integration.

Never describe host-simulator results as real-board measurements.

## Workflow

1. Record the current branch, commit, and working-tree status. Do not discard unrelated changes.
2. Run `scripts/verify_submission.sh` from the repository root.
3. Re-export the C weight headers with `training/export_to_c.py` when the model file changed, and confirm the three generated headers agree.
4. Build and run the C host simulator with `scripts/build_robust_c_simulator.sh` when the current platform supports it.
5. Re-run Python evaluation only when the required dataset or cached features are available. Use the existing scripts and the v11 model; do not silently substitute another dataset or an older result.
6. Treat `results/python_cv/` as the source for model-accuracy claims.
7. Inspect `docs/OpenVela端功能边界.md` before describing board features or limitations.
8. Apply or build the board overlay only when the user explicitly requests it. Do not flash hardware or remove build environments as part of validation.

## Required output

Report these items in order:

- Provenance: branch, commit, model filename, and model SHA-256.
- Checks: every command run and whether it passed or failed.
- Model evidence: Float and INT8 overall accuracy, macro recall, and per-class recall from the checked-in results.
- C simulator: build status and tested sample count.
- OpenVela status: implemented functions, integration boundary, and unverified measurements.
- Limitations: missing dataset, unavailable hardware, platform differences, or any result that could not be reproduced.

## Evidence rules

- Never invent a metric, board result, timing, RAM figure, or power figure.
- Keep Float, INT8, simulator, and board statements visibly distinct.
- If evidence conflicts, quote the conflicting paths and stop short of choosing a preferred value without further verification.
- Do not retrain, delete build artifacts, deploy to a board, or alter the model or weight headers unless the user asks for that change explicitly.

## Repository references

- `README.md`: project overview and quick start.
- `docs/阶段2_复现实验与指标.md`: reproducibility and metric interpretation.
- `docs/OpenVela端功能边界.md`: implemented and unimplemented board behavior.
- `results/README.md`: evidence layout.
- `training/README.md`: Python pipeline.
- `app/audiodetect/README.md`: C/OpenVela application.
