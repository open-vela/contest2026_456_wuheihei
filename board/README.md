# R528S3 Gemini S1 板级配置

本目录保存 v10 构建时使用的板级增量，而不是复制整个厂商 BSP。

- `r528s3-gemini-s1/configs/nsh/defconfig`：启用Audio、PCM、LCD、触摸、LVGL和audiodetect。
- `r528s3-gemini-s1/src/etc/init.d/rcS.nsh`：硬件初始化后自动启动 `audiodetect ui`。
- `r528s3-gemini-s1/reference/openvela_active.config`：冻结时的完整活动配置，仅供审计。
- `r528s3-gemini-s1/reference/nuttx_savedefconfig`：构建后保存的最小配置参考。

使用工作区根目录的 `contest2026_456_wuheihei/scripts/apply_board_overlay.sh` 将增量复制到对应vendor目录，再进行构建。

