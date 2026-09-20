# R528S3 Gemini S1 板级配置

本目录保存 v11 构建时使用的板级增量（不含厂商 BSP 全量拷贝）。

- `r528s3-gemini-s1/configs/nsh/defconfig`：v11 镜像实际使用的板级配置，启用 R528 Audio、PCM、LCD（ILI9341，320×240）、GT911 触摸、LVGL 和 audiodetect。
- `r528s3-gemini-s1/src/etc/init.d/rcS.nsh`：硬件初始化后自动启动 `audiodetect ui`。
- `r528s3-gemini-s1/reference/openvela_active.config`：冻结时的完整活动配置，仅供审计。
- `r528s3-gemini-s1/reference/nuttx_savedefconfig`：构建后保存的最小配置参考，与 `configs/nsh/defconfig` 逐字节相同。

使用工作区根目录的 `contest2026_456_wuheihei/scripts/apply_board_overlay.sh` 将增量复制到对应 vendor 目录，再进行构建：

```bash
./contest2026_456_wuheihei/scripts/apply_board_overlay.sh .
./nuttx/tools/configure.sh -e \
  vendor/allwinnertech/boards/r528/r528s3-gemini-s1/configs/nsh
make -C nuttx EXTRAFLAGS="-Wno-cpp -Wno-deprecated-declarations" -j16
```

## 显示面板

v11 镜像使用 **ILI9341（320×240）** 配置，`audiodetect_ui.c` 的界面按 320×232 布局，与之匹配。

厂商 BSP 自带的 `configs/nsh/defconfig` 选择的是 7 英寸 **T070S140B（1024×600）** MIPI 面板，与 v11 镜像实际使用的配置不一致；本仓库的 `configs/nsh/defconfig` 已替换为构建镜像时实际生效的 ILI9341 版本，因此 `apply_board_overlay.sh` + `configure.sh` 可以复现当前镜像的板级配置。

`app/audiodetect/Kconfig` 中 `AE_AUDIODETECT_UI` 的帮助文字描述的是同一块 320×240 界面。
