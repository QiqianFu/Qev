# Gameplay recordings / 游戏录屏

The [English README](../README.md) and [中文 README](../README.zh-CN.md) show looping GIF previews of recorded decision replays. Each preview links to an MP4 version. The interface retains the research name, BranchKev.

中英文 README 展示决策回放录屏的循环 GIF，点击可打开 MP4 版本。录屏界面保留研究阶段的名称 BranchKev。

| Recording / 录屏 | GIF | MP4 |
|---|---|---|
| Snake / 贪吃蛇 | [snake.gif](../assets/demos/snake.gif) | [snake.mp4](../assets/demos/snake.mp4) |
| Crafter / 生存建造 | [crafter.gif](../assets/demos/crafter.gif) | [crafter.mp4](../assets/demos/crafter.mp4) |

## Visible run details / 画面中的运行信息

- **Snake:** the interface identifies `BranchKev 9B · cleaned-v2-aqua`, seed 101, and 119 recorded decisions. The replay shows the board after each action alongside the selected action and candidate probabilities.
- **Crafter:** the interface identifies `BranchKev 9B · step2327`, seed 77, and 40 steps. The replay shows the environment after each action alongside the current goal, selected action and probabilities.

贪吃蛇画面标注 `BranchKev 9B · cleaned-v2-aqua`、种子 101、119 次已记录决策；Crafter 画面标注 `BranchKev 9B · step2327`、种子 77、40 步。两段均展示动作执行后的环境以及决策信息。

These individual replays illustrate behavior; benchmark results and evaluation settings are documented in [evaluation.md](evaluation.md).

单局回放用于展示行为，评测结果与设置见 [evaluation.md](evaluation.md)。

## Media settings / 媒体设置

Both previews use an 800 × 680 canvas at 20 fps. The complete recorded frame is scaled proportionally and padded to keep the README columns aligned. GIFs loop indefinitely; MP4s use H.264 with fast-start metadata and no audio. Original recordings are kept outside the repository.

两段预览均为 800 × 680、20 fps，完整画面按比例缩放并补边，保持 README 两列对齐。GIF 无限循环；MP4 使用 H.264 编码、支持渐进播放且无音轨。原始录屏保留在仓库外。
