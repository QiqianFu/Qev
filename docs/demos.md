# Gameplay recordings / 游戏录屏

The README opens with two side-by-side recording slots. Both currently contain explicitly labelled placeholders; no Qev Snake or Crafter performance is claimed by these cards.

| Slot | Current preview | Suggested recording preview |
|---|---|---|
| Snake / 贪吃蛇 | `assets/demos/snake-placeholder.svg` | `assets/demos/snake.gif` |
| Crafter / 生存建造 | `assets/demos/crafter-placeholder.svg` | `assets/demos/crafter.gif` |

## Replace the slots

1. Add a small GIF or another supported image preview under `assets/demos/`. Keep a full-resolution MP4 separately if the recording is large. Matching preview aspect ratios (for example, 16:9) keep the two columns aligned.
2. Replace the two image `src` values inside the `DEMO SLOTS` block in both [English README](../README.md) and [中文 README](../README.zh-CN.md). Optionally wrap each image in a link to the full recording. A GitHub-uploaded video can use the attachment link supplied by GitHub.
3. Replace the placeholder introduction with a brief description of the actual run: the checkpoint used, environment/settings, and playback speed. Any score or completion claim should come from that recording's run.

例如，真实录屏补充后可以把对应位置改为：

```html
<a href="YOUR_FULL_RECORDING_URL">
  <img src="assets/demos/snake.gif" alt="Qev在贪吃蛇中的实际运行录屏" width="100%">
</a>
```

这里预留的是媒体展示位置，相关运行信息与真实录屏一起补充；现有的模型评测见 [evaluation.md](evaluation.md)。
