# 题目 3：Ascend 运行时优化

## 目标

在给定的 Ascend 环境上优化固定 workload 的 decode 吞吐，评分指标为 `output_tokens_per_sec`。

## 固定条件

- 模型路径：`/cache/hf_model/Qwen3.5-35B-A3B`
- 硬件：`2 x 910B3`
- 数据集：`student-package/data/q3_ascend_decode_prompts.txt`
- 评分样本：默认只取前 `8` 条 prompt，顺序固定
- 生成长度：`max_tokens = 128`
- 脚本里内置了 3 组布局预设：`layout_a / layout_b / layout_c`
- 脚本里内置了 2 组运行模式：`mode_a / mode_b`

## 限制条件

- 不允许下载任何模型、数据集或额外产物
- 不允许修改数据集内容、顺序或样本数量
- 必须使用提供的 Python API 工作流，不能改成 `serve` 接口
- 最终提交必须写清楚你选择的布局预设和运行模式
- 如果某种配置无法稳定运行，不应作为最终提交

## 默认 baseline

- 脚本：`student-package/scripts/q3_ascend_baseline.py`
- 默认配置：
  - `layout = layout_b`
  - `engine_mode = mode_a`
- baseline 参考值：
  - `output_tokens_per_sec ≈ 141.66`

## 计分方式

- 核心指标：`output_tokens_per_sec`
- 满分线：`>= 338`

## 提示

- 这道题考察的是平台优化和并行布局选择，不是模型效果
- 请优先保证配置可复现、可解释、可稳定运行
