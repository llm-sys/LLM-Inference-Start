# 题目 2：延迟优化

## 目标

在固定 benchmark 脚本和固定数据集的前提下，尽可能降低端到端延迟。

## 评分指标

官方指标为 `mean_latency_sec`。

数值越低越好。

## 满分线

- 当 `mean_latency_sec <= 0.88` 时，视为达到满分线。

## 固定条件

- 模型路径固定为 `/ceph/arknet/hf_models/Qwen/Qwen3-14B`
- 数据集文件固定
- prompt 顺序固定，默认只对前 `8` 条 prompt 计分
- sampling 参数固定
- 必须使用提供的 Python API 工作流，也就是基于 `LLM(...)`

## 限制条件

- 不允许下载任何模型、tokenizer、checkpoint、adapter 或额外权重
- 不允许使用任何额外的辅助模型
- 不允许修改数据集文件
- 不允许修改指标计算逻辑
- 不允许改变请求数量
- 不允许使用外部推理服务
- 不允许切换为 HTTP server benchmark

## 提交内容

请提交：

- 你修改后的脚本
- 你实际运行的完整命令
- 输出得到的 JSON 结果文件
- 一段简短说明，描述你做了哪些改动
