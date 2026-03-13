# AI Inference Infra 基础课程讲义

这组文档是学生版讲义，不是授课提纲。

阅读目标不是“记住命令怎么跑”，而是逐步建立下面这套能力：

- 从系统视角理解 LLM inference
- 把系统问题映射到 vLLM / SGLang / vllm-ascend 的实际代码
- 用 benchmark 和实验去验证自己的理解

## 讲义结构

### [第 1 节：从 Hugging Face `generate()` 到现代 LLM 推理系统](D:/workspace/vllm-course/docs/lectures/lecture-01-inference-systems.md)

这一节解决两个问题：

- 为什么 LLM 推理会从“模型调用问题”演化成“系统设计问题”
- continuous batching、PagedAttention、prefix caching、speculative decoding、graph capture 等关键技术分别在解决什么问题

读完这一节后，你应该能把近几年的主流 inference optimization 放进一张统一地图中。

### [第 2 节：vLLM 与 SGLang 源码导读：从系统抽象到工程实现](D:/workspace/vllm-course/docs/lectures/lecture-02-vllm-sglang.md)

这一节的重点不是“会用命令”，而是学会：

- 怎么读一个 serving 框架
- 如何在仓库中定位入口、配置、scheduler、cache、worker、backend
- 如何比较 vLLM 与 SGLang 的设计取舍

### [第 3 节：Benchmark、调优与实验设计：把“会跑”变成“会研究”](D:/workspace/vllm-course/docs/lectures/lecture-03-benchmark-tuning.md)

这一节把前两节的理解落到实验上，重点包括：

- workload 设计
- 指标体系
- baseline 与优化版的构造
- NVIDIA / Ascend 平台调优思路
- 实验报告与证据链

## 建议阅读顺序

建议按 1 -> 2 -> 3 的顺序读，不建议跳着看。

原因很简单：

- 如果第 1 节没建立起性能模型，第 2 节源码会像目录浏览
- 如果第 2 节没建立起执行路径，第 3 节 benchmark 会像参数扫表

## 课前准备建议

在正式进入讲义前，建议先快速浏览这四个仓库的 README：

- `nano-vllm/README.md`
- `vllm/README.md`
- `sglang/README.md`
- `vllm-ascend/README.md`

目标不是理解所有细节，而是先知道：

- 这些框架各自强调什么
- 为什么业界会同时需要多个推理框架
- 为什么硬件平台差异会影响 inference optimization

## 阅读时的建议方法

每一节都建议你带着三个问题读：

1. 这一节主要想解释哪类系统问题？
2. 这些技术或代码结构在解决什么瓶颈？
3. 如果我要用 benchmark 验证它，我应该怎么设计 workload？

如果你始终带着这三个问题读，收获会比单纯记名词更大。
