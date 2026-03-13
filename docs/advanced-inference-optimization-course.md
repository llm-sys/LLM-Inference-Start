# Advanced 推理优化课程设计

## 文档目的

这份文档面向讲师备课，服务于三节关于 advanced inference optimization 的课程设计。默认受众为：

- 有较强 NLP / LLM 研究背景，论文阅读能力强
- 能读 PyTorch，能做一定程度的工程修改
- 对 MLSys、Serving、硬件性能分析、benchmark 方法论不够熟悉

因此，这门课不应该讲成“推理框架功能导览”，而应该讲成“如何建立推理系统的性能模型，并把这个模型映射到 vLLM / SGLang / Ascend plugin 的具体实现上”。

## 课程定位

### 课程主线

把学生从“模型研究者视角”切换到“推理系统研究者 / 性能工程师视角”。

学生在课程结束后，应该能够做到：

1. 解释一次 LLM inference request 从请求进入、调度、prefill、decode 到返回结果的完整路径。
2. 理解吞吐、时延、显存占用、KV cache、batching、并行策略之间的基本约束关系。
3. 读懂 vLLM / SGLang 中与调度、KV cache、执行器相关的核心代码，而不是只会调用 API。
4. 设计一个可信的 benchmark，知道如何控制变量、如何解释结果、如何避免“看起来更快但结论不可信”的测试。
5. 在 NVIDIA 与 Ascend 两类硬件上，对同一模型进行基础调优，并形成结构化实验记录。

### 非目标

这三节课不以“从零实现完整推理框架”为目标，也不以“系统性教授 CUDA / Triton kernel 开发”为目标。

kernel 级优化可以讲思想和接口位置，但不要让整门课陷入过多底层细节；对这批学生来说，更重要的是建立性能模型、知道瓶颈在哪里、知道源码里去哪里看。

## 受众画像与教学策略

### 受众特点

这批学生的典型特点是：

- 对 Transformer、attention、sampling、quantization 等算法概念熟悉
- 对 PyTorch module、tensor shape、forward path 不陌生
- 容易把推理问题想成“模型 forward 的问题”
- 容易低估 request scheduling、memory layout、KV cache 管理、host-device 协同对整体性能的影响
- 容易在 benchmark 中只盯总 tokens/s，而忽略 TTFT、TPOT、P50/P95 latency、输入输出长度分布等关键变量

### 教学原则

1. 先建性能模型，再讲源码细节。
2. 先讲“为什么这是瓶颈”，再讲“这个仓库里是怎么实现的”。
3. 先用 nano-vllm 建立最小心智模型，再切到 vLLM / SGLang 的工业实现。
4. 尽量把所有优化都归因到几个核心矛盾：
   - compute vs memory bandwidth
   - prefill vs decode
   - latency vs throughput
   - static graph vs dynamic workload
   - 通用性 vs 针对硬件的特化
5. 对名词保持克制。学生不缺名词，缺的是“这个名词和系统瓶颈之间到底是什么关系”。

### 建议的讲法

建议少讲“功能列表”，多讲“性能故事线”：

- 为什么 naive HF generate 慢？
- 为什么 decode 阶段看起来 FLOPs 不大，却常常卡在 memory / KV cache / 调度？
- 为什么 batch size 不是越大越好？
- 为什么同一套参数在 NVIDIA 上有效，在 Ascend 上可能失效？
- 为什么 benchmark 结果不写 workload 分布就几乎不可复现？

## 课程结构总览

建议按每节 2.5 到 3 小时设计；如果你的实际课时更短，可以把每节中的代码阅读和实验演示缩成 20 到 30 分钟。

### 第 1 节：从模型推理到推理系统

#### 本节目标

- 建立 LLM inference 的系统视角
- 让学生理解 prefill / decode、KV cache、continuous batching、paged/radix cache 的必要性
- 用 nano-vllm 建立“一个推理引擎最小需要哪些组件”的心智模型

#### 讲授内容

1. 从研究者熟悉的 generate 流程出发
   - tokenizer -> prompt -> prefill -> decode loop -> sampling -> stop condition
   - 单请求 forward 和多请求 serving 的根本差异
2. 推理系统核心指标
   - TTFT: time to first token
   - TPOT / ITL: time per output token / inter-token latency
   - throughput: requests/s, tokens/s
   - P50 / P95 / P99 latency
   - 显存占用与 KV cache footprint
3. 推理系统核心对象
   - request / sequence
   - scheduler
   - block / page / cache entry
   - model runner / worker
4. 常见优化背后的动机
   - continuous batching
   - prefix caching
   - paged attention / radix cache
   - chunked prefill
   - cuda graph / compiled graph
   - speculative decoding

#### 代码阅读路径

先读 `nano-vllm`，只抓主链路，不陷入实现细节：

- `nano-vllm/nanovllm/llm.py`
- `nano-vllm/nanovllm/engine/llm_engine.py`
- `nano-vllm/nanovllm/engine/scheduler.py`
- `nano-vllm/nanovllm/engine/block_manager.py`
- `nano-vllm/nanovllm/engine/model_runner.py`
- `nano-vllm/nanovllm/layers/attention.py`
- `nano-vllm/nanovllm/models/qwen3.py`
- `nano-vllm/bench.py`

#### 这一节的关键产出

学生应该能用一句话解释：

- “为什么 inference infra 不是把 model.forward 放到 server 里就结束了”
- “为什么 KV cache 管理是第一等公民”
- “为什么 scheduler 会直接影响性能和 tail latency”

#### 课后小作业

让学生画出一张最小推理引擎数据流图，至少包含：

- 请求进入
- tokenizer / input length
- prefill
- KV cache 分配
- decode loop
- sampling
- 结果返回

### 第 2 节：vLLM 与 SGLang 的执行路径和优化设计

#### 本节目标

- 让学生从“会用框架”进入到“能读关键代码、能比较设计取舍”
- 理解 vLLM 与 SGLang 的共同点和差异
- 学会把框架中的模块映射回上一节的性能模型

#### 讲授内容

1. vLLM 的核心设计线索
   - 面向通用 serving 的高吞吐架构
   - continuous batching
   - KV cache manager
   - worker / model runner
   - benchmark CLI 与配置系统
2. SGLang 的核心设计线索
   - 高性能 runtime
   - radix/prefix cache 体系
   - scheduler 与 manager 拆分
   - structured outputs / speculative / disaggregation 等扩展能力
3. 两者比较时的建议维度
   - API 体验不是重点，执行模型才是重点
   - prefix caching 粒度
   - 调度器与 worker 的耦合方式
   - benchmark 工具是否方便控制 workload
   - 针对特定模型 / 特定硬件的特化程度

#### vLLM 推荐阅读路径

优先读“用户请求如何进入系统并最终落到 worker”这条链：

- `vllm/vllm/entrypoints/llm.py`
- `vllm/vllm/engine/arg_utils.py`
- `vllm/vllm/engine/llm_engine.py`
- `vllm/vllm/config/cache.py`
- `vllm/vllm/config/scheduler.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/core/kv_cache_manager.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/gpu/model_runner.py`
- `vllm/vllm/entrypoints/openai/api_server.py`
- `vllm/vllm/entrypoints/cli/benchmark/throughput.py`
- `vllm/vllm/entrypoints/cli/benchmark/latency.py`
- `vllm/vllm/entrypoints/cli/benchmark/serve.py`

#### SGLang 推荐阅读路径

优先读“服务入口 -> scheduler -> worker -> cache”的路径：

- `sglang/python/sglang/launch_server.py`
- `sglang/python/sglang/srt/server_args.py`
- `sglang/python/sglang/srt/entrypoints/http_server.py`
- `sglang/python/sglang/srt/entrypoints/http_server_engine.py`
- `sglang/python/sglang/srt/managers/scheduler.py`
- `sglang/python/sglang/srt/managers/tp_worker.py`
- `sglang/python/sglang/srt/managers/cache_controller.py`
- `sglang/python/sglang/srt/mem_cache/radix_cache.py`
- `sglang/python/sglang/srt/mem_cache/hiradix_cache.py`
- `sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `sglang/python/sglang/bench_serving.py`
- `sglang/python/sglang/bench_offline_throughput.py`

#### 本节建议的课堂活动

做一次“对照式代码阅读”：

1. 用同一个 prompt 请求分别走一遍 vLLM 和 SGLang 的入口文件。
2. 让学生回答：
   - 请求对象在哪里被封装？
   - 调度决策在哪里发生？
   - KV cache 的元数据由谁维护？
   - 哪一层开始和具体硬件强相关？

#### 本节结束时学生应具备的能力

- 能在陌生 serving 框架中快速定位入口、调度、cache、worker、benchmark 脚本
- 能说清楚一个优化是在解决调度问题、内存问题，还是 kernel 问题
- 不会再把“框架快”笼统地理解成“某个 kernel 更快”

### 第 3 节：Benchmark、参数调优与 NVIDIA / Ascend 对比实验

#### 本节目标

- 建立可信 benchmark 的基本方法
- 学会从 workload、参数、硬件三个层次解释性能差异
- 让学生能完成 NVIDIA 与 Ascend 上的同模型实验记录

#### 讲授内容

1. Benchmark 的最小方法论
   - 明确任务类型：offline throughput / online serving / latency-sensitive
   - 固定 workload 分布：输入长度、输出长度、并发数、到达模式
   - 固定模型、精度、batching 策略、并行策略
   - 分开报告 warmup 与 steady-state
2. 必须汇报的指标
   - TTFT
   - TPOT / ITL
   - requests/s
   - output tokens/s
   - P50 / P95 latency
   - 显存或 NPU memory 占用
   - 失败率 / OOM / 超时情况
3. 参数调优要从“性能模型”出发
   - batch size / max_num_seqs
   - max_num_batched_tokens
   - tensor_parallel_size
   - context length
   - precision: bf16 / fp16 / fp8 / int4 等
   - prefix cache 相关开关
   - graph compile 相关开关
4. 常见误区
   - 只看平均吞吐，不看 tail latency
   - 没有固定输入输出长度分布
   - 把不同精度或不同 prompt distribution 的结果放在一起比较
   - 没有区分 prefill-bound 和 decode-bound workload
   - 在一个硬件平台调出来的参数直接照搬到另一个平台

#### NVIDIA 平台建议实验

建议在 vLLM 与 SGLang 上各跑至少两类 workload：

1. 短输入短输出
   - 更容易观察 scheduling / launch overhead / TTFT
2. 长输入长输出
   - 更容易观察 KV cache、memory pressure、prefill 与 decode 的占比变化

建议对以下参数做 sweep：

- `max_num_seqs`
- `max_num_batched_tokens`
- `tensor_parallel_size`
- precision
- prefix cache 开关

#### Ascend 平台建议实验

Ascend 部分建议以 `vllm-ascend` 为主，不把重点放在“如何安装环境”，而是放在“同样的系统问题在 Ascend 上如何体现”为主。

建议阅读：

- `vllm-ascend/vllm_ascend/platform.py`
- `vllm-ascend/vllm_ascend/ascend_config.py`
- `vllm-ascend/vllm_ascend/attention/attention_v1.py`
- `vllm-ascend/vllm_ascend/compilation/acl_graph.py`
- `vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py`
- `vllm-ascend/vllm_ascend/distributed/device_communicators/pyhccl.py`
- `vllm-ascend/vllm_ascend/ops/register_custom_ops.py`
- `vllm-ascend/benchmarks/scripts/run-performance-benchmarks.sh`

引导学生关注：

- 哪些模块是复用 vLLM 主干的，哪些是 Ascend plugin 自己接管的
- graph compile、custom ops、通信栈会怎样影响调优手法
- 为什么硬件 plugin 的优化往往不是“复制一份 CUDA 思路”这么简单

#### 模型选择建议

建议课程使用“同一模型、两套平台”的实验设计。模型名以实验环境实际可用版本为准。

如果实验机已经准备好 Qwen3.5，就直接沿用；如果没有，退到同量级 Qwen 系列模型即可。关键不是具体型号，而是：

- 模型规模要足够让 cache / batching 差异显现
- 同一个模型要能在两类硬件环境中都稳定运行
- 精度设置和上下文长度要可对齐

#### 本节结束时学生应具备的能力

- 知道 benchmark 报告里哪些信息必须写
- 知道调参时应该优先改哪几类参数
- 能解释“NVIDIA 更快”或“Ascend 更稳”背后可能是哪类系统因素

## 建议的代码阅读节奏

不要要求学生课前读完整仓库。建议按下面的节奏推进：

### 课前

- 只读 `nano-vllm/README.md`
- 只扫一眼 `vllm/README.md`、`sglang/README.md`、`vllm-ascend/README.md`

### 第 1 节后

- 精读 `nano-vllm` 的 engine 和 attention
- 能手绘出最小执行路径

### 第 2 节后

- 选读 vLLM 或 SGLang 其中一条主链
- 小组内互相解释 scheduler / cache / worker 的对应关系

### 第 3 节前

- 看 benchmark 脚本
- 预先写好实验记录模板

## 建议给学生补的 MLSys 最小背景

如果学生的 MLSys 基础明显偏弱，可以在第一节课前 20 到 30 分钟补一个最小背景包，只讲和推理直接相关的内容：

1. GPU / NPU 的计算与带宽不是同一个瓶颈
2. latency-sensitive workload 和 throughput-oriented workload 的目标不一样
3. host 调度、device 执行、memory movement 三者必须一起看
4. 一个系统变快，可能不是因为单次算得更快，而是因为空转更少、等待更少、cache 命中更多

## 考核设计建议

### 1. 飞书问卷

题目不要过度考名词定义，应该考“能否用系统视角解释现象”。

适合出的题型：

- 选择题：给一个 benchmark 结果，问最可能的瓶颈来源
- 判断题：判断某个实验设计是否公平
- 简答题：解释 prefill-bound 与 decode-bound 的差别

### 2. 上手实验记录

实验报告建议强制包含以下字段：

- 硬件环境
- 软件环境
- 模型与精度
- workload 定义
- 核心参数
- benchmark 脚本或命令
- 原始结果
- 调优过程
- 结论与反思

重点不应该是“谁跑得更快”，而应该是“是否形成了可解释、可复现、可对比的实验过程”。

## 讲师备注

这门课面对的是很强的研究者，所以不要低估他们理解抽象概念的能力；但也不要高估他们对工程约束的直觉。

最容易出问题的地方不是学生看不懂代码，而是：

- 把系统设计问题重新翻译成算法问题
- 把 benchmark 当成跑脚本，而不是做实验
- 把硬件差异理解成“厂商不同，所以结果不同”

如果你能在三节课里让他们真正建立下面这个意识，这门课就已经成功了：

> 推理优化不是若干 isolated tricks 的堆砌，而是 workload、scheduler、KV cache、执行器、kernel 与硬件共同决定的系统问题。

## 下一步可继续扩写的方向

这份文档后续可以继续拆成四份材料：

1. 讲师版 slides 大纲
2. 学生版阅读手册
3. Benchmark 实验说明书
4. 飞书问卷题库
