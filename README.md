# ICIP LLM-Inference-Start

这个仓库整理了 `AI Inference Infra` 基础课程的公开材料，重点覆盖推理框架阅读、性能测试方法和平台侧优化作业。

## 仓库内容

- `docs/`
  课程讲义与阅读材料
- `delivery/student-package/`
  面向学生发放的作业包
- `nano-vllm/`
- `sglang/`
- `vllm/`
- `vllm-ascend/`
  以上 4 个目录以 `submodule` 方式引用上游源码，便于课程中进行代码阅读和版本对照

## 公开范围

这个公开仓库只保留学生可见内容：

- 课程讲义
- 学生作业包
- 上游源码引用

教师侧调参记录、私钥、私有实验结果和其他敏感文件不包含在公开仓库中。

## 作业概览

- `Q1`：共享输入场景下的推理吞吐优化
- `Q2`：固定 workload 下的单请求延迟优化
- `Q3`：Ascend 平台上的运行时与并行配置优化

## 获取源码子模块

首次拉取后执行：

```bash
git submodule update --init --recursive
```

## 说明

- 学生应从 `delivery/student-package/` 开始
- 课程讲义位于 `docs/`
