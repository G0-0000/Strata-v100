# Strata-v100

面向 **NVIDIA Tesla V100** 与 **双 CPU／NUMA 系统**的优化分支。本页只介绍本分支的修改；通用功能与使用说明请参阅[上游 Strata](https://github.com/Niko1221/Strata)。

> **优化代码与日常开发位于 [`b1-fp16`](https://github.com/G0-0000/Strata-v100/tree/b1-fp16)。使用、构建或查看实现时，请以该分支为准，而非默认的 `main` 分支。**

## V100 显卡相关修改

- **Volta FP16 WMMA 预填充路径**：针对 `sm_70` 的专家计算作适配与扩展。相关工作基于 rewin123 的开放 [PR #1401](https://github.com/Niko1221/Strata/pull/1401)，本分支的适配另见 [PR #1771](https://github.com/Niko1221/Strata/pull/1771)。
- **Q8_0 计算路径调整**：引入交错 MMVQ、V100 实测行表与 Q8_0 反量化合宽相关改动，针对 Volta 的执行特性作适配。

这些改动不代表所有负载都会更快。尤其 WMMA 路径的本机 A/B 结果并无稳定提速，不作通用性能承诺。

## 双 CPU／NUMA 相关修改

- **可指定热节点**：通过 `--numa-hot-node` 配置热内存的 NUMA 放置，适应 GPU 与内存分属不同 CPU 节点的系统。
- **会话缓存冷热放置**：区分活动会话与停靠缓存，修正检查点在不同路径下的 NUMA 绑定。
- **恢复时实际迁移页面**：会话从 RAM 缓存恢复后，将页面迁至热节点，而不只是修改绑定策略，避免页面继续留在原节点。
- **内存区域可观测性**：为专家、活动会话及会话缓存等自有 VMA 添加名称，便于通过 `/proc` 核对内存用途与 NUMA 分布。
- **驻留预算修复**：修正 arena 预占内存被重复计入预算的问题，避免本可容纳的专家补集被错误削减。
- **文件页缓存回收**：增加 `--drop-file-cache`，在满足全驻留条件时回收重复的专家文件页缓存。

## 验证范围与限制

主要验证环境为 **Tesla V100-SXM2-32GB、双路 CPU／NUMA、Linux**。效果与模型量化、专家驻留情况、上下文长度及硬件拓扑有关，不能直接外推到其他设备或负载。

- WMMA 路径仍有实验性质，数值不保证与原路径逐位一致。
- 会话恢复页迁移修复目前位于 CUDA 路径；SYCL 尚无对应实现，相关改动也未经 SYCL 编译验证。
- 本分支包含仍在上游审议的修改；列出 PR 不表示已被上游合并。

## 代码与资料

- [优化分支 `b1-fp16`](https://github.com/G0-0000/Strata-v100/tree/b1-fp16)
- [本分支开发说明](https://github.com/G0-0000/Strata-v100/blob/b1-fp16/CONTEXT.md)
- [上游项目与通用文档](https://github.com/Niko1221/Strata)

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was
compressed by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE). A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).
