# Strata V100 项目参考文档

本文件为直忆（strata-ask）存盘语料之固定参考文档，置于项目 git 根；档语料【项目参考文档】节整节读取本文件。内容唯稳定概念、名词口径与路径，版本号、性能数字、进程状态等易漂移者悉归记忆或运维记录，不入此档。

## 项目介绍

Strata 是本地运行大模型的开源项目：桌面应用、Terminal、Python 服务与 C++/GPU 推理引擎；部分采用 llama.cpp/ggml 代码，但不是 llama.cpp 兼容实现，不可混言。本仓 strata-v100 是面向 Tesla V100/Volta（sm_70、CUDA 12、部分路径 FP16 张量核）的个人 fork 工作仓；生产运行另在 `/home/g0/strata`（systemd 用户单元 strata.service，端口 8080）。**改工作仓不等于已部署**，二者代码、配置、构建不必一致。

## 本体

- **服务与引擎**：`serve/server.py` 的 `Service` 管 HTTP、排队与会话动作；引擎子进程加载模型、持有会话真实状态（token IDs、运行态、KV、检查点），服务经 SAVE/RESTORE 消息与之交互。
- **前缀匹配**：请求与已有 token IDs 从头逐位比较；存在旧会话状态不保证新请求可复用。
- **档（会话文件缓存）**：显式动作——`/slots/0?action=save|restore`，或 chat 请求的 `strata_session` 对象一次 POST 完成读、生成、写（同队列轮次）。文件承载引擎状态，非完整聊天导出；普通请求不会自动读档。
- **旁档**：`<档名>.tokens.json`，存 token IDs、引擎报告 token 数、分词器/模板哈希与版本；是缝合之据，非 KV 本体，亦非磁盘缓存的 `.meta`。
- **磁盘缓存（自动）**：引擎自动换会话时的 ConversationSpillCache（`.sess`/`.meta`/`.dlog`/`.dchain` 增量链），与显式指定的档是两物；RAM 中已停放会话副本、操作系统页缓存又各是一物，四者不可混。
- **槽位**：`/slots/0` 之 0 是保存/恢复接口唯一槽号；`--batch N` 的并行批处理槽是另一概念——`/slots` 与 `strata_session` 均不支持并行批处理。

## 名词规范

- **档**：会话文件（save file）；`.bin`、`.sess` 后缀皆可，后缀非格式判据。
- **缝合（splice）**：增量写之拼接；必须以 restore 先行、首条新消息为 user、不带新 system 消息或图像。
- **三动作约定**（用户钦定）：**读** = restore；**增量写** = restore + splice + save（只发新轮次，但引擎仍整体保存，勿误解为只向文件追加字节）；**全部重置** = 仅带 save，不读旧档、生成后覆盖。
- **缓存复用规则/账本**：`docs/CACHE_REUSE_RULES.md` 与 `.strata-cache-gate`；规则文本、现行门控代码与实际配置三者须分别核验，不可单执一端言现状。
- **缓存词汇表**：`personal/docs/GLOSSARY.md` 为准；「当前会话内存缓存」已有定名，勿自造第三个名称。
- **slot / session / cache** 在不同子系统各有所指；答问先声明语境（`/slots` 档、批处理并发槽、自动磁盘缓存、RAM 监控分类）。

## 俗称对照

| 俗称 | 档/码用词 |
|---|---|
| 档 | save file / 会话文件 |
| 旁档 | sidecar / `*.tokens.json` |
| 缝合 | splice |
| 账本、台账 | cache gate 账本 / `.strata-cache-gate` |
| 层切分 | `layer_split` / `--layer-split` |
| 批处理槽位 | batch slots |
| 工作仓 | `/home/g0/strata-v100` |
| 生产仓 | `/home/g0/strata` |

## 关键位置

- 工作仓 `/home/g0/strata-v100`；生产目录 `/home/g0/strata`（单元 `~/.config/systemd/user/strata.service`）
- 服务入口 `serve/server.py`；引擎源码 `src/`、`include/`；针对性测试 `serve/test_session_composite.py`、`serve/test_slots.py`
- 文档：`docs/DETAILS.md`（引擎/API/会话/缓存）、`docs/BATCHING.md`、`docs/NVIDIA_V100.md`、`docs/CACHE_REUSE_RULES.md`、`personal/docs/GLOSSARY.md`（personal/ 不入 git，部分文档系快照，须核验现行代码）
- 生产配置 `/home/g0/strata/strata-nvfp4-flashnext.json`；档目录 `/models/strata-slots/nvfp4`；磁盘缓存 `/models/strata-conv-cache/nvfp4`

## 答问戒条

- 静态证据（代码路径、配置文本）不等于已部署、已启用、已实测；性能与命中须实测方可言。
- 生成完成后的保存失败仍可返回 200，须查 `session.save_error`；恢复失败则不生成。
- 不可声称每份旧缓存均已经 KV 实测合格——存在 `--conversation-cache-assume-pass` 直接放行同模型条目之开关。
