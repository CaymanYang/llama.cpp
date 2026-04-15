# llama.cpp 项目目录地图

> 仓库路径：`llama.cpp/`（本笔记针对 `/home/bookery/CPU_inference/llama.cpp`）。**不含** `build/` 等生成目录。

---

## 顶层一览

| 目录 / 文件 | 职责 |
|-------------|------|
| **`include/`** | 对外稳定 C API 头文件，主要是 **`llama.h`**（`libllama` 入口）。 |
| **`src/`** | **llama 核心实现**：模型加载、构图、`llama_context`、KV、采样、各架构 `llm_build_*` 等（`.cpp` 主体）。 |
| **`ggml/`** | **GGML 子项目**：张量、计算图、量化、**GGUF**、**backend**（CPU/CUDA/Vulkan/…）、调度与内存分配。 |
| **`common/`** | **多工具共享**逻辑：命令行解析、日志、下载、采样封装、`nlohmann/json` 等 CLI/server 共用代码。 |
| **`tools/`** | **正式命令行工具**：`llama-cli`、`llama-server`、`quantize`、`llama-bench` 等，每子目录一目标。 |
| **`examples/`** | **示例程序**：最小 C API（`simple`）、聊天、embedding、投机解码、Android/SwiftUI 等演示。 |
| **`tests/`** | **单元 / 回归测试**：tokenizer、量化、alloc、rope、grammar 等。 |
| **`docs/`** | **文档**：构建、Docker、算子说明 `ops.md`、多模态、后端说明、`development/` 开发指南。 |
| **`cmake/`** | **CMake 模块**与辅助脚本（依赖探测、安装规则等）。 |
| **`ci/`** | **CI 脚本**（如 `run.sh`）：自动化构建与测试流程。 |
| **`scripts/`** | **杂项脚本**：Apple、Jinja、Snapdragon 等平台或生成任务。 |
| **`grammars/`** | **GBNF 语法文件**等资源，供约束解码 / grammar sampling 使用。 |
| **`models/`** | **聊天模板等元数据**（如 `templates/`），非模型权重本身。 |
| **`licenses/`** | **第三方许可证**文本汇总。 |
| **`media/`** | **图片等静态资源**（README、文档引用）。 |
| **`benches/`** | **基准测试相关**配置或场景（如特定硬件目录）。 |
| **`pocs/`** | **概念验证 / 小实验**（如 `vdot` 向量点乘 POC）。 |
| **`gguf-py/`** | **Python 侧 GGUF** 读写库与示例，便于与 Hugging Face 工作流衔接。 |
| **`requirements/`** | **Python 依赖列表**（部分脚本 / 转换工具用）。 |
| **`vendor/`** | **随仓库分发的第三方源码**（httplib、miniaudio、minja、nlohmann、stb 等），减少外部依赖。 |
| **`.devops/`** | **Nix 等 DevOps** 配置。 |
| **`CMakeLists.txt`** | 根构建入口：聚合 `ggml`、`src`、`common`、`tools`、`examples`、`tests`。 |

---

## `src/`（llama 核心）

| 子路径 / 模式 | 职责 |
|---------------|------|
| **`llama.cpp`** | C API 导出函数实现（`llama_model_*`、`llama_decode`、包装层）。 |
| **`llama-model*.cpp`** | 模型加载、张量布局、`build_graph` 分发到各架构。 |
| **`llama-context*.cpp`** | 上下文、decode、调度器、图 reserve/compute。 |
| **`llama-memory*.cpp` / `llama-kv-cache*.cpp`** | KV / 记忆抽象与缓存实现。 |
| **`llama-graph*.cpp` / `llama-graph.h`** | 计算图构建参数与辅助。 |
| **`llama-sampling*.cpp`** | 采样链、penalties 等。 |
| **`llama-arch.h` / `llama-hparams*`** | 架构枚举、超参。 |
| **`models/`** | **各模型架构前向实现**（如 `llama.cpp`、`qwen2.cpp`、`bert.cpp`），由总控 `build_graph` 按 `LLM_ARCH_*` 分派。 |

---

## `ggml/`（张量库 + 后端）

| 路径 | 职责 |
|------|------|
| **`ggml/include/`** | 公共头：`ggml.h`、`ggml-backend.h`、`ggml-alloc.h`、`gguf.h` 等。 |
| **`ggml/src/`** | **核心**：`ggml.c`、`ggml.cpp`、`ggml-alloc`、`ggml-backend*.cpp`、`gguf.cpp`、量化 `ggml-quants`、线程 `ggml-threading`。 |
| **`ggml/src/ggml-cpu/`** | **CPU 后端**（x86/ARM/RISC-V 等子目录、vec/ops）。 |
| **`ggml/src/ggml-cuda/`** | **NVIDIA CUDA** 内核与接口。 |
| **`ggml/src/ggml-vulkan/`** | **Vulkan** 后端。 |
| **`ggml/src/ggml-metal/`** | **Apple Metal** 后端。 |
| **`ggml/src/ggml-opencl/`** | **OpenCL** 后端。 |
| **`ggml/src/ggml-sycl/`** | **oneAPI SYCL**（Intel GPU 等）。 |
| **`ggml/src/ggml-hip/`** | **AMD ROCm HIP**（CUDA 同源思路）。 |
| **`ggml/src/ggml-musa/`** | **摩尔线程 MUSA** 等。 |
| **`ggml/src/ggml-rpc/`** | **RPC**：远程设备上跑算子。 |
| **`ggml/src/ggml-blas/`** | **BLAS** 加速路径。 |
| **`ggml/src/ggml-cann/`、`ggml-zendnn/`、`ggml-zdnn/`、`ggml-hexagon/`** | 厂商/加速器专用后端（华为、IBM、高通 Hexagon 等）。 |
| **`ggml/src/ggml-webgpu/`** | **WebGPU** 实验性后端。 |
| **`ggml/cmake/`** | GGML 侧 CMake 辅助。 |

---

## `common/`

CLI、`llama-server`、bench 等共用的 **参数解析、日志、模型下载、JSON、线程池封装**，避免每个 `tools/*` 重复实现。

---

## `tools/`（按子目录）

| 子目录 | 典型产物 / 职责 |
|--------|------------------|
| **`cli/`** | **`llama-cli`**：交互/批处理推理主程序。 |
| **`server/`** | **`llama-server`**：HTTP/OpenAI 兼容 API、WebUI 静态资源生成。 |
| **`quantize/`** | **`llama-quantize`**：GGUF 量化。 |
| **`llama-bench/`** | 性能基准测试。 |
| **`tokenize/`** | 仅分词测试/工具。 |
| **`perplexity/`** | 困惑度评估。 |
| **`mtmd/`** | **多模态**（如 CLIP 投影）相关工具与库。 |
| **`run/`** | 终端 **REPL 风格**运行器。 |
| **`batched-bench/`** | 批处理性能。 |
| **`completion/`** | completion 专用工具。 |
| **`imatrix/`** | 重要性矩阵（量化用）。 |
| **`gguf-split/`** | GGUF 分片合并/拆分。 |
| **`export-lora/`、`cvector-generator/`、`fit-params/`、`tts/`、`rpc/`** | LoRA 导出、向量、参数拟合、TTS、RPC 相关工具。 |

---

## `examples/`（按类型）

| 类型 | 代表目录 |
|------|----------|
| **入门** | `simple/`、`simple-chat/`、`simple-cmake-pkg/` |
| **批处理 / 并行** | `batched/`、`parallel/` |
| **投机解码** | `speculative/`、`speculative-simple/`、`lookahead/` |
| **Embedding / 检索** | `embedding/`、`retrieval/` |
| **状态持久化** | `save-load-state/` |
| **GGUF** | `gguf/`、`gguf-hash/` |
| **微调** | `training/` |
| **扩散等** | `diffusion/` |
| **移动端** | `llama.android/`、`llama.swiftui/`、`batched.swift/` |
| **其它** | `passkey/`、`idle/`、`eval-callback/`、`model-conversion/`、`sycl/`、`convert-llama2c-to-ggml/`、`deprecation-warning/`、`gen-docs/` |

---

## `tests/`

针对 **tokenizer、grammar、量化、alloc、backend、thread-safety** 等的可执行测试；部分依赖 `get-model.cpp` 与外部 GGUF。

---

## `docs/` 子目录提示

| 路径 | 内容倾向 |
|------|----------|
| **`docs/backend/`** | 各计算后端说明。 |
| **`docs/development/`** | 加模型、构建细节等开发者文档。 |
| **`docs/multimodal/`** | 多模态管线。 |
| **`docs/ops.md`**（根文档引用） | 算子与后端能力对照。 |

---

## 阅读顺序建议（结合本地图）

1. **`include/llama.h`** + **`examples/simple/`**  
2. **`src/llama.cpp`**、**`src/llama-context.cpp`**、**`src/llama-model.cpp`**（可择章）  
3. **`ggml/include/ggml-backend.h`** + **`ggml/src/ggml-backend*.cpp`**  
4. **`tools/cli/`** 或 **`tools/server/`** 看完整产品化路径  

---

*生成说明：根据仓库目录结构与 CMake 组织归纳；若上游重命名目录，以官方 `README.md` / `docs/build.md` 为准。*
