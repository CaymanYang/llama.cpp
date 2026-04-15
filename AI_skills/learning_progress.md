# llama.cpp 学习进度

> 与 Cursor 对话中整理的笔记：问题 + 结论摘要。代码路径以本机仓库 `/home/bookery/CPU_inference/llama.cpp` 为准。

---

## 当前进度概览

| 主题 | 状态 |
|------|------|
| 项目分层（API / model / ggml / tools） | 已建立整体地图 |
| C API 最小推理链（`simple.cpp`） | 已跟读：`load` → tokenize → `llama_init_from_model` → batch → `llama_decode` → sampler |
| `n_gpu_layers` 与 CPU/GPU 放置 | 已理解：`get_layer_buft_list`、KV 与 `dev_layer` 一致 |
| 权重 buffer 与 `supports_op` 探针 | 已理解：`select_weight_buft` / `weight_buft_supported` |
| 图推理 compute buffer | 已理解：`ggml_backend_sched` + `ggml_gallocr` |
| GGML 后端加载与注册 | 已扫：`ggml_backend_load_all`、`ggml_backend_score`、动态库命名 |
| 调试方式 | 已记录：Debug 构建、`gdb` 需先 `run`/`start` |

**待深入（可自行标进度）**：单算子 CUDA 内核、`build_graph` 某一架构、`tests/` 跑通、自定义后端原型。

---

## 1. 从零学什么（首回合）

- **llama.cpp**：C/C++ LLM 推理；权重多为 **GGUF**；张量栈在 **GGML**。
- **建议顺序**：先会跑 `llama-cli` / `llama-server` → 读 `include/llama.h` → 跟 `examples/simple/simple.cpp` → 再分模块下钻 `src/llama-model.cpp`、`src/llama-context.cpp`、`ggml/`。
- **文档入口**：`docs/build.md`、`docs/install.md`、`docs/development/HOWTO-add-model.md`。

---

## 2. C API 推理链（`examples/simple/simple.cpp`）

**核心对象**：`llama_model`（静态权重）、`llama_vocab`、`llama_context`（KV、调度、运行参数）、`llama_sampler`；喂给计算的是 **`llama_batch`**（见 `include/llama.h` 中 `llama_batch` 注释）。

**典型顺序**：

1. `ggml_backend_load_all()`（可选，注册动态后端）
2. `llama_model_load_from_file` + `llama_model_default_params()`（如 `n_gpu_layers`）
3. `llama_model_get_vocab` + `llama_tokenize`
4. `llama_context_default_params()`：设置 `n_ctx`、`n_batch` 等 → **`llama_init_from_model`**
5. 构造 `llama_sampler_chain` + greedy/top-p 等
6. `llama_batch_get_one` → 若有 encoder 则 `llama_encode` 再换 batch
7. 循环：`llama_decode` → `llama_sampler_sample` → 打印 → 下一 token 再组 batch
8. `llama_sampler_free` / `llama_free` / `llama_model_free`

---

## 3. 加载模型后的「优化」指什么

- **不是**训练意义上的优化；主要是：**权重放哪（CPU/GPU buffer）**、**计算图怎么切分**、**部分后端上的 graph optimize**、**图复用**。
- **加载**：`load_tensors` 里按层分配 `dev_layer` / `buft_list`；量化已在 GGUF 里。
- **推理**：`build_graph` → `ggml_backend_sched_alloc_graph` / `reserve`；部分后端在 split 上调用 `graph_optimize`（CPU 常为 NULL）。

---

## 4. 如何决定 CPU 还是 GPU

- **`model->devices`**：显式 `params.devices`，或默认扫描 `ggml_backend_dev_t`（GPU、RPC 等）；`split_mode == NONE` 且 **`main_gpu < 0`** 会清空 `devices` → 全 CPU。
- **`n_gpu_layers`**（配合 `devices` 非空）：`i_gpu_start = max(n_layer - n_gpu_layers, 0)`，从该层起往后放到 GPU（多卡时用 `tensor_split` / 显存比例切到不同 GPU）；**输入层（embedding）固定 CPU**。
- **纯 CPU**：`n_gpu_layers = 0` 或没有可用 GPU 设备。

---

## 5. 能否「CPU → GPU → CPU → GPU」交错层

- **`n_gpu_layers` 只能**「前 CPU、后连续一段 GPU」，**不能**按层奇偶交错。
- **`tensor_buft_overrides`** 主要改权重 buffer，**KV 仍跟 `model.dev_layer(il)`**，乱 override 会不一致。
- **若要交错**：需改 **`get_layer_buft_list`**（或等价逻辑），使 **`dev_layer[il]`** 与权重、KV 一致；代价是跨设备传输多，一般更慢。

---

## 6. GPU 不支持某算子时

- **加载权重**：`buft_list` 里 GPU 在前、**CPU 等 fallback 在后**；`select_weight_buft` 用 **`ggml_backend_dev_supports_op`** 探针（`weight_buft_supported`）逐个试。
- **跑图**：调度器给节点选支持该 op 的后端；必要时多段 split + **tensor copy**；若没有任何后端支持则可能失败（通常 CPU 覆盖最全）。

---

## 7. 仅 RISC-V + GPU 的推理卡

- **可以**回退到 RISC-V 上的 **CPU GGML 后端**（与 x86/ARM 一样，是「CPU backend」而非「某架构」）；前提是 **链接了 CPU 后端**（正常构建都有）。

---

## 8. 加载时如何知道「GPU 不支持」

- 对候选 `(dev, buft)` 构造**与真实用法一致**的微型 op 图，给权重挂 dummy buffer，调 **`ggml_backend_dev_supports_op(dev, op_tensor)`**。

---

## 9. 图推理的 buffer 怎么分配

- **三类**：model 权重、context（如 KV）、**compute**（图中间量）。
- **compute**：`ggml_backend_sched` 里 **`ggml_gallocr_new_n(bufts, n_backends)`**，按节点 backend 分配；`graph_reserve` / `alloc_graph` 走 reserve + alloc。

---

## 10. 调试建议

- **Debug 构建**：`cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug`，再编 `llama-simple`。
- **GDB**：先 **`run`** 或 **`start`**，再 **`n` / `s`**；否则会提示 *The program is not being run.*。
- **分模块断点**：`llama_model_load_from_file` → `load_tensors` → `llama_init_from_model` → `llama_decode` → `process_ubatch` → `build_graph` → `ggml_backend_sched_alloc_graph`。

---

## 11. GGML / 后端杂项

| 话题 | 结论 |
|------|------|
| **`GGML_API`** | 在 `ggml/include/ggml.h`：`GGML_SHARED` 时 Windows 用 `dllexport/dllimport`，否则 GCC `visibility` 或仅 `extern`。 |
| **`extern`** | 声明「定义在别处」；函数声明常可省略 `extern`。 |
| **`ggml_backend_load_all`** | 调 `load_all_from_path(nullptr)`，在默认路径下尝试加载 `libggml-<name>-*.so` 等；`NDEBUG` 时加载失败更安静。 |
| **`ggml_backend_score`** | 各后端 `.so` 可选导出；**分高优先**；**0 = 当前机器不适用**；CPU x86 多变体用 `cpu-feats.cpp` 里 ISA 检测 + `GGML_BACKEND_DL_SCORE_IMPL`。 |
| **`libggml-<name>-*.so` 从哪来** | **`GGML_BACKEND_DL=ON` + `BUILD_SHARED_LIBS=ON`** 时 CMake **`add_library(... MODULE)`** 生成；CPU 变体名如 `ggml-cpu-haswell` → `libggml-cpu-haswell.so`；默认 `GGML_BACKEND_DL` 常为 OFF，后端可能静态链进主程序。 |
| **注册新后端** | 实现 `ggml_backend_reg`；静态：`#ifdef GGML_USE_ZZZ` + `register_backend`；动态：`ggml_backend_init` + `GGML_BACKEND_DL_IMPL`；或 **`ggml_backend_register`**。 |
| **CUDA 何时「跑起来」** | 首次 **`ggml_cuda_info()`** → **`ggml_cuda_init()`** → **`cudaGetDeviceCount`**；真正算子在 **`llama_decode` 图执行**里。需 **驱动 + 兼容的 CUDA Runtime**。 |

---

## 12. `llama_init_from_model` 做什么

- **参数校验**（model 非空、`n_batch`/`n_ubatch`、flash_attn 与量化 KV 等）后 **`new llama_context(*model, params)`**。
- **不负责**再读 GGUF；建立 **KV、调度器、图缓冲、线程池** 等**每 context 一份**的运行时状态。
- 旧名 **`llama_new_context_with_model`** 已弃用，行为相同。

---

## 后续可记在这里

- 日期 / 新读的文件 / 仍模糊的函数名 / 下一步计划（例如：只跟一条 `build_graph` 的 Llama 路径）。
