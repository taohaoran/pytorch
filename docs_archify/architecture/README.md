# PyTorch 系统架构文档

> 基于 PyTorch 源码（commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`，2026-09-12）深度分析产出，
> 覆盖系统级、8 个域、30 个叶子子系统的功能、问题域、系统边界、架构图、时序图与数据流图。
> 所有图表由 archify 渲染为自包含交互式 HTML。

## 文档导航

### 系统级

| 文档 | 说明 | 图表 |
|------|------|------|
| [system-overview.md](system-overview.md) | 功能总览、解决的问题、系统边界、测试口径、语言适配口径 | [系统架构图](system-architecture.html) · [前向反向时序图](system-sequence.html) · [编译数据流图](system-dataflow.html) |

---

### core 域（`core/`）

核心张量基础设施：c10 核心抽象、ATen 算子分发、native 算子实现、内存分配器。

| 子系统 | 文档 | 架构图 | 时序图 / 数据流图 |
|--------|------|--------|-------------------|
| 域总览 | [core.md](core/core.md) | — | — |
| c10-core | [c10-core.md](core/c10-core/c10-core.md) | [架构图](core/c10-core/c10-core-architecture.html) | [数据流图](core/c10-core/c10-core-dataflow.html) |
| aten-dispatch | [aten-dispatch.md](core/aten-dispatch/aten-dispatch.md) | [架构图](core/aten-dispatch/aten-dispatch-architecture.html) | [时序图](core/aten-dispatch/aten-dispatch-sequence.html) · [数据流图](core/aten-dispatch/aten-dispatch-dataflow.html) |
| aten-native | [aten-native.md](core/aten-native/aten-native.md) | [架构图](core/aten-native/aten-native-architecture.html) | [数据流图](core/aten-native/aten-native-dataflow.html) |
| memory-allocator | [memory-allocator.md](core/memory-allocator/memory-allocator.md) | [架构图](core/memory-allocator/memory-allocator-architecture.html) | [数据流图](core/memory-allocator/memory-allocator-dataflow.html) |

### autograd 域（`autograd/`）

自动微分：C++ 反向引擎与计算图执行、Python 侧 Function API 与 grad 模式。

| 子系统 | 文档 | 架构图 | 时序图 / 数据流图 |
|--------|------|--------|-------------------|
| 域总览 | [autograd.md](autograd/autograd.md) | — | — |
| autograd-engine | [autograd-engine.md](autograd/autograd-engine/autograd-engine.md) | [架构图](autograd/autograd-engine/autograd-engine-architecture.html) | [时序图](autograd/autograd-engine/autograd-engine-sequence.html) |
| autograd-python | [autograd-python.md](autograd/autograd-python/autograd-python.md) | [架构图](autograd/autograd-python/autograd-python-architecture.html) | [数据流图](autograd/autograd-python/autograd-python-dataflow.html) |

### python-frontend 域（`python-frontend/`）

Python 前端与 API 面：Tensor API、nn.Module、优化器、数据加载、序列化、设备上下文。

| 子系统 | 文档 | 架构图 | 时序图 / 数据流图 |
|--------|------|--------|-------------------|
| 域总览 | [python-frontend.md](python-frontend/python-frontend.md) | — | — |
| torch-api | [torch-api.md](python-frontend/torch-api/torch-api.md) | [架构图](python-frontend/torch-api/torch-api-architecture.html) | — |
| nn-modules | [nn-modules.md](python-frontend/nn-modules/nn-modules.md) | [架构图](python-frontend/nn-modules/nn-modules-architecture.html) | — |
| optim | [optim.md](python-frontend/optim/optim.md) | [架构图](python-frontend/optim/optim-architecture.html) | — |
| data-loading | [data-loading.md](python-frontend/data-loading/data-loading.md) | [架构图](python-frontend/data-loading/data-loading-architecture.html) | [时序图](python-frontend/data-loading/data-loading-sequence.html) |
| serialization | [serialization.md](python-frontend/serialization/serialization.md) | [架构图](python-frontend/serialization/serialization-architecture.html) | — |
| device-context | [device-context.md](python-frontend/device-context/device-context.md) | [架构图](python-frontend/device-context/device-context-architecture.html) | — |

### compile-graph 域（`compile-graph/`）

图编译与导出：TorchScript JIT、FX 图 IR、ExportedProgram 导出、ONNX 导出。

| 子系统 | 文档 | 架构图 | 数据流图 |
|--------|------|--------|----------|
| 域总览 | [compile-graph.md](compile-graph/compile-graph.md) | — | — |
| torchscript-jit | [torchscript-jit.md](compile-graph/torchscript-jit/torchscript-jit.md) | [架构图](compile-graph/torchscript-jit/torchscript-jit-architecture.html) | — |
| fx | [fx.md](compile-graph/fx/fx.md) | [架构图](compile-graph/fx/fx-architecture.html) | — |
| export | [export.md](compile-graph/export/export.md) | [架构图](compile-graph/export/export-architecture.html) | — |
| onnx | [onnx.md](compile-graph/onnx/onnx.md) | — | [数据流图](compile-graph/onnx/onnx-dataflow.html) |

### dynamo-inductor 域（`dynamo-inductor/`）

动态编译栈：TorchDynamo 字节码追踪、Inductor 代码生成、编译缓存、算子分解。

| 子系统 | 文档 | 架构图 | 时序图 |
|--------|------|--------|--------|
| 域总览 | [dynamo-inductor.md](dynamo-inductor/dynamo-inductor.md) | — | — |
| dynamo | [dynamo.md](dynamo-inductor/dynamo/dynamo.md) | [架构图](dynamo-inductor/dynamo/dynamo-architecture.html) | [时序图](dynamo-inductor/dynamo/dynamo-sequence.html) |
| inductor | [inductor.md](dynamo-inductor/inductor/inductor.md) | [架构图](dynamo-inductor/inductor/inductor-architecture.html) | — |
| compile-cache | [compile-cache.md](dynamo-inductor/compile-cache/compile-cache.md) | [架构图](dynamo-inductor/compile-cache/compile-cache-architecture.html) | — |
| decomposition | [decomposition.md](dynamo-inductor/decomposition/decomposition.md) | [架构图](dynamo-inductor/decomposition/decomposition-architecture.html) | — |

### distributed 域（`distributed/`）

分布式训练：c10d 通信库、DDP/FSDP 数据并行、RPC 与 DTensor、elastic 容错启动。

| 子系统 | 文档 | 架构图 | 时序图 |
|--------|------|--------|--------|
| 域总览 | [distributed.md](distributed/distributed.md) | — | — |
| c10d | [c10d.md](distributed/c10d/c10d.md) | [架构图](distributed/c10d/c10d-architecture.html) | — |
| ddp-fsdp | [ddp-fsdp.md](distributed/ddp-fsdp/ddp-fsdp.md) | [架构图](distributed/ddp-fsdp/ddp-fsdp-architecture.html) | [时序图](distributed/ddp-fsdp/ddp-fsdp-sequence.html) |
| rpc-dtensor | [rpc-dtensor.md](distributed/rpc-dtensor/rpc-dtensor.md) | [架构图](distributed/rpc-dtensor/rpc-dtensor-architecture.html) | — |
| elastic | [elastic.md](distributed/elastic/elastic.md) | [架构图](distributed/elastic/elastic-architecture.html) | [时序图](distributed/elastic/elastic-sequence.html) |

### backends 域（`backends/`）

后端与内核：CUDA 后端、CPU 向量化内核、MPS/HIP/XPU/Metal 加速后端、性能分析器。

| 子系统 | 文档 | 架构图 | 数据流图 |
|--------|------|--------|----------|
| 域总览 | [backends.md](backends/backends.md) | — | — |
| cuda-backend | [cuda-backend.md](backends/cuda-backend/cuda-backend.md) | [架构图](backends/cuda-backend/cuda-backend-architecture.html) | — |
| vec-kernels | [vec-kernels.md](backends/vec-kernels/vec-kernels.md) | [架构图](backends/vec-kernels/vec-kernels-architecture.html) | — |
| other-accelerators | [other-accelerators.md](backends/other-accelerators/other-accelerators.md) | [架构图](backends/other-accelerators/other-accelerators-architecture.html) | — |
| profiler | [profiler.md](backends/profiler/profiler.md) | [架构图](backends/profiler/profiler-architecture.html) | [数据流图](backends/profiler/profiler-dataflow.html) |

### extensions 域（`extensions/`）

扩展与函数变换：自定义算子注册与 C++ 扩展构建、functorch 函数变换。

| 子系统 | 文档 | 架构图 |
|--------|------|--------|
| 域总览 | [extensions.md](extensions/extensions.md) | — |
| custom-ops | [custom-ops.md](extensions/custom-ops/custom-ops.md) | [架构图](extensions/custom-ops/custom-ops-architecture.html) |
| functorch | [functorch.md](extensions/functorch/functorch.md) | [架构图](extensions/functorch/functorch-architecture.html) |

---

## 产出统计

| 层级 | MD 文档 | HTML 图 | JSON IR |
|------|---------|---------|---------|
| 系统级 | 2（README + system-overview） | 3 | 3 |
| 域总览 | 8 | — | — |
| 叶子子系统 | 30 | 42 | 42 |
| **合计** | **40** | **45** | **45** |

## 覆盖范围与说明

- **叶子数**：30 个，覆盖 8 个域，与规划一致
- **third_party/**：vendored 上游依赖，视为外部组件标注"不在本仓库源码内"，不拆叶子
- **caffe2/**：遗留模块，归入系统边界说明，不深入分析
- **test/**：1965 个测试文件，测试口径并入系统级总览，不单独成叶
- **语言适配口径**：Python + C++ 混合代码库，不套用 Go/TS 专项；C++ 侧分析线程池/互斥/依赖边界，Python 侧按能力缝归组。详见 [system-overview.md](system-overview.md) 第 9 节
- **质量档位**：系统级 3 图均为 standard（组件多、跨层连接复杂，showcase 严格布局未通过）；叶子级图含 showcase 与 standard 混合，各叶子 MD 第 10 小节如实披露
- **源码基准**：commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`，所有路径可追溯至该版本
