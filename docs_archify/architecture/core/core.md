# core 域总览

> 本域包含以下叶子子系统；各叶子详情见对应文档。
> 源码基准：PyTorch（C++ 核心），commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 域职责

`core` 域是 PyTorch C++ 核心的最底层与算子分发/执行骨架：定义张量的底层抽象类型、把算子调用按 DispatchKey 路由到具体内核、用 TensorIterator 做广播形状推导与并行遍历、并提供内存分配接口与 StorageImpl 生命周期管理。它承上启下——向上服务 Python 前端与 TorchScript，向下调用各后端（CPU/CUDA/...）数值实现。

核心代码路径：
- 类型抽象：`c10/core/`
- 分发与注册：`aten/src/ATen/core/dispatch/`、`op_registration/`
- 通用算子：`aten/src/ATen/native/`、`aten/src/ATen/TensorIterator.*`
- 内存：`c10/core/Allocator.*`、`CPUAllocator.*`、`StorageImpl.*`

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序图 / 数据流图 | 职责一句话 |
|------|------|--------|-------------------|-----------|
| c10-core | [c10-core.md](c10-core/c10-core.md) | [架构图](c10-core/c10-core-architecture.html) | [数据流](c10-core/c10-core-dataflow.html) | Tensor/Storage/DispatchKey/Device 等核心抽象类型定义 |
| aten-dispatch | [aten-dispatch.md](aten-dispatch/aten-dispatch.md) | [架构图](aten-dispatch/aten-dispatch-architecture.html) | [时序](aten-dispatch/aten-dispatch-sequence.html) · [数据流](aten-dispatch/aten-dispatch-dataflow.html) | c10 Dispatcher 按 DispatchKey 选内核的分发与注册机制 |
| aten-native | [aten-native.md](aten-native/aten-native.md) | [架构图](aten-native/aten-native-architecture.html) | [数据流](aten-native/aten-native-dataflow.html) | 通用算子实现与 TensorIterator 广播/并行遍历 |
| memory-allocator | [memory-allocator.md](memory-allocator/memory-allocator.md) | [架构图](memory-allocator/memory-allocator-architecture.html) | [数据流](memory-allocator/memory-allocator-dataflow.html) | Allocator 接口、CPU 分配器与 StorageImpl 生命周期 |

## 3. 域级机制细节

- **DispatchKey 贯穿全域**：c10-core 定义 DispatchKey/DispatchKeySet；aten-dispatch 用它查内核表；aten-native 在选中的 CPU 内核内用 TensorIterator 做数据并行；memory-allocator 的分配按 DeviceType（DispatchKey 的后端部分）注册。
- **调用链分层**：Python/TorchScript → Dispatcher（aten-dispatch）→ CPU 数值内核（aten-native）→ TensorIterator → cpu_kernel 并行；内存由 memory-allocator 提供。
- **类型→键→内核的映射**：TensorOptions（c10-core）→ computeDispatchKey → Dispatcher lookup → KernelFunction（aten-dispatch）→ native 实现（aten-native）。
- **依赖方向**：`c10`（c10-core）← `aten/core/dispatch`（aten-dispatch）← `aten/native`（aten-native）；memory-allocator 的 Allocator 接口被 c10-core 的 StorageImpl 持有，被 aten-native 算子触发。

## 4. 语言专项适配口径（域级）

本域为 **Python 前端 + C++ 核心**，**不适用** Go 专项（goroutine/Reconciler/informer）与 TS 专项（capability seam/workspace）。C++ 侧适配口径：
- 侵入式引用计数（`intrusive_ptr`）替代共享指针，便于跨语言边界裸指针传递。
- 模板化 unboxed 分发消除虚调用/boxing，是 eager 热路径性能关键。
- 数据并行（TensorIterator 分块 + `at::parallel_for`）而非任务并行 goroutine。
- Allocator 纯虚接口可插拔，后端策略（CPU 无缓存 vs CUDA 缓存）差异化。
