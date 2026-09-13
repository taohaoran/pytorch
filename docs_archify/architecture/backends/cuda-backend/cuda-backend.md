# CUDA 后端（cuda-backend）

> 本文是 `backends` 域下的叶子子系统文档。域级总览见 `../backends.md`。
> 本文聚焦 **CUDA 算子 dispatch、CUDACachingAllocator、CUDAStream/CUDAGuard/CUDAEvent、PinnedMemory 分配**；
> CPU 向量化内核见 `../vec-kernels/vec-kernels.md`，其他加速器见 `../other-accelerators/other-accelerators.md`。
>
> 源码基准：PyTorch 主分支，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| CUDA 算子内核 | 221 个 `.cu` 内核实现各类算子的 GPU 版本 | `aten/src/ATen/native/cuda/*.cu` |
| 算子 dispatch | ATen Dispatcher 按 `ComputeType=CUDA` 路由到 CUDA 内核 | `aten/src/ATen/cuda/`、`aten/src/ATen/native/cuda/` |
| CUDACachingAllocator | 显存缓存分配器，按大小/流缓存块，避免反复 cudaMalloc | `c10/cuda/CUDACachingAllocator.{h,cpp}` |
| emptyCache | 释放未使用的缓存块回驱动 | `c10/cuda/CUDACachingAllocator.h:366` |
| CUDAStream | 封装 `cudaStream_t`，表示 GPU 执行流 | `c10/cuda/CUDAStream.{h,cpp}` |
| CUDAGuard | RAII 切换当前 CUDA 设备 | `c10/cuda/CUDAGuard.h` |
| CUDAEvent | GPU 事件，用于流间/主机-GPU 同步 | `aten/src/ATen/cuda/CUDAEvent.h`、`c10/cuda/CUDAEvent.h` |
| PinnedMemory 分配 | 页锁定主机内存缓存分配器 | `aten/src/ATen/cuda/CachingHostAllocator.{h,cpp}` |
| cuBLAS/cuDNN 句柄 | 线性代数/卷积句柄池 | `aten/src/ATen/cuda/CUDABlas.*`、`CublasHandlePool.cpp` |
| CUDA Graph | 捕获/重放 CUDA 图 | `aten/src/ATen/cuda/CUDAGraph.*` |
| 设备断言 | GPU 端设备断言上报 | `c10/cuda/CUDADeviceAssertion*.{h,cpp}` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `CUDACachingAllocator` | `CUDACachingAllocator.h` | 显存分配器抽象，`allocate/raw_alloc/emptyCache` |
| `CUDAStream` | `c10/cuda/CUDAStream.h` | GPU 流封装 |
| `CUDAGuard` | `c10/cuda/CUDAGuard.h` | 设备上下文 RAII |
| `CUDAEvent` | `CUDAEvent.h` | 同步事件 |
| `CUDAGeneratorImpl` | `aten/src/ATen/cuda/CUDAGeneratorImpl.*` | CUDA 随机数生成器状态 |

## 3. 关键调用链

1. **算子 CUDA 执行**：ATen Dispatcher 按张量 device 为 CUDA 选择 CUDA 内核（`native/cuda/*.cu`）。
2. **显存分配**：内核需工作区/输出时经 `CUDACachingAllocator::allocate` 在当前设备当前流上取一块缓存；空闲块命中即复用，未命中才向驱动 `cudaMalloc`。
3. **内核启动**：在当前 `CUDAStream` 上 `<<<>>>` 启动 kernel；`CUDAGuard` 保证线程切到正确设备；`CUDAEvent` 用于多流依赖同步。
4. **页锁定内存**：host→device 拷贝用 `CachingHostAllocator` 分配 pinned 内存，提升拷贝带宽。

## 4. 配置项

| 配置 / 环境变量 | 默认 / 行为 | 位置 |
|------|------|------|
| `PYTORCH_CUDA_ALLOC_CONF` | 配置分配器：max_split_size_mb、expandable_segments、backend 等 | `c10/cuda/CUDACachingAllocator.cpp:439` |
| `CUDA_VISIBLE_DEVICES` | 可见 GPU 列表（由 CUDA 驱动消费） | 外部组件 |
| `CUDA_MODULE_LOADING` | 模块加载策略（驱动消费） | 外部组件 |

## 5. 错误与重试语义

- **分配失败**：缓存分配器无法满足时抛 CUDA OOM；不自动重试，用户需 `empty_cache()` 或减小 batch。
- **内核错误**：CUDA 异步错误由驱动在同步点发现，经 `CUDAException` 上抛；设备断言（CUDADeviceAssertion）把 GPU 端断言带回 host。
- 无应用层重试；训练进程崩溃由 `../../distributed/elastic/elastic.md` 重启。

## 6. 并发细节

- **多流并发**：同一设备上多个 `CUDAStream` 可并发执行不同 kernel；分配器按流跟踪块，跨流复用需事件同步。
- **设备上下文**：`CUDAGuard` 保证线程局部当前设备；多线程各自有 current stream。
- **分配器锁**：缓存池用互斥保护块查找/拆分；`emptyCache` 在持有锁时释放。

## 7. 系统边界

**In-Scope**
- `aten/src/ATen/cuda/`：CUDA 算子封装、BLAS/Graph/Event
- `c10/cuda/`：CUDA 分配器、流、设备、异常

**Out-of-Scope**
- CUDA Runtime/Driver、cuBLAS、cuDNN 库本体（外部组件）
- CPU 向量化内核：`../vec-kernels`；MPS/HIP/XPU：`../other-accelerators`
- 通用分配器接口与 CPU 缓存分配器（相邻分片 memory-allocator）

## 8. 与相邻子系统交互

- **上游**：ATen Dispatcher 与各算子（native 层）按 dispatch key 进入 CUDA 实现。
- **下游**：调用外部 CUDA Runtime/Driver 做显存与 kernel 启动。
- **协作**：c10d NCCL 后端（`../../distributed/c10d/c10d.md`）在 CUDA 流上做集合通信。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++/CUDA 核心**，不适用 Go/TS 专项。
- C++ 侧：分配器用策略/缓存池模式；并发以 CUDA stream 为单位，而非 goroutine；RAII（Guard）管理设备上下文。
- CUDA kernel 用 `.cu`（CUDA 扩展），经 CMake 编译，是典型 GPU 后端适配。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|------|------|------|------|
| CUDA 后端架构图 | `cuda-backend-architecture.html` | architecture | standard（showcase 因多分支 fan-out 标签间距自动回退） |

JSON IR 源位于 `json/`。算子 dispatch+流执行的时序已在调用链文字化描述。

![架构图](cuda-backend-architecture.html)
