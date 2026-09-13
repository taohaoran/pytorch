# backends 计算后端域（backends）总览

> 本域包含以下叶子子系统；各叶子详情见对应文档。
> 源码基准：PyTorch 主分支，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 域职责

`backends` 域提供 PyTorch 的**硬件加速后端与性能观测**，覆盖：

- **CUDA 后端**：GPU 算子 dispatch、显存缓存分配器、CUDA 流/事件/设备上下文。
- **CPU 向量化内核**：跨 SIMD ISA（AVX2/AVX512/SVE/VSX/zarch）的向量化抽象与 CPU 算子。
- **非 CUDA 加速器**：MPS(Apple)、HIP(AMD)、XPU(Intel)、Metal 的 dispatch 适配层。
- **Profiler**：性能活动采集、内存分析与 kineto 桥接。

各后端按设备类型分离，互不重叠。

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序 / 数据流 | 职责一句话 |
|------|------|--------|-------------------|-----------|
| cuda-backend | [cuda-backend.md](cuda-backend/cuda-backend.md) | [架构图](cuda-backend/cuda-backend-architecture.html) | — | CUDA 算子 dispatch 与 CUDACachingAllocator/流 |
| vec-kernels | [vec-kernels.md](vec-kernels/vec-kernels.md) | [架构图](vec-kernels/vec-kernels-architecture.html) | — | CPU 向量化内核与跨 ISA Vec 抽象 |
| other-accelerators | [other-accelerators.md](other-accelerators/other-accelerators.md) | [架构图](other-accelerators/other-accelerators-architecture.html) | — | MPS/HIP/XPU/Metal 非 CUDA 后端适配 |
| profiler | [profiler.md](profiler/profiler.md) | [架构图](profiler/profiler-architecture.html) | [数据流图](profiler/profiler-dataflow.html) | 性能活动采集与 kineto 桥接 |

## 3. 域级机制细节

- **dispatch 统一入口**：ATen Dispatcher 按 device type（CUDA/CPU/MPS/HIP/XPU）路由到对应后端；各后端实现同一组算子接口。
- **内存分配器分层**：CUDA 用 CUDACachingAllocator，XPU 用 XPUCachingAllocator，MPS 用 MPSAllocator；通用分配器接口与 CPU 缓存分配器由相邻 memory-allocator 分片覆盖。
- **流为并发单位**：CUDA/MPS/XPU 都封装 stream/event/guard，模型一致；CPU 向量化用 SIMD 宽度 + OpenMP 线程两级并行。
- **观测横切**：profiler 在 dispatch/后端打点，跨所有后端采集事件，不绑定具体算子。

## 4. 语言专项适配口径（域级披露）

本项目为 **C++/CUDA/Obj-C++/HIP/SYCL 多语言核心**，不适用 Go/TS 专项。C++ 侧分析 c10/cuda 分配器与 CUDA stream 并发、模板元编程的跨 ISA Vec 抽象、各加速器后端的 dispatch 适配；外部框架（CUDA Runtime、Metal/MPS、ROCm、Level Zero、libkineto）一律标注"不在本仓库源码内"。

## 5. 质量档汇总

| 叶子 | 架构图 | 第二图 |
|------|--------|--------|
| cuda-backend | standard | — |
| vec-kernels | showcase | — |
| other-accelerators | showcase | — |
| profiler | showcase | 数据流 standard |
