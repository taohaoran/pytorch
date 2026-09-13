# CPU 向量化内核（vec-kernels）

> 本文是 `backends` 域下的叶子子系统文档。域级总览见 `../backends.md`。
> 本文聚焦 **CPU 向量化内核与跨 SIMD ISA 抽象（Vec256/Vec512/SVE/VSX）**；
> CUDA 后端见 `../cuda-backend/cuda-backend.md`，其他加速器见 `../other-accelerators/other-accelerators.md`。
>
> 源码基准：PyTorch 主分支，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| CPU 算子内核 | 64 个 `.cpp` 内核实现各类算子的 CPU 版本 | `aten/src/ATen/native/cpu/*.cpp` |
| 向量化循环 | 内核用向量化循环（Vec 批量处理）替代逐元素循环 | `aten/src/ATen/native/cpu/` |
| Vec 抽象基类 | `vec_base.h` 定义跨宽度的向量化类型接口 | `aten/src/ATen/cpu/vec/vec_base.h` |
| Vec256 (SSE/AVX2) | 256 位向量化类型（float/double/int/half/bf16/...） | `aten/src/ATen/cpu/vec/vec256/*.h` |
| Vec512 (AVX512) | 512 位 AVX512 向量化类型 | `aten/src/ATen/cpu/vec/vec512/*.h` |
| ARM SVE | ARM Scalable Vector Extension 后端 | `aten/src/ATen/cpu/vec/sve/` |
| POWER VSX / s390x zarch | 其他架构 SIMD 后端 | `vec/vec256/vsx/`、`vec/vec256/zarch/` |
| 掩码向量化 | 掩码化向量化操作 | `vec_mask.h`、`vec256_mask.h`、`vec512_mask.h` |
| 向量化函数库 | `functional.h` 提供向量化 map/reduce 等原语 | `aten/src/ATen/cpu/vec/functional.h` |
| 精度转换 | 跨类型向量化转换 | `vec_convert.h`、`vec256_convert.h`、`vec512_convert.h` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `Vec256` | `vec/vec256/vec256.h` | 256 位向量化类型，封装 AVX2/SSE 指令 |
| `Vec512` | `vec/vec512/vec512_float.h` 等 | 512 位 AVX512 向量化类型 |
| `Vectorized<T>` 模板 | `vec_base.h` | 平台无关向量化接口 |
| `vec512/vec256` 分派 | 编译期 `__AVX2__/__AVX512F__` | 按编译 ISG 选实现 |

## 3. 关键调用链

1. **算子 CPU 执行**：ATen Dispatcher 按 device 为 CPU 选择 `native/cpu/*.cpp` 内核。
2. **向量化**：内核用 `Vectorized<T>`（Vec256/Vec512）批量加载/计算/存储，循环按 SIMD 宽度步长推进；尾部剩余元素标量处理。
3. **ISA 选择**：`vec256/vec512/sve/vsx/zarch` 由编译宏（`__AVX2__`、`__AVX512F__`、ARM/POWER 宏）在编译期选定同一抽象的不同实现，内核源码无需分支。

## 4. 配置项

| 配置 / 编译宏 | 行为 | 位置 |
|------|------|------|
| `__AVX2__` / `__AVX512F__` 等 | 编译期选择 SIMD 后端 | CMake 编译选项 |
| CPU 调度 | 运行时按 CPU 能力选择最优 ISA 实现 | build 系统 |

## 5. 错误与重试语义

- 纯计算内核，无网络/进程失败路径；数值错误不在本层处理。
- 不支持的 ISA 回退到标量实现（卡片说明），不抛错。

## 6. 并发细节

- 本层为无状态 CPU 内核；并行由上层 OpenMP/线程池（ATen 线程）调度，多线程处理不同数据块。
- 向量化利用 SIMD 宽度数据级并行，与线程级正交。

## 7. 系统边界

**In-Scope**
- `aten/src/ATen/cpu/vec/`：跨 ISA 向量化抽象
- `aten/src/ATen/native/cpu/`：CPU 算子内核

**Out-of-Scope**
- CUDA GPU 内核：`../cuda-backend`
- MPS/HIP/XPU：`../other-accelerators`
- CPU 通用缓存分配器（相邻分片 memory-allocator）

## 8. 与相邻子系统交互

- **上游**：ATen Dispatcher / 各 native 算子调用 CPU 内核。
- **下游**：依赖编译目标 CPU 的 SIMD 指令集；不直接调用外部库（BLAS 见 `BlasKernel.cpp`）。

## 9. 语言专项适配口径

本项目为 **C++（模板元编程）** 核心，不适用 Go/TS 专项。
- 用 C++ 模板在编译期把 `Vectorized<T>` 实例化为具体 ISA 类型（Vec256/Vec512/SVE），实现零开销多后端抽象。
- 数据并行以 SIMD 宽度 + OpenMP 线程两级表达，无 goroutine/actor 概念。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|------|------|------|------|
| CPU 向量化内核架构图 | `vec-kernels-architecture.html` | architecture | showcase |

JSON IR 源位于 `json/`。

![架构图](vec-kernels-architecture.html)
