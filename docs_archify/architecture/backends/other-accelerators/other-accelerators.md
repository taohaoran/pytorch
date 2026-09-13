# 非 CUDA 加速后端（other-accelerators）

> 本文是 `backends` 域下的叶子子系统文档。域级总览见 `../backends.md`。
> 本文聚焦 **MPS（Apple）、HIP（AMD）、XPU（Intel）、Metal 数学头** 等非 CUDA 加速器的 dispatch 适配层；
> CUDA 后端见 `../cuda-backend/cuda-backend.md`，CPU 向量化见 `../vec-kernels/vec-kernels.md`。
>
> 源码基准：PyTorch 主分支，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| MPS 设备封装 | `MPSDevice`/`MPSStream`/`MPSEvent`/`MPSGuardImpl` | `aten/src/ATen/mps/*.{h,mm}` |
| MPS 内存分配 | `MPSAllocator` 封装 Metal 显存分配 | `aten/src/ATen/mps/MPSAllocator.{h,mm}` |
| MPS 算子 | `native/mps/operations/*.mm` 各算子（Activation/Blas/Binary/Attention...） | `aten/src/ATen/native/mps/operations/` |
| Metal 着色器库 | Metal shader 编译/加载 | `aten/src/ATen/native/mps/MetalShaderLibrary.h` |
| MPSGraph Sequoia | 新 MPS 图算子 | `native/mps/MPSGraphSequoiaOps.h` |
| HIP 算子 | AMD ROCm 算子（GEMM 等） | `aten/src/ATen/native/hip/*.hip` |
| XPU 设备封装 | `XPUCachingAllocator`/`XPUEvent`/`XPUDeviceProp` | `c10/xpu/XPU*.{h,cpp}` |
| XPU 算子 | Intel XPU 内核 | `aten/src/ATen/native/xpu/` |
| XPU 对等访问 | P2P 访问 | `c10/xpu/PeerToPeerAccess.{h,cpp}` |
| Metal 数学头 | GPU 数学函数（expm1f/igamma/reduction/random...） | `c10/metal/*.h` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `MPSAllocator` | `aten/src/ATen/mps/MPSAllocator.h` | MPS 显存分配 |
| `MPSDevice`/`MPSStream`/`MPSEvent`/`MPSGuardImpl` | `aten/src/ATen/mps/` | MPS 设备/流/事件/上下文 RAII |
| `XPUCachingAllocator` | `c10/xpu/XPUCachingAllocator.h` | XPU 显存缓存分配 |
| dispatch key | ATen 分发系统 | 按 `mps`/`hip`/`xpu` device 路由到对应后端 |

## 3. 关键调用链

1. **MPS 算子执行**：dispatch key 为 `mps` → `native/mps/operations/*.mm` 实现 → 通过 `MPSStream` 提交到 Metal/MPS 框架 → `MPSAllocator` 管理显存。
2. **HIP 执行**：HIP 内核经 hipify 从 CUDA 内核派生（`native/hip/*.hip`），在 AMD GPU 上运行，复用 CUDA 风格 dispatch。
3. **XPU 执行**：dispatch key `xpu` → `native/xpu/` 内核 + `c10/xpu` 分配器/事件 → Level Zero/SYCL 运行时。

## 4. 配置项

| 配置 | 行为 | 位置 |
|------|------|------|
| `PYTORCH_ENABLE_MPS_FALLBACK` | MPS 不支持时回退 CPU | `aten/src/ATen/mps/MPSFallback.mm` |
| 设备可见性环境变量 | 各加速器厂商运行时消费 | 外部组件 |

## 5. 错误与重试语义

- MPS 不支持的算子按 fallback 策略回退 CPU（`MPSFallback.mm`），不抛致命错。
- HIP/XPU 运行时错误经各后端 exception 上抛；无应用层重试。

## 6. 并发细节

- 各后端各自封装流/事件（MPSStream、XPUEvent），模型与 CUDA 后端一致：流为并发执行单位。
- MPS 用 Objective-C++ 桥接 Metal command queue。

## 7. 系统边界

**In-Scope**
- `aten/src/ATen/mps/`、`aten/src/ATen/native/mps/`：MPS
- `aten/src/ATen/native/hip/`：HIP 算子
- `c10/xpu/`、`aten/src/ATen/native/xpu/`：XPU
- `c10/metal/`：Metal 数学头

**Out-of-Scope**
- Metal/MPS 框架、ROCm/HIP 运行时、Level Zero/SYCL（外部组件）
- CUDA 后端：`../cuda-backend`

## 8. 与相邻子系统交互

- **上游**：ATen Dispatcher 按 device type 选择后端。
- **下游**：MPS→Metal/MPS 框架；HIP→ROCm；XPU→Level Zero/SYCL（均外部）。

## 9. 语言专项适配口径

本项目为 **C++/Obj-C++/HIP/SYCL 多语言核心**，不适用 Go/TS 专项。
- MPS 用 Objective-C++（.mm）桥接 Apple 框架；HIP 复用 CUDA 代码经 hipify；XPU 用 SYCL/Level Zero。
- 各后端以同一 dispatch 抽象接入 ATen，外部框架标注为外部组件。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|------|------|------|------|
| 非 CUDA 加速器架构图 | `other-accelerators-architecture.html` | architecture | showcase |

JSON IR 源位于 `json/`。

![架构图](other-accelerators-architecture.html)
