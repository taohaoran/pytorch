# 通用算子实现与 TensorIterator（aten-native）

> 本文是 `core` 域下的叶子子系统文档。域级总览见 `../core.md`。
> 本文只展开 **CPU 通用算子（元素级/二元/归约）如何借助 TensorIterator 做广播形状推导与并行遍历**；不重复展开：
> - 算子如何被 Dispatcher 选中 → 见 `../aten-dispatch/aten-dispatch.md`
> - 张量/Storage 等底层类型 → 见 `../c10-core/c10-core.md`
>
> 源码基准：PyTorch（C++ 核心），commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`，目录 `aten/src/ATen/TensorIterator.*`、`aten/src/ATen/native/`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| TensorIterator 构建管线 | 从输入/输出算子配置推导出广播后形状、strides、dtype、内存重叠，并合并/coalesce 维度 | `aten/src/ATen/TensorIterator.cpp`（`build` 管线第 622-635 行）、`TensorIterator.h` |
| TensorIteratorConfig 流式配置器 | 链式 `add_output/add_input/check_all_same_dtype/...` 构建迭代器 | `TensorIterator.h:779`（`class TORCH_API TensorIteratorConfig`） |
| 构建阶段（私有） | populate_operands → mark_resize_outputs → compute_mem_overlaps → compute_shape → compute_strides → compute_types → fast_set_up | `TensorIterator.h:623-635` |
| CPU 元素级遍历 cpu_kernel | 对迭代器逐元素调用 lambda，经 `at::parallel_for` 多线程并行；默认 grain=GRAIN_SIZE | `native/cpu/Loops.h:304`（`cpu_kernel`） |
| CPU 向量化遍历 cpu_kernel_vec | 同时提供标量 lambda 与 SIMD(AVX) lambda，自动选向量化路径 | `native/cpu/Loops.h:348`（`cpu_kernel_vec`） |
| 多输出遍历 cpu_kernel_multiple_outputs | 一次遍历写多个输出张量 | `native/cpu/Loops.h:337` |
| 归约并行 parallel_reduce | 二维分块归约，各块局部归约后合并 | `TensorIterator.h:450`（`parallel_reduce`）、`native/ReduceOps.cpp:1894` |
| 一元算子 UnaryOps | abs/neg/sqrt/exp/log 等逐元素一元 | `native/UnaryOps.cpp` |
| 二元算子 BinaryOps | add/sub/mul/div 等；`add_cpu` 经 TensorIteratorConfig 构建 | `native/BinaryOps.cpp:1575`（`add_cpu`） |
| 逐点算子 PointwiseOps | 其他 pointwise 组合 | `native/PointwiseOps.cpp` |
| 归约算子 ReduceOps/ReduceAllOps | sum/mean/max/min 及全行归约 | `native/ReduceOps.cpp`、`ReduceAllOps.cpp` |
| GPU 遍历 gpu_kernel | CUDA 侧对应遍历（在 `native/cuda/`，本叶子仅引用其契约） | `native/cuda/`（外部后端实现） |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `at::TensorIterator` / `TensorIteratorBase` | `TensorIterator.h` | 描述一次元素级/归约运算的"计算网格"：shape/strides/各操作数 data_ptr/类型；提供 `data_ptr(i)`、`n_dim`、coalesced 视图 |
| `at::TensorIteratorConfig` | `TensorIterator.h:779` | 流式 builder；`build()` 返回配置好的 iterator |
| `at::native::cpu_kernel` | `cpu/Loops.h:304` | 模板函数：把 iter 的遍历委托给 `at::parallel_for` 调用标量 lambda |
| `at::native::cpu_kernel_vec` | `cpu/Loops.h:348` | 带 SIMD 向量函数的版本 |
| `at::parallel_reduce` | `TensorIterator.h:450` | 二维分块归约调度器 |
| `at::internal::GRAIN_SIZE` | `native/cpu/` | 并行分块的默认最小粒度 |
| `TensorIterator::coalesce_dimensions` | `TensorIterator.cpp` | 合并连续维度以减少循环层数 |

## 3. 关键调用链

**调用链一：CPU `add(Tensor a, Tensor b)` 的元素级实现**

1. Dispatcher（见 aten-dispatch）选中 `add` 的 CPU 内核，进入 `add_cpu`（`BinaryOps.cpp:1575`）。
2. 内核构造 `TensorIteratorConfig().add_output(out).add_input(a).add_input(b).build()`：
   - `populate_operands` 登记各操作数（`TensorIterator.h:623`）；
   - `compute_shape` 做广播对齐各操作数形状（第 627 行）；
   - `compute_strides` 推导出广播后的 strides，对 0/1 维做 stride=0 广播（第 628 行）；
   - `compute_types` 做类型提升与 dtype 统一（第 631 行）；
   - `coalesce_dimensions` 合并可连续维度。
3. 调用 `cpu_kernel(iter, [](Scalar a, Scalar b) { return a+b; })`（`Loops.h:304`）。
4. `cpu_kernel` 内部按 `GRAIN_SIZE` 切分迭代区间，经 `at::parallel_for` 分发到多线程，每块调用 lambda；每个线程通过 `iter.data_ptr` 读写各操作数。

**调用链二：归约 sum（parallel_reduce）**

1. `sum_cpu` 构造 TensorIterator 描述归约维度。
2. 调用 `at::parallel_reduce(...)`（`ReduceOps.cpp:1894`）：全局切分为二维块，每块先局部归约，再跨块合并。

**调用链三：向量化快路径**

1. 若算子提供 `cpu_kernel_vec` 的 SIMD lambda，Loops 自动检测数据对齐与连续性，满足条件时走 AVX/AVX512 向量化循环（DispatchStub 选指令集，见 aten-dispatch）。

## 4. 配置项

| flag / option | 默认 / 行为 | 位置 |
|---------------|-------------|------|
| `grain_size` | `cpu_kernel` 默认 `at::internal::GRAIN_SIZE`；过大退化为串行，过小线程开销大 | `Loops.h:304` |
| `check_dynamic_casting` | 默认开启，元素级做动态类型转换检查 | `Loops.h:304` |
| `HAVE_AVX2/AVX512_CPU_DEFINITION` | 编译期控制 cpu_kernel_vec 是否启用对应指令集 | `Loops.h` / CMake |
| `torch.set_num_threads` | 控制 `at::parallel_for` 线程池大小 | 线程池（c10/core/thread_pool） |

## 5. 错误与重试语义

- **形状广播失败**：`compute_shape` 阶段对不可广播形状抛 `RuntimeError`（形状不匹配），无重试。
- **类型提升失败**：`compute_types` 对无法安全转换的 dtype 抛错。
- **并行异常传播**：`parallel_for`/`parallel_reduce` 中任意线程抛异常会被捕获并重新抛出到调用线程，无自动重试。
- 无 IO/网络失败路径。

## 6. 并发细节

- **多元素级并行**：`cpu_kernel` 经 `at::parallel_for` 把元素区间切分到 OpenMP/线程池多线程；每个线程独立读写自己的区间，操作数 data_ptr 只读共享、输出区间互不重叠（TensorIterator 在 `compute_mem_overlaps` 阶段检测输出与输入重叠）。
- **归约分块**：`parallel_reduce` 二维分块，局部归约无共享写，合并阶段单线程或加锁聚合。
- **线程安全**：TensorIterator 本身在 build 完成后只读；多线程遍历共享同一 iter 只读。
- **GIL**：CPU 算子在 C++ 侧执行，期间释放/不依赖 GIL；Python 回调（如自定义 elementwise）另有处理。
- **无锁**：元素级并行无跨线程临界区；归约合并阶段是数据依赖串行点。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `aten/src/ATen/TensorIterator.*`（广播形状推导与遍历网格）
- `aten/src/ATen/native/` 通用算子（UnaryOps/BinaryOps/PointwiseOps/ReduceOps/...）
- `aten/src/ATen/native/cpu/Loops.h`（cpu_kernel 并行遍历）

**Out-of-Scope（不在本仓库源码内）**
- CUDA/MPS/CPU(后端) 具体 kernel → `native/cuda/`、`native/mps/`（后端实现，cuda-backend 分片覆盖 CUDA）
- BLAS/cuBLAS/MKL 等第三方库 → 外部组件
- 算子分发如何选中本实现 → `../aten-dispatch/aten-dispatch.md`
- 反向（autograd）→ autograd 域

## 8. 与相邻子系统交互

- 上游 → 本叶子：Dispatcher（aten-dispatch）按 DispatchKey=CPU 选中本叶子内核。
- 本叶子 → 下游：
  - `TensorIterator` 用 `TensorImpl::sizes/strides/storage`（c10-core）读取操作数元数据。
  - `cpu_kernel` → `at::parallel_for`（线程池）→ 各线程调 lambda。
  - 数值底层可能调用 BLAS（BlasKernel.cpp，经 DispatchStub 选后端）。
- 方向：Dispatcher → 本叶子算子内核 → TensorIterator 网格 → cpu_kernel 并行 → lambda 计算。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，不适用 Go/TS 专项。本叶子适配口径：
- **C++ 模板 lambda 复用**：`cpu_kernel(iter, lambda)` 用模板把"遍历调度"与"元素计算"解耦，同一套并行框架服务数百个算子。
- **数据并行而非任务并行**：TensorIterator 是"数据并行迭代器"，把多维张量 flatten/coalesce 为一维遍历区间，再分块到线程；这与 Go 的 goroutine worker pool 范式不同。
- **SIMD 特化**：`cpu_kernel_vec` 是 C++ 模板 + 编译期指令集宏的向量化快路径，DispatchStub 在运行期按 CPU 特性选 AVX2/AVX512 实例。
- **无 GIL**：C++ 算子执行期不持有 GIL。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| TensorIterator 与算子架构图 | `aten-native-architecture.html` | architecture | showcase |
| add_cpu 构建+并行遍历数据流图 | `aten-native-dataflow.html` | dataflow | standard（降档披露：多行节点中文 label 宽度/垂直流标签在 showcase 严格布局校验下多次迭代仍触发重叠，按规则降 standard 渲染成功） |

- JSON IR 源文件：`json/` 下对应两个文件。
- 说明：构建管线是分阶段数据流，用 dataflow；算子分类与 cpu_kernel 关系用 architecture。本叶子无跨进程消息时序，不补 sequence。
