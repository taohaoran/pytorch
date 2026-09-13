# 内存分配器与 StorageImpl 生命周期（memory-allocator）

> 本文是 `core` 域下的叶子子系统文档。域级总览见 `../core.md`。
> 本文只展开 **c10 Allocator 抽象接口的实现、CPU 分配器与缓存分配器基类、以及 StorageImpl 的内存生命周期**；不重复展开：
> - Allocator/DataPtr 接口定义本身 → 见 `../c10-core/c10-core.md`
> - CUDA 缓存分配器（CUDACachingAllocator）的具体池算法 → cuda-backend 分片
>
> 源码基准：PyTorch（C++ 核心），commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`，目录 `c10/core/Allocator.*`、`c10/core/CPUAllocator.*`、`c10/core/CachingDeviceAllocator.*`、`c10/core/StorageImpl.*`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| Allocator 纯虚接口 | `allocate(nbytes)->DataPtr` + `copy_data`；按 DeviceType 注册 | `c10/core/Allocator.h:180`（`struct C10_API Allocator`） |
| DataPtr 所有权契约 | 唯一指针 + deleter + device；Storage 持有 | `c10/core/Allocator.h:40` |
| CPU 默认分配器 CPUAllocator | 直接 `c10::alloc_cpu`（malloc），无块缓存；可插拔 | `c10/core/CPUAllocator.cpp:20`（`allocate`）、`CPUAllocator.h` |
| CPU 分配器获取/设置 | `GetCPUAllocator/SetCPUAllocator/GetDefaultCPUAllocator` | `CPUAllocator.h:41-50` |
| CPU 缓存分配器（实验） | `SetCPUCachingAllocator/GetCPUCachingAllocator`，当前主要供 StaticRuntime 使用 | `CPUAllocator.h:55-57` |
| 缓存分配器基类 CachingDeviceAllocator | 设备缓存分配器通用框架：Block/Pool/DeviceStats；CUDA 分配器继承之 | `c10/core/CachingDeviceAllocator.h`（`DeviceStats` 第 13 行、`BlockInfo` 第 83 行） |
| 内存统计 DeviceStats | 跟踪 reserved/allocated/peak/oom 次数等 | `CachingDeviceAllocator.h:13-63` |
| StorageImpl 持有缓冲区 | DataPtr + SymInt size_bytes + resizable + allocator 回指 | `c10/core/StorageImpl.h:55` |
| StorageImpl 生命周期 | intrusive_ptr 引用计数；析构/`release_resources` 释放 DataPtr | `StorageImpl.h:97,107` |
| 惰性物化钩子 materialize_fn_ | 写时复制（COW）/惰性分配：首次可变访问时触发物化 | `StorageImpl.h:384`（`maybe_materialize`） |
| CPU 内存上报 ProfiledCPUMemoryReporter | 把分配/释放/oom 上报给 profiler | `CPUAllocator.h:24` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `c10::Allocator` | `Allocator.h:180` | 纯虚基类：`allocate(size_t)=0`、`copy_data=0`；可选 `raw_deleter()`、`clone()` |
| `c10::DataPtr` | `Allocator.h:40` | 带 deleter/device 的唯一指针；StorageImpl 的 `data_ptr_` |
| `c10::CPUAllocator` | `CPUAllocator.cpp:20` | 默认 CPU 分配器实现：`allocate` 调 `c10::alloc_cpu`，`copy_data` 用 memcpy |
| `c10::CachingDeviceAllocator` | `CachingDeviceAllocator.h` | 缓存分配器基类：维护 free/used 块树与 pool；CUDA 等后端继承实现 |
| `c10::CachingDeviceAllocator::DeviceStats` | `CachingDeviceAllocator.h:13` | 内存统计：reserved/allocated/peak/oom 计数 |
| `c10::StorageImpl` | `StorageImpl.h:55` | 缓冲区唯一持有者；`data_ptr_`/`size_bytes_`/`allocator_`/`materialize_fn_` |
| `c10::SetAllocator/GetAllocator` | `Allocator.h:289-290` | 按 DeviceType 注册/取回分配器（初始化期） |
| `c10::SetCPUAllocator/GetCPUAllocator` | `CPUAllocator.h:41-44` | CPU 专用注册入口 |

## 3. 关键调用链

**调用链一：一次 CPU 张量分配**

1. 算子经 Dispatcher（aten-dispatch）选中 CPU 内核，需要新张量。
2. `GetAllocator(DeviceType::CPU)`（`Allocator.h:290`）→ `GetCPUAllocator()`（`CPUAllocator.h:41`）返回默认 `CPUAllocator` 单例。
3. `CPUAllocator::allocate(nbytes)`（`CPUAllocator.cpp:20`）调 `c10::alloc_cpu(nbytes)` 得到裸指针，包成 `DataPtr(ptr, device=CPU, deleter=free_cpu)`。
4. `make_storage_impl(...)`（`StorageImpl.h:427`）构造 `StorageImpl`，把 DataPtr 与 allocator 回指存入。
5. 包成 `Storage` → 构造 `TensorImpl`（c10-core）。

**调用链二：StorageImpl 释放**

1. 最后一个 `intrusive_ptr<StorageImpl>` 析构 → `~StorageImpl`（`StorageImpl.h:97`）。
2. `data_ptr_`（DataPtr）析构触发其 deleter（`free_cpu`）归还内存；注释指出析构不调 `release_resources`（第 105 行）。

**调用链三：惰性物化（COW/惰性分配）**

1. 一个 storage 设置了 `materialize_fn_`（如 COW 来源）。
2. 首次 `mutable_data()`（`StorageImpl.h:210`）命中 `has_mutable_data_ptr_check_` → `maybe_materialize()`（第 384 行）执行物化回调并清空钩子。
3. 此后普通路径。

## 4. 配置项

| flag / option | 默认 / 行为 | 位置 |
|---------------|-------------|------|
| `SetCPUAllocator priority` | 仅 ≥priority 可覆盖默认 CPU 分配器 | `CPUAllocator.h:44` |
| `SetCPUCachingAllocator` | 实验性 CPU 缓存分配器，供 StaticRuntime | `CPUAllocator.h:55` |
| `TORCH_ALLOC_CONF` / 环境变量 | 缓存分配器细分配置（后端池行为），CUDA 侧为主 | CachingDeviceAllocator 后端 |
| `caffe2_report_cpu_memory_usage` | 开启 CPU 内存使用上报 | `CPUAllocator.h:13` |

## 5. 错误与重试语义

- **OOM**：`alloc_cpu` 失败时由 `ProfiledCPUMemoryReporter::OutOfMemory`（`CPUAllocator.h:28`）记录并抛 std::bad_alloc/异常，无重试。
- **resizable storage 无 allocator**：`StorageImpl` 构造时 `resizable=true` 但 allocator 空则 `TORCH_INTERNAL_ASSERT`（`StorageImpl.h:71`）。
- **data_ptr 访问违规**：`throw_on_immutable_data_ptr_`/`throw_on_mutable_data_ptr_` 置位时 `data()/mutable_data()` 抛错（`StorageImpl.h:145,210`）。
- 无网络/IO 重试；缓存分配器的 oom 后可尝试 releaseCachedMemory 再分配（后端侧）。

## 6. 并发细节

- **分配器注册非线程安全**：`SetAllocator` 注释明确仅初始化期调用（`Allocator.h:282`）。
- **CPUAllocator 本身无锁**：默认 CPUAllocator 直接 malloc/free，无内部锁；malloc 自身线程安全。
- **缓存分配器有锁**：`CachingDeviceAllocator` 基类在 free/used 块表操作上用互斥保护（CUDA 实现细节，cuda-backend 分片）。
- **ProfiledCPUMemoryReporter 有 mutex**：`CPUAllocator.h:32` 的 `mutex_` 保护 `size_table_`。
- **StorageImpl 引用计数**：intrusive_ptr 原子计数，跨线程安全持有；对象内部字段非自动加锁。
- **无工作线程**：本叶子不启线程。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `c10/core/Allocator.*`（接口）
- `c10/core/CPUAllocator.*`（默认 CPU 分配器）
- `c10/core/CachingDeviceAllocator.h`（缓存分配器基类与统计）
- `c10/core/StorageImpl.*`（缓冲区生命周期）

**Out-of-Scope（不在本仓库源码内）**
- CUDA 缓存分配器具体池/段算法 → cuda-backend 分片（`c10/cuda/CUDACachingAllocator.cpp`）
- malloc/mmap 系统调用 → 操作系统内核，不在本仓库源码内
- TensorImpl 如何使用 Storage → `../c10-core/c10-core.md`
- 算子如何触发分配 → `../aten-native/aten-native.md`

## 8. 与相邻子系统交互

- 上游 → 本叶子：`aten-native` 算子内核在产出新张量时调用 `GetAllocator(CPU)->allocate`。
- 本叶子 → 下游：
  - `CPUAllocator::allocate` → `c10::alloc_cpu` → malloc（系统）。
  - `StorageImpl.data_ptr_` 的 deleter 归还到分配器。
  - 缓存分配器基类被 CUDA 分配器继承（cuda-backend）。
- 方向：算子内核 → Allocator::allocate → DataPtr → StorageImpl → TensorImpl。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，不适用 Go/TS 专项。本叶子适配口径：
- **C++ RAII 所有权**：DataPtr 的 deleter 把"释放策略"编码进唯一指针，StorageImpl 析构即安全归还，无需 Python 侧手动管理。
- **可插拔分配器**：`SetAllocator(DeviceType)` 策略模式，用户/库可替换 CPU 分配器（如 jemalloc、实验 CPUCachingAllocator）而不改动算子。
- **后端差异化**：默认 CPU 分配器无缓存（直接 malloc），而 CUDA 用 CachingDeviceAllocator 基类做块缓存——同一 `Allocator` 接口下后端策略不同。
- **无 GIL**：纯 C++ 内存层。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 分配器与 StorageImpl 架构图 | `memory-allocator-architecture.html` | architecture | showcase |
| 张量分配-释放数据流图 | `memory-allocator-dataflow.html` | dataflow | standard（降档披露：节点中文 label 宽度在 showcase 严格校验下超宽，按规则降 standard 渲染成功） |

- JSON IR 源文件：`json/` 下对应两个文件。
- 说明：分配器层级用 architecture；"allocate→DataPtr→StorageImpl→析构释放"用 dataflow。
