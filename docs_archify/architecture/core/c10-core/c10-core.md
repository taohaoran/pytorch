# c10 核心抽象（c10-core）

> 本文是 `core` 域下的叶子子系统文档。域级总览见 `../core.md`。
> 本文只展开 **Tensor 与 Storage 的底层核心抽象类型定义**（TensorImpl / DispatchKey / Device / Storage / Scalar / TensorOptions / Allocator 接口），不重复展开：
> - 内核如何依据 DispatchKey 选中并执行 → 见 `../aten-dispatch/aten-dispatch.md`
> - CPU 缓存分配器的具体实现与 StorageImpl 生命周期管理 → 见 `../memory-allocator/memory-allocator.md`
>
> 源码基准：PyTorch（C++ 核心），commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`，目录 `c10/core/`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| TensorImpl 低层表示 | 张量的底层表示：持有 Storage 指针 + view 元数据（sizes/strides/offset/dtype），侵入式引用计数 | `c10/core/TensorImpl.h`（`struct C10_API TensorImpl`，第 510 行）、`c10/core/TensorImpl.cpp` |
| DispatchKey / BackendComponent 枚举 | 算子分发的位级标签体系：后端组件位 + 功能键位，组合出运行时 DispatchKey | `c10/core/DispatchKey.h`（`enum class DispatchKey` 第 136 行、`enum class BackendComponent` 第 65 行）、`c10/core/DispatchKey.cpp` |
| DispatchKeySet | 64 位 bitset，一次操作上所有"可能生效"的分发键集合；取最高位即优先级最高键 | `c10/core/DispatchKeySet.h/.cpp` |
| Device / DeviceType | 计算设备标识：类型（CPU/CUDA/...）+ 设备序号（int8） | `c10/core/Device.h`（`struct C10_API Device` 第 31 行）、`c10/core/DeviceType.h/.cpp` |
| Storage / StorageImpl | 数据缓冲区持有器：`DataPtr`（唯一指针 + deleter + device）+ 字节大小 + 可调整标志；侵入式引用计数 | `c10/core/StorageImpl.h`（`struct C10_API StorageImpl` 第 55 行）、`c10/core/Storage.h`（`Storage` 薄包装第 32 行） |
| ScalarType / TypeMeta | 元素数据类型枚举（float/int/half/bf16/quant 等）与类型元信息（itemsize/对齐） | `c10/core/ScalarType.h`、`c10/core/ScalarType.cpp`、`c10/core/ScalarTypeToTypeMeta.h` |
| Scalar | 0 维单元素"张量"的 C++ 变体类型，支持整型/浮点/复数及 SymInt 符号化取值 | `c10/core/Scalar.h`（`class C10_API Scalar` 第 47 行）、`c10/core/Scalar.cpp` |
| TensorOptions | 构造张量的属性聚合（dtype/device/layout/memory_format/requires_grad），可推导 DispatchKey | `c10/core/TensorOptions.h`（`computeDispatchKey` 第 131 行） |
| Allocator 抽象接口 | 纯虚内存分配接口 `allocate(n)->DataPtr` + `copy_data`；按 DeviceType 注册 | `c10/core/Allocator.h`（`struct C10_API Allocator` 第 180 行）、`c10/core/Allocator.cpp` |
| DataPtr | 带 deleter 与 device 的唯一指针，是 Allocator 与 Storage 之间的内存契约 | `c10/core/Allocator.h`（`class C10_API DataPtr` 第 40 行） |
| SymInt / SymBool / SymFloat | 符号化形状/尺寸的惰性求值表示，用于动态形状（torch 编译） | `c10/core/SymInt.h/.cpp`、`c10/core/SymBool.h`、`c10/core/SymFloat.h` |
| GeneratorImpl / RNG 种子基类 | 随机数生成器的抽象基类 | `c10/core/GeneratorImpl.h/.cpp` |
| Layout / MemoryFormat / QScheme | 布局（dense/sparse）、内存格式（channels_last 等）、量化方案枚举 | `c10/core/Layout.h`、`c10/core/MemoryFormat.h`、`c10/core/QScheme.h` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `c10::TensorImpl` | `TensorImpl.h:510` | 张量底层实现。成员：`storage_`（Storage，第 2888 行）、`sizes_and_strides_`（SizesAndStrides，第 2923 行）、`key_set_`（DispatchKeySet，第 3045 行）、`data_type_`（TypeMeta，第 2934 行）、`storage_offset_`、`version_counter_`（VariableVersion，第 2919 行）、`extra_meta_`（惰性扩展元数据，第 2917 行） |
| `c10::StorageImpl` | `StorageImpl.h:55` | 数据缓冲区唯一持有者。成员：`data_ptr_`（DataPtr）、`size_bytes_`（SymInt）、`resizable_`、`allocator_`、`materialize_fn_`（COW/惰性物化钩子） |
| `c10::Storage` | `Storage.h:32` | `intrusive_ptr<StorageImpl>` 的薄值包装，供 TensorImpl 共享同一缓冲区 |
| `c10::Dispatcher` 外的键选择 | — | 本叶子只定义键；`TensorImpl::key_set()` 返回 `key_set_` 供 Dispatcher 取最高位（见 aten-dispatch） |
| `c10::Device` | `Device.h:31` | `DeviceType type_` + `DeviceIndex index_`（int8，-1 表示"当前设备"）；`supports_as_strided()` 等设备能力查询 |
| `c10::DispatchKey` | `DispatchKey.h:136` | 分发键枚举。分五类：(1) 不可定制后端 (2) 不可定制功能键 (3) 可按后端定制功能键（Dense/Sparse/Quantized/NestedTensor/AutogradFunctionality）(4) 按后端实例化运行时键（CPU/SparseCUDA/...）(5) 别名键（Autograd/CompositeImplicitAutograd/...） |
| `c10::BackendComponent` | `DispatchKey.h:65` | 后端位：CPU/CUDA/HIP/XLA/MPS/.../PrivateUse1-3/Meta，共 ≤16 个（`static_assert` 第 552 行） |
| `c10::Scalar` | `Scalar.h:47` | 单元素变体。内部 `union v` + `tag`；支持从所有数值类型隐式构造，也可持有 SymInt/SymFloat/SymBool |
| `c10::TensorOptions` | `TensorOptions.h:138` | 聚合 dtype/device/layout/memory_format；`computeDispatchKey()` 把属性映射为 DispatchKey 候选 |
| `c10::Allocator` | `Allocator.h:180` | 纯虚接口：`allocate(size_t)=0`、`copy_data(dest,src,count)=0`；可选 `raw_deleter()`（Thrust 友好）、`clone()` |
| `c10::DataPtr` | `Allocator.h:40` | `UniqueVoidPtr ptr_` + `Device device_`；`compare_exchange_deleter()` 支持安全替换 deleter |
| `c10::SetAllocator/GetAllocator` | `Allocator.h:289-290` | 按 DeviceType 注册/取回分配器（初始化期调用，非线程安全） |
| `c10::make_storage_impl` | `StorageImpl.h:427` | 工厂：按 device 选择 StorageImplCreateHelper（各后端可注册自己的 StorageImpl 子类） |

## 3. 关键调用链

**调用链一：构造一个 CPU 浮点张量（属性 → DispatchKey → 底层对象）**

1. 用户调用 `at::empty({2,2}, at::dtype(at::kFloat).device(at::kCPU))`，`TensorOptions` 聚合 dtype/device。
2. 分发期由 `TensorOptions::computeDispatchKey()`（`TensorOptions.h:131`）把 `kFloat`+`CPU` 映射为运行时键 `DispatchKey::CPU`，并与 Dense 功能位组合成 `DispatchKeySet`。
3. 经 Dispatcher（见 aten-dispatch）选中 `CPU` 后端 kernel，在 kernel 内：
   - `GetAllocator(DeviceType::CPU)`（`Allocator.h:290`）取回 CPU 分配器；
   - 调用 `allocator->allocate(nbytes)` 得到 `DataPtr`；
   - `make_storage_impl(...)`（`StorageImpl.h:427`）构造 `StorageImpl`（持有 data_ptr_/size_bytes_/allocator_）；
   - 包成 `Storage` 后构造 `TensorImpl(storage, key_set, TypeMeta::Make<float>())`（`TensorImpl.h:524`），写入 `sizes_and_strides_`（{2,2} 行主序 stride）、`storage_offset_=0`、`key_set_`。
4. 返回的 `Tensor` 句柄即 `intrusive_ptr<TensorImpl>`。

**调用链二：访问张量数据（不可变 vs 可变路径与惰性物化）**

1. 读 `.data_ptr()` → `StorageImpl::data()`（`StorageImpl.h:203`），先检查 `throw_on_immutable_data_ptr_`，否则返回 `data_ptr_.get()`。
2. 写 `.mutable_data()` → `StorageImpl::mutable_data()`（`StorageImpl.h:210`），命中 `has_mutable_data_ptr_check_` 快路径开关后：若设了 `materialize_fn_`（COW/惰性物化钩子，`StorageImpl.h:384`）则先执行物化，再返回可变指针。
3. 所有特殊检查被折叠到单个布尔 `has_mutable_data_ptr_check_`（`StorageImpl.h:401`），因为 `.data/.data_ptr` 常在热路径上——这是性能设计。

**调用链三：DispatchKeySet 优先级取键**

1. 张量 `TensorImpl::key_set()`（`TensorImpl.h:590`）返回 `key_set_`。
2. Dispatcher 对该 64 位 bitset 做 `count leading zeros` 取最高优先级位（注释 `DispatchKey.h:110`）；BackendComponent 低位决定具体后端，功能高位决定功能层（Autograd/Functionalize/...）。
3. 别名键（如 `Autograd`）在调度表计算期被展开为各后端键（`DispatchKey.h:431` Note）。

## 4. 配置项

| flag / option | 默认 / 行为 | 位置 |
|---------------|-------------|------|
| `C10_MOBILE_TRIM_DISPATCH_KEYS` | 移动端裁剪：`num_runtime_entries` 从完整表缩为 8 项 | `DispatchKey.h:566-572` |
| 默认 dtype | `get_default_dtype()`（默认 `kFloat`）；`TensorOptions` 未指定 dtype 时回退 | `TensorOptions.h:34`、`c10/core/DefaultDtype.h` |
| CPU 设备序号约束 | CPU index 必须为 -1 或 0（debug 断言） | `Device.h:181-184` |
| 后端数量上限 | BackendComponent ≤ 16（编译期 `static_assert`） | `DispatchKey.h:552` |
| `allocator priority` | `SetAllocator(t, alloc, priority=0)`，仅 ≥priority 可覆盖默认分配器 | `Allocator.h:289` |
| PrivateUse1/2/3 键 | 预留供外部加速器（不改 PyTorch 即可挂新后端） | `DispatchKey.h:49-51,480` Note |

## 5. 错误与重试语义

- 本叶子是纯数据类型/接口定义层，**不涉及重试或退避**；失败以 C++ 异常/断言形式立即抛出。
- `Device::validate()`（`Device.h:172`）用 `TORCH_INTERNAL_ASSERT_DEBUG_ONLY` 校验 index 范围，release 构建关闭以提升微基准性能（注释第 173 行）。
- `StorageImpl::data_ptr()` 访问不可变指针时若 `throw_on_immutable_data_ptr_` 置位则 `throw_data_ptr_access_error()`（`StorageImpl.h:145,307`）；`mutable_data_ptr()` 在 `set_throw_on_mutable_data_ptr()` 时抛 `throwNullDataPtrError()`（`StorageImpl.h:324`）。
- `make_storage_impl` / 构造 resizable storage 时要求 allocator 非空，否则 `TORCH_INTERNAL_ASSERT`（`StorageImpl.h:71-74`）。
- 字符串解析 `parseDispatchKey(string)`（`DispatchKey.h:589`）失败抛异常。

## 6. 并发细节

- **引用计数线程安全**：`TensorImpl` 与 `StorageImpl` 均继承 `c10::intrusive_ptr_target`，`intrusive_ptr` 的 refcount 为原子操作（`c10/util/intrusive_ptr.h`），可跨线程安全持有/复制句柄；但对象内部字段（如 resize 后的 sizes）本身不自动加锁，由上层调度串行化保证。
- **Allocator 注册非线程安全**：`SetAllocator/GetAllocator` 注释明确"非线程安全，仅初始化期调用"（`Allocator.h:282`）。
- **无自建线程池**：本叶子不启动 goroutine/线程；并发原语（mutex/worker 池）在 autograd-engine 与缓存分配器层。
- **TLS 状态**：默认 dtype、本地 DispatchKeySet（`c10/core/impl/LocalDispatchKeySet.h`）等为线程局部状态，配合 `InferenceMode`/`GradMode` 守卫（`c10/core/InferenceMode.h`、`GradMode.h`）做 RAII 切换。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `c10/core/` 下全部核心值类型与抽象接口：TensorImpl/StorageImpl/DispatchKey(Set)/Device/Scalar/TensorOptions/Allocator/DataPtr/SymInt 等。

**Out-of-Scope（不在本仓库源码内）**
- 各后端具体 kernel 数值实现 → `aten/src/ATen/native/`（见 aten-native 叶子）
- Dispatcher 如何用 key_set_ 选内核 → `aten/src/ATen/core/dispatch/Dispatcher.*`（见 aten-dispatch 叶子）
- CPU/CUDA 缓存分配器具体算法 → `memory-allocator` / cuda-backend 叶子
- autograd 元数据（`AutogradMeta`/`grad_fn`）如何挂到 TensorImpl 上 → `torch/csrc/autograd/`（见 autograd-engine 叶子）
- CUDA runtime / cuBLAS / cuDNN 等第三方库 → 外部组件，不在本仓库源码内

## 8. 与相邻子系统交互

- 上游 → 本叶子：`aten-dispatch` 用 `TensorImpl::key_set()` 与 `TensorOptions::computeDispatchKey()` 得到候选键集合，再查内核表。
- 本叶子 → 下游：
  - `TensorImpl.storage_` → `StorageImpl` → `Allocator::allocate()` → 各后端分配器（memory-allocator 叶子实现 CPU 缓存分配器）。
  - `TensorImpl` 持有 `version_counter_` 供 autograd 版本追踪（inplace/view 检测）；`extra_meta_` 可挂 `AutogradMeta`（autograd-engine 叶子）。
  - `TensorImpl.data_type_`（TypeMeta）→ 决定 kernel 内元素读写宽度。
- 方向：Python/TorchScript 前端 → `at::Tensor`（包装 TensorImpl）→ Dispatcher → 本叶子定义的抽象。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，**不适用** Go 专项（goroutine/Reconciler/informer）与 TS 专项（capability seam/workspace）。本叶子的适配口径为：
- **C++ 对象模型**：侵入式引用计数（`intrusive_ptr_target`）替代共享指针，便于跨 Python/C 边界裸指针传递；PImpl 不使用，关键热路径字段（sizes/strides/key_set）直接内联在 TensorImpl 布局中（见 `TensorImpl.h:3239` 起的 `are_equal<sizeof(...)>` 静态布局断言）。
- **依赖方向**：`c10` 是最底层库，**不依赖** aten/torch/csrc；被 `aten`（ATen）与 `torch/csrc`（Python 绑定）双向依赖。本叶子定义纯抽象，不反向依赖上层。
- **符号化形状**：`SymInt`/`SymBool`/`SymFloat` 是 torch 编译（dynamo/inductor）与 eager 共享的形状抽象，本叶子只定义类型，求值在 compile-graph 域。
- **无 GIL 交互**：本叶子为纯 C++ 值类型，不直接持有 Python 对象；与 Python 桥接经 `PyObjectSlot`/`SafePyObject`（`c10/core/impl/`）在更上层完成。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| c10 核心对象关系架构图 | `c10-core-architecture.html` | architecture | showcase |
| 张量构造数据流图 | `c10-core-dataflow.html` | dataflow | standard（降档披露：多行节点的中文 label/sublabel 宽度与垂直流标签位置在 showcase 严格布局校验下多次迭代仍触发重叠/超宽，按规则降 standard 渲染成功） |

- JSON IR 源文件：`json/c10-core-architecture.json`、`json/c10-core-dataflow.json`
- 说明：本叶子为类型定义层，无运行时"消息时序"，故不补 sequence 图；用 dataflow 表达"属性 → 键 → 分配 → 构造"的数据流。
