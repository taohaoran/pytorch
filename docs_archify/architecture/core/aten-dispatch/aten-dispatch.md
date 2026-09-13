# 算子分发与内核注册（aten-dispatch）

> 本文是 `core` 域下的叶子子系统文档。域级总览见 `../core.md`。
> 本文只展开 **c10 Dispatcher 如何把一次算子调用按 DispatchKey 路由到具体内核**，以及内核注册机制；不重复展开：
> - DispatchKey / DispatchKeySet 这些标签类型本身的定义 → 见 `../c10-core/c10-core.md`
> - 选中内核后具体的数值实现（add/mul/reduce 等）→ 见 `../aten-native/aten-native.md`
>
> 源码基准：PyTorch（C++ 核心），commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`，目录 `aten/src/ATen/core/dispatch/`、`aten/src/ATen/core/op_registration/`、`aten/src/ATen/native/DispatchStub.h`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| Dispatcher 门面 | 全局唯一的分发器单例；所有算子调用的统一入口（boxed/unboxed） | `aten/src/ATen/core/dispatch/Dispatcher.h`（`class TORCH_API Dispatcher final` 第 71 行）、`Dispatcher.cpp` |
| 未装箱分发 call() | 模板化 C++ 调用路径：从实参提取 DispatchKeySet → 查内核表 → 直接调用函数指针 | `Dispatcher.h:784`（`Dispatcher::call`） |
| 装箱分发 callBoxed() | TorchScript 栈式调用路径：从 `jit::Stack*` 提取键集 → 查表 → 调用 boxed kernel | `Dispatcher.h:861`（`Dispatcher::callBoxed`） |
| 二次分发 redispatch() | 内核内部跳过当前层、用裁剪后的键集再查一次表（如 Autograd 内核跑完后落到后端） | `Dispatcher.h:843`（`Dispatcher::redispatch`） |
| OperatorEntry 调度表 | 每个算子一条记录：固定大小 `dispatchTable_[num_runtime_entries]` 内核数组 + 元数据 | `dispatch/OperatorEntry.h`（`class TORCH_API OperatorEntry` 第 70 行，`dispatchTable_` 第 237 行） |
| DispatchKeyExtractor | 从实参（unboxed 模板展开或 boxed 栈）推导出本次调用的 DispatchKeySet，并合并 TLS 局部键集 | `dispatch/DispatchKeyExtractor.h`（`getDispatchKeySetUnboxed` 第 194 行、`getDispatchKeySetBoxed` 第 163 行） |
| KernelFunction | 内核函数指针封装：同时持有 unboxed（直接 C++ 调用）与 boxed（经栈）两种入口 | `boxing/KernelFunction.h`、`boxing/BoxedKernel.h` |
| 算子注册 registerImpl | 把 (operator, dispatch_key, kernel) 三元组写入调度表；别名键在此阶段展开为运行时键 | `Dispatcher.h:260`（`registerImpl`） |
| 算子注册 API（用户侧） | `TORCH_LIBRARY`/`Library` 宏与 `def()/op()/kernel()` 链式注册 | `op_registration/op_registration.h`、`library.cpp` |
| DispatchStub（低层函数指针切换） | 独立于完整 Dispatcher 的设备级函数指针表：按 DeviceType 选 CPU/AVX/AVX512/CUDA 实现 | `aten/src/ATen/native/DispatchStub.h`（`struct DispatchStub` 第 222 行） |
| 算子查找 findSchemaOrThrow | 按名字查算子 schema，返回 `OperatorHandle` | `Dispatcher.h:160` |
| 后端 fallback 内核 | 为整个后端注册一个兜底内核（如 MetaFallback、BackendSelectFallback） | `MetaFallbackKernel.cpp`、`BackendSelectFallbackKernel.cpp`、`VariableFallbackKernel.cpp` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `c10::Dispatcher` | `Dispatcher.h:71` | 分发器单例。持有全局算子注册表；提供 `call`/`callBoxed`/`redispatch`/`registerImpl`/`findSchemaOrThrow` |
| `c10::OperatorHandle` / `TypedOperatorHandle<Return(Args...)>` | `Dispatcher.h` | 算子句柄；`operatorDef_->op` 指向 `OperatorEntry`；typed 版本带签名以支持 unboxed 直调 |
| `c10::OperatorEntry` | `OperatorEntry.h:70` | 单算子调度记录。`lookup(DispatchKeySet ks)`（第 182 行）：取 `ks.highestPriorityTypeId()` 得索引，从 `dispatchTable_[idx]` 取内核 |
| `c10::DispatchKeyExtractor` | `DispatchKeyExtractor.h` | 编译期按算子 schema 计算"哪些参数携带 DispatchKey"，运行期从实参抽取设备/布局键，并与 TLS 局部键集合并 |
| `c10::KernelFunction` | `boxing/KernelFunction.h` | 类型擦除的内核指针；`call<Return,Args...>()` 按 unboxed 快路径调用，否则经 boxed boxing |
| `c10::RegistrationHandleRAII` | `dispatch/RegistrationHandleRAII.h` | RAII 句柄，析构即注销（动态注册/反注册） |
| `at::DispatchStub` | `native/DispatchStub.h:222` | 模板 `DispatchStub<rT(*)(Args...), T>`；`operator()(device_type, args...)` 先 `get_call_ptr(device_type)` 再调；`set_cuda_dispatch_ptr` 等按后端注入 |
| `TORCH_LIBRARY` 宏 / `Library` | `op_registration/op_registration.h`、`library.cpp` | C++ 用户/库扩展算子的声明式注册门面 |

## 3. 关键调用链

**调用链一：一次 unboxed 算子调用（eager 主热路径）**

1. 生成的 `at::add(Tensor, Tensor, Scalar)` 绑定到某个 `TypedOperatorHandle`。
2. 进入 `Dispatcher::call(op, args...)`（`Dispatcher.h:784`）。
3. `op.operatorDef_->op.dispatchKeyExtractor().getDispatchKeySetUnboxed<Args...>(args...)`（第 787-789 行）扫描各 Tensor 实参的 `TensorImpl::key_set()`，与线程局部键集（`LocalDispatchKeySet`）合并，得到本次调用的 `DispatchKeySet`。
4. `op.operatorDef_->op.lookup(dispatchKeySet)`（第 797 行）→ `OperatorEntry::lookup`（`OperatorEntry.h:182`）：对键集取 `highestPriorityTypeId()`（即优先级最高的运行时键），用它索引 `dispatchTable_[idx]` 得到 `KernelFunction`。
5. 未命中观测/慢路径时，直接 `kernel.call<Return,Args...>(op, dispatchKeySet, args...)`（第 836 行）执行——这是无虚函数、无栈装箱的快路径。

**调用链二：Autograd 层 redispatch**

1. 步骤 4 查到的内核可能是 `AutogradCPU` 等"包装层"内核（见 dispatch key 体系）。
2. Autograd 内核构造反向图节点后，调用 `Dispatcher::redispatch(op, currentDispatchKeySet - Autograd位, args...)`（`Dispatcher.h:843`）。
3. redispatch 用裁剪后的键集再次 `lookup`，跳过 Autograd 层，落到 `CPU`/`CUDA` 等数值内核。

**调用链三：TorchScript 装箱调用**

1. TorchScript 解释器把参数压入 `jit::Stack`。
2. `Dispatcher::callBoxed(op, &stack)`（`Dispatcher.h:861`）：`getDispatchKeySetBoxed(stack)`（第 867 行）从栈上各值类型读键集 → `entry.lookup`（第 875 行）取 boxed 内核 → 在栈上执行。

**调用链四：注册期（静态/动态）**

1. 静态：构建期 `RegisterSchema.cpp`/`Register*.cpp` 调用 `Dispatcher::registerImpl`（`Dispatcher.h:260`），把 `(op_name, dispatch_key, kernel_func)` 写入 `OperatorEntry`。
2. 别名键（如 `CompositeImplicitAutograd`）在此时被展开映射到各后端运行时键，填充 `dispatchTable_`。
3. 动态：`TORCH_LIBRARY` 宏在模块加载期通过同样路径注册；`RegistrationHandleRAII` 析构反注册。

## 4. 配置项

| flag / option | 默认 / 行为 | 位置 |
|---------------|-------------|------|
| `num_runtime_entries` | 运行时调度表槽位数 = 功能键数 + 每后端功能键×后端数；移动端裁剪为 8 | `c10/core/DispatchKey.h:570` |
| `HAS_TORCH_SHOW_DISPATCH_TRACE` / NDEBUG | 开启后打印 `[call]/[redispatch]` 分发轨迹 | `Dispatcher.h:790-796` |
| `PYTORCH_DISABLE_PER_OP_PROFILING` | 定义后跳过 per-op RecordFunction 慢路径检查 | `Dispatcher.h:798` |
| DispatchStub 后端指针 | 静态零初始化，首次按 DeviceType 懒选；AVX2/AVX512 由 `HAVE_AVX*_DEFINITION` 编译期开启 | `native/DispatchStub.h:285-295` |
| `torch.__future__` / `TORCH_DISABLE_*` | 各类调试/性能开关经环境变量或编译宏控制 | 编译期 CMake 选项 |

## 5. 错误与重试语义

- 本叶子为查表 + 函数指针调用，**无重试/退避**；查不到内核时抛 C++ 异常。
- `lookup` 未命中：`dispatchTable_[idx]` 为空 kernel 时由 Dispatcher 抛出明确错误（"Operator xxx does not have kernel for dispatch key yyy"），并附带已注册键集信息辅助定位。
- `findSchemaOrThrow`（`Dispatcher.h:160`）找不到算子时抛 schema 查找异常。
- 注册冲突/重复注册由 `registerImpl` 阶段校验并抛错；`RegistrationHandleRAII` 保证异常安全。
- 无网络/IO 失败，无后台异步失败路径。

## 6. 并发细节

- **调度表写多读少**：注册发生在初始化/模块加载期；运行期 `call`/`lookup` 只读 `dispatchTable_`，无锁。注释（`Dispatcher.h:863`）明确写列表迭代器在删除时仍有效，因此 `callBoxed` 不需加锁。
- **全局 Dispatcher 单例**：注册用的互斥在 `Dispatcher.cpp` 内保护注册表；运行期读路径不加锁（写后发布，happens-before 经初始化同步）。
- **TLS 状态**：`DispatchKeyExtractor` 合并线程局部 `LocalDispatchKeySet`（`c10/core/impl/LocalDispatchKeySet.h`），配合 `InferenceMode`/`TorchDispatchMode` 守卫做 RAII 键集增删；各线程独立。
- **无工作线程池**：本叶子纯同步调用，不启线程；autograd-engine 才建线程池。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `aten/src/ATen/core/dispatch/`（Dispatcher/OperatorEntry/DispatchKeyExtractor/KernelFunction）
- `aten/src/ATen/core/op_registration/`（TORCH_LIBRARY 注册 API）
- `aten/src/ATen/native/DispatchStub.h`（低层设备级函数指针切换）

**Out-of-Scope（不在本仓库源码内）**
- DispatchKey/DispatchKeySet/Device 等类型定义本身 → `../c10-core/c10-core.md`
- 各内核数值实现与 TensorIterator → `../aten-native/aten-native.md`
- CUDA/cuBLAS 等后端库 → 外部组件，不在本仓库源码内
- 反向计算图的执行引擎 → `../autograd-engine/autograd-engine.md`

## 8. 与相邻子系统交互

- 上游 → 本叶子：
  - Python 前端（`torch/_C` 绑定）→ `at::Tensor` 方法 → `Dispatcher::call/callBoxed`。
  - TorchScript/JIT 解释器 → `callBoxed`。
- 本叶子 → 下游：
  - `lookup` 选中的 `KernelFunction` → 落到 `aten-native` 通用算子（CPU）或各后端目录（CUDA/MPS/...）。
  - Autograd 包装层内核 → 构造反向节点并 redispatch（autograd-engine）。
  - `DispatchStub` 被 `aten-native` 内部分子（如 Blas/Reduce）用于 CPU 指令集级切换。
- 依赖方向：`c10`（DispatchKey 定义）← 本叶子（Dispatcher）← `aten/native`（内核）← `torch/csrc`（绑定）。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，**不适用** Go/TS 专项。本叶子适配口径：
- **C++ 模板消除虚调用**：unboxed `Dispatcher::call` 用模板 `Args...` 在编译期展开参数抽取与函数指针调用，避开 boxing 栈与虚分发，是 eager 热路径性能关键（`Dispatcher.h:782` Note: Argument forwarding）。
- **类型擦除边界**：`KernelFunction` 把 typed kernel 擦除为统一指针；boxed 路径经 `jit::Stack` 做运行期 boxing/unboxing（`boxing/`）。
- **两层分发并存**：完整 Dispatcher（算子×后端×功能键）与 `DispatchStub`（同一后端内 CPU 指令集/CUDA 切换）是两套互补机制，前者面向算子注册，后者面向 native 内部性能特化。
- **无 GIL 交互**：Dispatcher 为纯 C++；Python 桥接在 `torch/csrc`。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 分发注册架构图 | `aten-dispatch-architecture.html` | architecture | showcase |
| 一次算子调用分发时序图 | `aten-dispatch-sequence.html` | sequence | showcase |
| 算子调用数据流图 | `aten-dispatch-dataflow.html` | dataflow | standard（降档披露：单行流水线中文 label 在 showcase 严格布局校验下触发宽度/高度约束，按规则降 standard 渲染成功） |

- JSON IR 源文件：`json/` 下对应三个文件。
- 说明：call/redispatch 的"实参→键集→查表→内核"是请求生命周期，适合 sequence；注册期"三元组→调度表"用 dataflow。
