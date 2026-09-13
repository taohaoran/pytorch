# torch-api（torch-api）

> 本文是 `python-frontend` 域下的叶子子系统文档。域级总览见 `../python-frontend.md`。
> 本文聚焦 **Python 包入口、Tensor 类本身、算子命名空间与 C++ 绑定面**；内置算子的内核实现见相邻叶子，
> 用户自定义算子注册机制见 `extensions/custom-ops`，设备/流上下文见 `device-context`。
>
> 源码基准：PyTorch `main`，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| 包初始化与符号再导出 | `torch/__init__.py` 是用户 `import torch` 的入口，聚合 C 扩展符号、子模块、工具函数 | `torch/__init__.py`（约 3570 行） |
| 单次进程初始化守护 | 通过 `sys._torch_import_started` 阻止 reload/重复 import，避免 C++ 全局状态被污染 | `torch/__init__.py:44-52` |
| C 扩展符号注入 | `from torch._C import *` 把编译期 pybind 注册的符号（Tensor、dtype、设备、API）注入顶层命名空间 | `torch/__init__.py:488,505` |
| Tensor Python 类 | `_tensor.py` 定义 `Tensor`，继承 C++ 侧 `torch._C.TensorBase`，补充 dunder、序列化、deepcopy、转换方法 | `torch/_tensor.py:102`（约 1748 行） |
| 算子命名空间 `torch.ops` | `_ops.py` 提供 `_OpNamespace`/`OpOverload`/`OpOverloadPacket`，按 `torch.ops.ns.op.overload` 惰性解析算子 | `torch/_ops.py:837,1237,1417` |
| 高阶算子 | `HigherOrderOperator` 表示 autograd 变换类高阶算子（如 `vmap`、`associate`），不走普通分发 | `torch/_ops.py:282` |
| C++ pybind 绑定面 | `csrc/Module.cpp` 在 `py_module` 上 `def(...)` 注册崩溃处理、环境变量、错误 demangle、API 用量统计等 | `torch/csrc/Module.cpp:2734+` |
| 调度模式栈（pre-dispatch） | `_ops.py` 维护 dispatch mode 的 pre-dispatch 栈状态，供 functorch/dynamo 挂钩 | `torch/_ops.py:621-795` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `class Tensor(torch._C.TensorBase)` | `torch/_tensor.py:102` | 用户可见张量对象；真正的数据/尺寸/ strides 存于 C++，Python 仅封装行为 |
| `Tensor.__deepcopy__` / `__reduce__` | `torch/_tensor.py:121+` | 控制深拷贝与 pickle 重建，区分 leaf/non-leaf、sparse/量化/meta 等设备类型 |
| `class OperatorBase` | `torch/_ops.py:62` | 所有可调用算子对象的基类 |
| `class OpOverload` | `torch/_ops.py:837` | 单个特定 overload 的可包装对象，`__call__` 触发 C++ 分发 |
| `class OpOverloadPacket` | `torch/_ops.py:1237` | 同名算子的全部 overload 集合，按参数解析默认 overload |
| `class _OpNamespace(types.ModuleType)` | `torch/_ops.py:1417` | `torch.ops.<ns>` 动态命名空间，按属性访问惰性建包 |
| `class HigherOrderOperator` | `torch/_ops.py:282` | 高阶算子基类，不按 DispatchKey 走常规内核分发 |
| `dl_open_guard()` | `torch/_ops.py:46` | 加载共享库时的 RTLD_GLOBAL 上下文守卫 |

## 3. 关键调用链

1. **`import torch`**：`torch/__init__.py` 先置 `sys._torch_import_started=True` → 导入 `torch._utils`/`_utils_internal` → `from torch._C import *`（`torch/__init__.py:488`）把 pybind 注册符号注入 → 继续导入子模块（nn、optim、cuda 等）。重复 import 直接 `raise ImportError`。
2. **调用内置算子**：用户 `torch.mm(a,b)` → 该符号来自 `torch._C` 或 `_tensor.py` 封装 → 进入 C++ PyTorch Dispatcher → 按 DispatchKey（CPU/CUDA/Autograd…）选中内核执行。
3. **`torch.ops.aten.mm(...)`**：属性访问逐层落到 `_OpNamespace` → `_get_packet`（`torch/_ops.py:1485`）取 `OpOverloadPacket` → `OpOverload.__call__`（`_ops.py:837`）→ `_python_dispatcher` 或直接 C++ 分发。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `torch.set_default_dtype(d)` | 设置后续无 dtype 指定张量的默认浮点类型 | `torch/__init__.py:1762` → `_C._set_default_dtype` |
| `torch.use_deterministic_algorithms(mode)` | 控制是否强制确定性算法，`warn_only` 仅告警 | `torch/__init__.py:1924` |
| `torch.set_float32_matmul_precision` | 控制 FP32 矩阵乘精度模式（highest/high/medium） | `torch/__init__.py:2005` |
| `USE_RTLD_GLOBAL_WITH_LIBTORCH` | 决定加载 libtorch 是否用 RTLD_GLOBAL | `torch/_utils_internal.py` |

## 5. 错误与重试语义

- `import torch` 失败后重试/reload 会被 `sys._torch_import_started` 直接拒绝（`torch/__init__.py:44-52`），因为 C++ 全局状态不可二次初始化——**不提供重试**。
- `Tensor.__deepcopy__` 对非 leaf 张量直接 `raise RuntimeError`（`torch/_tensor.py:127`），引导用户使用 `clone()`。
- 算子分发失败由 C++ 侧抛出 pybind 异常，Python 侧不做重试；`handle_torch_function` 将未识别张量类型的 TypeError 转为 NotImplemented，供 Tensor 子类协议接管。

## 6. 并发细节

- Python 侧受 GIL 约束，`Tensor` 对象本身无线程亲和；真正并发由 C++ Dispatcher 在持有/释放 GIL 时驱动设备内核。
- `torch/__init__.py` 顶部 `import threading`，配合多进程环境准备 `prepare_multiprocessing_environment`（`torch/_utils_internal.py`），为 DataLoader worker fork/spawn 做前置。
- `torch._C` 扩展模块的 C++ 全局状态是单进程单例，故禁止重复 import。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/__init__.py` 包入口、`torch/_tensor.py` 的 Tensor Python 类、`torch/_ops.py` 算子命名空间、`torch/csrc/Module.cpp` 的顶层 pybind 注册。

**Out-of-Scope（不在本仓库源码内 / 相邻叶子）**
- 各设备算子内核（CUDA/CPU kernel）与 CUDACachingAllocator → cuda-backend 分片，不在本叶子。
- 用户自定义算子注册与 `cpp_extension` JIT 构建 → `extensions/custom-ops`。
- 函数变换 vmap/grad 与 AOTAutograd → `extensions/functorch`。
- dynamo/inductor 字节码追踪与代码生成 → dynamo-inductor 分片。
- 真正的 C++ Tensor 内存/分发实现位于 aten/，本叶子只描述 Python↔C++ 绑定面。

## 8. 与相邻子系统交互

- 用户脚本 → `torch/__init__.py` →（再导出）→ `torch/_C` 与各子模块（nn/optim/data/utils）。
- `Tensor`（`_tensor.py`）→ 继承 `torch._C.TensorBase`（C++）→ Dispatcher → 设备内核（下游）。
- `_ops.py` 命名空间 ↔ `torch.library`（`extensions/custom-ops`）：用户注册的自定义算子通过同一 OpOverloadPacket 机制暴露为 `torch.ops.*`。
- functorch/dynamo 通过 `_ops.py` 的 dispatch mode 栈（pre-dispatch）插入拦截点。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，不适用 Go（goroutine/informer）或 TS（workspace）专项。Python 侧按**能力缝（capability seam）**归组：
- 服务定义侧：`torch/_C` pybind 暴露的 C++ 接口签名；
- 提供方：`_tensor.py` 的 Tensor 方法、`_ops.py` 的 OpOverload 实现；
- 消费方：用户脚本与上层子模块（nn/optim/data）。
C++ 绑定层（`torch/csrc/Module.cpp`）是 Python↔C++ 的唯一边界，依赖方向单向：Python → pybind → libtorch，禁止反向。GIL 与 C++ 多线程交互在 device-context 与 data-loading 叶子中展开。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 包入口与绑定面结构 | `torch-api-architecture.html` | architecture | showcase |

JSON IR 源文件：`json/torch-api-architecture.json`。本叶子以静态结构为主，调用链已在第 3 节文字化说明，不另出时序图。

![torch-api 架构图](torch-api-architecture.html)
