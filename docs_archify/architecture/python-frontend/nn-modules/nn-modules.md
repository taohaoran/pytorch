# nn-modules（nn-modules）

> 本文是 `python-frontend` 域下的叶子子系统文档。域级总览见 `../python-frontend.md`。
> 本文聚焦 **Module 基类、层级容器、Parameter 与 hooks 机制、functional 算子**；
> 优化器见 `optim` 叶子，张量与算子命名空间见 `torch-api`。
>
> 源码基准：PyTorch `main`，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| Module 基类 | 所有神经网络模块的根，管理子模块/参数/buffer 注册、forward 调用、hooks、state_dict | `torch/nn/modules/module.py:407`（约 2800+ 行） |
| Parameter | 带 `_is_param` 标记的 Tensor，被 Module 自动识别为可训练参数 | `torch/nn/parameter.py:30`（元类 `_ParameterMeta` 见 `:19`） |
| 层级容器 | `Sequential`/`ModuleList`/`ModuleDict`/`ParameterList`/`ParameterDict` 组合子模块 | `torch/nn/modules/container.py:59,341,511,652,800` |
| 各种层实现 | 卷积/线性/归一化/池化/循环/Transformer/激活/损失等具体层 | `torch/nn/modules/{conv,linear,normalization,pooling,rnn,transformer,activation,loss,...}.py` |
| hooks 机制 | forward pre/full、backward pre/full、state_dict pre/post 钩子 | `module.py:1633,1696,1443,1469,2115,2139` |
| 无状态函数算子 | `functional.py` 提供 F.* 无状态函数，层内部调用 | `torch/nn/functional.py`（数千个函数，如 `max_pool2d` `:761`） |
| 序列化 | `state_dict()`/`load_state_dict()` 存取参数与 buffer | `module.py:2203,2325` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `class Module` | `module.py:407` | 网络模块基类；`__call__` 经 `_wrapped_call_impl` 触发 hooks 后调 `forward` |
| `class _ParameterMeta` | `parameter.py:19` | Parameter 的元类，控制 isinstance 语义 |
| `class Parameter(Tensor, metaclass=_ParameterMeta)` | `parameter.py:30` | 可训练参数张量 |
| `class Sequential(Module)` | `container.py:59` | 按顺序串联子模块 |
| `class ModuleList/ModuleDict` | `container.py:341,511` | 以 list/dict 持有子模块并自动注册 |
| `class _WrappedHook` | `module.py:74` | 包装 hook 回调，存储可移除句柄 |
| `Module.register_forward_hook` 等 | `module.py:1696` | 注册前向/反向钩子，返回 `RemovableHandle` |

## 3. 关键调用链

1. **一次前向**：用户 `output = model(x)` → `Module.__call__` → `_wrapped_call_impl`（`module.py`）→ 依次执行 forward pre-hooks → 调用用户定义 `forward`（内部通常调用 `functional.*` 或内置算子）→ 执行 forward hooks → 返回输出。
2. **参数注册**：`nn.Linear(...)` 构造时 `register_parameter`（`module.py:592`）把 `Parameter` 存入 `self._parameters`；`add_module`（`:642`）把子模块存入 `self._modules`。
3. **参数遍历**：`model.parameters()`（`module.py:2674`）递归遍历 `_modules` 与 `_parameters`，交给 `optim` 优化器。
4. **存取权重**：`model.state_dict()`（`module.py:2203`）递归收集参数/buffer 为有序字典；`load_state_dict` 反向加载并报告 missing/unexpected keys（`_IncompatibleKeys`，`module.py:40`）。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `bias`（各层构造） | 多数层默认带可训练偏置 | 各层 `__init__` |
| `requires_grad`（Parameter） | 新建 Parameter 默认 `True` | `parameter.py` |
| `deterministic` 模式 | 影响非确定性算子的告警/报错 | `torch/__init__.py`（见 torch-api） |
| 模块注册 hook | `register_module_*_registration_hook` 在注册阶段触发回调 | `module.py:139,165,191` |

## 5. 错误与重试语义

- `load_state_dict` 对缺失/多余 key 不抛异常，而是返回 `_IncompatibleKeys(missing_keys, unexpected_keys)`，默认 `strict=True` 时不一致才报错。
- hooks 执行中抛出的异常直接向上传播，不被 Module 吞掉；注册返回的 `RemovableHandle.remove()` 可卸载钩子。
- 层前向 shape 不匹配由 C++ 算子抛 `RuntimeError`，Python 层不重试。

## 6. 并发细节

- 纯 Python 层逻辑受 GIL 约束；层内调用的算子在 C++ 侧异步提交到设备流。
- 参数/buffer 以普通 dict（`_parameters`/`_buffers`/`_modules`）存储，单线程训练下无锁；多线程共享同一 Module 仅读前向、不并发写参数时安全，并发 `backward` 写梯度不安全。
- DataLoader 多进程 worker（见 data-loading）会在子进程深拷贝模型副本，不共享参数。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/nn/modules/` 全部层与 Module 基类、`torch/nn/parameter.py`、`torch/nn/functional.py`。

**Out-of-Scope（不在本仓库源码内 / 相邻叶子）**
- 优化器与学习率调度 → `optim` 叶子。
- 张量/算子本身与 `torch.ops` 命名空间 → `torch-api`。
- 算子内核实现（卷积/GEMM/CUDA kernel）→ cuda-backend 分片。
- 数据加载 → `data-loading` 叶子。

## 8. 与相邻子系统交互

- 用户模型代码 → 继承 `Module`，组合 `nn.modules.*` 层。
- Module 前向 → `functional.py F.*` / 内置算子 → `torch._C` Dispatcher（下游，见 torch-api）。
- Module.parameters() → `optim.Optimizer`（下游叶子）接收参数组。
- `state_dict()` → `torch.save/load`（serialization 叶子）持久化权重。

## 9. 语言专项适配口径

本项目为 Python 前端 + C++ 核心，不适用 Go/TS 专项。按能力缝归组：
- 服务定义侧：层的 `forward` 签名与 `functional.py` 函数签名；
- 提供方：`Module` 基类的注册/遍历/钩子框架、各具体层；
- 消费方：用户模型、`optim`、`serialization`。
C++ 边界在于 `functional.py`/层最终落到 `torch._C` 算子；Python 侧负责组合与状态管理，不持有设备内存。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| Module 层级与 functional 结构 | `nn-modules-architecture.html` | architecture | **standard**（showcase 因相邻连线标签间距未过 0 错误门槛，按脚本自动回退 standard 并如实披露） |

JSON IR 源文件：`json/nn-modules-architecture.json`。前向调用链已在第 3 节文字化说明，不另出时序图。

![nn-modules 架构图](nn-modules-architecture.html)
