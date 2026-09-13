# Python 侧自动求分 API（autograd-python）

> 本文是 `autograd` 域下的叶子子系统文档。域级总览见 `../autograd.md`。
> 本文只展开 **Python 侧用户-facing 的自动求分 API**：自定义反向 `Function`、grad 模式开关、functional 线性算子、graph 可视化；不重复展开：
> - C++ 反向引擎如何执行计算图 → 见 `../autograd-engine/autograd-engine.md`
>
> 源码基准：PyTorch（Python 前端），commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`，目录 `torch/autograd/`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `torch.autograd.Function` | 用户自定义可微算子：子类实现 `forward`/`backward` | `torch/autograd/function.py`（`class Function` 第 560 行） |
| `Function.apply()` 用户入口 | `MyFunction.apply(*args)` 触发前向并记录反向节点 | `function.py`（Function 类静态 apply） |
| `FunctionCtx` | forward/backward 间通过 ctx 共享信息（saved_tensors 等） | `function.py:38`（`class FunctionCtx`） |
| `FunctionMeta` 元类 | 动态生成分 diff 版本类、连接 forward/backward | `function.py:387`（`class FunctionMeta`） |
| 引擎侧 apply/apply_boxed/apply_jvp | 反向引擎调用 Python 自定义反向的入口 | `function.py:354/368/376` |
| `no_grad` / `enable_grad` | 禁用/启用梯度记录的上下文管理器 | `grad_mode.py:22`（no_grad）、`:89`（enable_grad） |
| `set_grad_enabled` | 可作上下文管理器或装饰器开关梯度 | `grad_mode.py:144` |
| `inference_mode` | 推理模式：更高性能地禁用版本计数与视图追踪 | `grad_mode.py:213` |
| `functional.vjp/jvp` | 向量-Jacobian 积 / Jacobian-向量积 | `functional.py:271`（vjp）、`:366`（jvp） |
| `functional.jacobian/hessian` | 数值/前向模式计算雅可比/黑塞 | `functional.py:587`（jacobian）、`:856`（hessian） |
| `graph.Node` 抽象 | Python 侧计算图节点抽象（name/next_functions/metadata） | `graph.py:59`（`class Node`） |
| 反向运行封装 `_engine_run_backward` | Python 调 C++ Engine 的胶水 | `graph.py:1082` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `torch.autograd.Function` | `function.py:560` | 用户自定义算子基类。子类须实现 `forward(ctx, *args)` 与 `backward(ctx, *grad_outputs)` |
| `Function.apply`（实例） | `function.py:354` | 引擎反向时调用用户 backward；`boxed_grads_call` 控制是否把梯度打包成 list |
| `FunctionCtx` | `function.py:38` | 容器：`save_for_backward`、`mark_dirty`、`mark_non_differentiable` |
| `FunctionMeta` | `function.py:387` | 元类：为每个 Function 生成 `_backward_cls` |
| `no_grad`/`enable_grad`/`inference_mode`/`set_grad_enabled` | `grad_mode.py` | 梯度模式 RAII 上下文管理器（`__enter__/__exit__`） |
| `functional.vjp/jvp/jacobian/hessian` | `functional.py` | 基于反向/前向模式的线性求积与雅可比/黑塞工具 |
| `graph.Node` | `graph.py:59` | Python 计算图节点抽象基类（ABC） |

## 3. 关键调用链

**调用链一：用户自定义 Function 的前向+记录反向**

1. 用户定义 `class MyFn(Function)` 实现 `forward(ctx,x)` 与 `backward(ctx,gy)`。
2. 调用 `MyFn.apply(x)`：`FunctionMeta` 注册并构造对应的 autograd Node。
3. 执行 `forward`，经 `ctx.save_for_backward` 保存需要的张量；输出张量被挂上 `grad_fn=MyFn 的反向 Node`。
4. 反向时 C++ 引擎（autograd-engine）调用该 Node 的 `apply(grads)`（`function.py:354`），转发到用户 `backward(ctx, grads)`。

**调用链二：no_grad 上下文**

1. `with torch.no_grad():` 进入 `no_grad.__enter__`（`grad_mode.py:81`）。
2. 该上下文把 C++ 侧 `GradMode` 设为 false，此后算子不再构造反向节点。
3. `__exit__` 恢复原状态（RAII）。

**调用链三：functional.vjp**

1. `vjp(func, inputs, v)`（`functional.py:271`）：先跑 `func(inputs)` 得到输出，再按 v 反向计算梯度向量。

## 4. 配置项

| flag / option | 默认 / 行为 | 位置 |
|---------------|-------------|------|
| `torch.is_grad_enabled()` | 当前梯度记录开关 | `grad_mode.py` |
| `torch.is_inference_mode_enabled()` | 推理模式开关 | `grad_mode.py:213` |
| `create_graph`（vjp/jvp 等） | 是否构建高阶导图 | `functional.py` |
| `anomaly_mode` | 由 `torch.autograd.detect_anomaly` 开启，定位反向错误 | `anomaly_mode.py` |

## 5. 错误与重试语义

- **未实现 backward**：用户 Function 缺 backward 时，反向调用抛 `NotImplementedError`/运行时错误，提示需要实现。
- **保存张量冲突**：`ctx.mark_dirty`/版本计数错误在反向期抛错。
- **grad 模式错误使用**：在 no_grad 下对 requires_grad 张量求梯度，按规则在反向期报错。
- 无重试；Python 异常透传到调用方。

## 6. 并发细节

- **GIL**：Python 侧 API 在解释器内运行；C++ 引擎执行反向时释放 GIL（autograd-engine），回到 Python 钩子时重新获取。
- **grad 模式 TLS**：`no_grad`/`inference_mode` 改的是 C++ 线程局部 `GradMode` 状态（`c10/core/GradMode`），Python 上下文管理器只是 RAII 包装。
- **无线程池**：本叶子是 API 层，不建线程；实际并行在 C++ 引擎。
- **ctx 非线程安全**：单个 Function 实例的 ctx 由单次反向执行使用，不跨任务共享。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/autograd/function.py`、`grad_mode.py`、`functional.py`、`graph.py`、`variable.py`

**Out-of-Scope（不在本仓库源码内）**
- C++ 反向引擎与 Node 执行 → `../autograd-engine/autograd-engine.md`
- 前向算子如何挂 grad_fn → `core` 域 aten-dispatch 包装层
- profiler/gradcheck 等工具（同目录但非本叶子重点）

## 8. 与相邻子系统交互

- 上游 → 本叶子：用户脚本调用 `Function.apply` / `no_grad` / `vjp`。
- 本叶子 → 下游：
  - `Function.apply` 经 `torch/csrc` 绑定调用前向算子（aten）；
  - 反向节点回调到 C++ 引擎（autograd-engine）的 `apply`；
  - `grad_mode` 上下文改 C++ `GradMode` TLS。
- 方向：用户 → 本叶子 Python API → C++ 绑定 → aten 前向算子 / C++ 反向引擎。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，不适用 Go/TS 专项。本叶子适配口径（Python 侧能力缝）：
- **能力缝分组**：按"用户可扩展点"分组——`Function`（自定义反向）、`grad_mode`（梯度开关）、`functional`（线性求积工具）、`graph`（节点抽象）。
- **Python↔C++ 边界**：Python Function 经 `python_cpp_function.cpp`（torch/csrc）桥接到 C++ Node；本叶子只描述 Python 侧契约。
- **装饰器/上下文管理器**：`no_grad`/`inference_mode` 既是 context manager 又是装饰器（`_DecoratorContextManager`）。
- **GIL**：Python API 层持 GIL，下沉 C++ 引擎时释放。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| Python 自动求分 API 架构图 | `autograd-python-architecture.html` | architecture | standard（降档披露：垂直边标签贴节点，去标签后按 standard 渲染成功） |
| Function.apply 调用数据流图 | `autograd-python-dataflow.html` | dataflow | standard（降档披露：节点中文 label 宽度在 showcase 校验下超宽，按规则降 standard） |

- JSON IR 源文件：`json/` 下对应两个文件。
- 说明：API 组成用 architecture；apply 前向→记录节点→反向回调用 dataflow。
