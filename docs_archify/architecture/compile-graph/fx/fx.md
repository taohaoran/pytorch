# FX 图 IR（fx）

> 本文是 `compile-graph` 域下的叶子子系统文档。域级总览见 `../compile-graph.md`。
> 本文只展开 FX Graph IR 与 Tracer/Interpreter；TorchScript IR 见 `../torchscript-jit/torchscript-jit.md`，
> 基于 FX 的导出见 `../export/exported_program`。两者 IR 不同，不重复。
>
> 源码基准：pytorch，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| FX Graph IR | Python 级中间表示：`Graph` 持有有序 `Node` 列表，`Node.op` 为 `placeholder`/`call_function`/`call_module`/`get_attr`/`output` 之一 | `torch/fx/graph.py:Graph`@1401、`torch/fx/node.py:Node`@260 |
| 符号追踪（Tracer） | 以 `_Patcher` 猴子补丁替换 torch/nn.Module 调用，把执行中的算子记录成 Node；支持 `concrete_args` 静态特化控制流 | `torch/fx/_symbolic_trace.py:Tracer`@282、`_Patcher`@1164、`symbolic_trace()`@1390 |
| Proxy 抽象 | 包装一个 Node 的输出，重载 `__torch_function__`，使 `x + y` 等算子调用自动追加到 Graph | `torch/fx/proxy.py:Proxy`@600、`TracerBase`@186、`GraphAppendingTracer`@579 |
| GraphModule | 把 Graph 编译成可执行的 `torch.nn.Module`，动态生成 `forward` 源码 | `torch/fx/graph_module.py:GraphModule`@519 |
| 代码生成 | 把 Graph 反序列化为可读 Python 源码（`CodeGen`/`PythonCode`），供 GraphModule 与打印 | `torch/fx/graph.py:CodeGen`@369、`_PyTreeCodeGen`@1175 |
| 图变换 / Interpreter | 按拓扑序逐节点重放 Graph，`Transformer` 是可改写节点的子类 | `torch/fx/interpreter.py:Interpreter`@52、`Transformer`@518 |
| 子图重写 | 模式匹配替换子图（pattern matching） | `torch/fx/subgraph_rewriter.py` |
| 图工具 | pytree 输入输出扁平化、图序列化（pickle）、可视化 | `torch/fx/_pytree.py`、`_graph_pickler.py`、`passes/graph_drawer.py` |
| 内置 Passes | 假张量传播、常量折叠、规范化等 | `torch/fx/passes/`（`fake_tensor_prop.py`、`canonicalize.py`）、`experimental/`（`const_fold.py`、`sym_node.py`、`shape_inference/`） |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `Graph` | `torch/fx/graph.py:1401` | 持有 Node 列表、输入输出、插入点（`inserting_before`），提供 `node_copy`/`eliminate_dead_code`/`lint` |
| `Node` | `torch/fx/node.py:260` | 单节点：op、target（函数/模块名）、args/kwargs、返回元数据、stack_trace；提供 `replace_all_uses_with` |
| `Tracer` | `_symbolic_trace.py:282` | 默认符号追踪器：`trace(root)` 遍历模块前向，用 Proxy 记录 |
| `_Patcher` | `_symbolic_trace.py:1164` | 上下文管理器，临时替换全局 torch 函数与模块方法，退出时回滚 |
| `Proxy` | `proxy.py:600` | 绑定一个 Graph 与一个 Node，重载算术/张量操作把调用转成 `call_function` |
| `GraphModule` | `graph_module.py:519` | `nn.Module` 子类，`graph` 字段持图，`forward` 由 CodeGen 生成 |
| `Interpreter` | `interpreter.py:52` | 按 `run`(args)→按拓扑序 `run_node` 逐节点执行 |
| `Transformer` | `interpreter.py:518` | `Interpreter` 子类：执行同时把节点改写后写回新 Graph |
| `symbolic_trace` | `_symbolic_trace.py:1390` | 对外门面：`symbolic_trace(root, concrete_args=...) -> GraphModule` |

## 3. 关键调用链

**链 A：`symbolic_trace(fn)` 记录一张图**
1. 用户调 `torch.fx.symbolic_trace(root, concrete_args=...)`（`_symbolic_trace.py:1390`）。
2. `Tracer.trace(root)` 进入 `_Patcher` 上下文：把 `torch.*` 函数、`torch.nn.Module.__call__` 等替换为 `_create_wrapped_func`/`_create_wrapped_method`。
3. 真正执行 `root(*args)` 时，每次被 patch 的算子调用都通过 `Proxy.__torch_function__` 在新 `Graph` 上 `call_function` 追加 `Node`，返回新 `Proxy` 继续链式记录。
4. `concrete_args` 中的参数被折叠为常量节点，使 if/循环按本次取值静态化。
5. 追踪结束后 `GraphModule(root, graph)` 构造对象，`CodeGen` 把 Graph 编译成 `forward` Python 源码并 exec 进模块。

**链 B：`Interpreter(gm).run(args)` 重放**
1. `run`（`interpreter.py:148`）先跑 `placeholder` 节点绑定输入。
2. 按拓扑序 `run_node(n)`（@274）：按 `n.op` 分派到 `call_function`/`call_module`/`get_attr`，把结果存入 env。
3. `Transformer` 覆写 `run_node`：执行旧节点的同时把等价新节点 append 到输出图，实现"边执行边改写"。

## 4. 配置项

| 配置 | 默认/行为 | 位置 |
|------|-----------|------|
| `torch.fx.experimental.proxy_tensor` | ProxyTensor 模式下的追踪开关 | `torch/fx/experimental/proxy_tensor.py` |
| `torch.fx.graph_module._use_jit_script_for_methods` | 是否用 TorchScript 编译 GraphModule 方法 | `torch/fx/graph_module.py` |
| `torch.fx.experimental.symbolic_shapes` | 符号形状（SymInt）开启 | `torch/fx/experimental/symbolic_shapes.py` |
| 环境变量 `FX_GRAPH_PARTITION` 等 | 分区/实验特性开关 | `torch/fx/experimental/` |

## 5. 错误与重试语义

- **追踪失败**：遇到不支持的控制流/动态行为时，`Tracer` 抛 `TraceError`（`proxy.py:595`），通常提示用户用 `concrete_args` 或 `@wrap`；不做自动重试。
- **图校验**：`Graph.lint()` 检查拓扑顺序、输入输出一致性、游离节点，失败抛 `AssertionError`；在 `recompile`/导出前调用。
- **执行期**：`Interpreter.run_node` 对未识别 op 抛 `RuntimeError`；由调用方捕获回退到 eager（Dynamo 层处理，见 `../dynamo/dynamo.md`）。
- **不做自动重追踪**：同一 GraphModule 不会因新输入重 trace；重新追踪由用户显式调用。

## 6. 并发细节

- **追踪期**：`Tracer` 用线程局部状态（`_is_fx_tracing` 全局开关，`_symbolic_trace.py:65`）标记正在追踪；多线程下不同线程各自追踪，不共享同一个 `Graph`。
- **GraphModule 执行**：编译出的 `forward` 是普通 Python 函数，可被多线程并发调用；Graph 本身在构造后只读。
- **无内置锁**：`Graph.append/erase_node` 非线程安全，图变换必须在单线程内完成后再并发执行。
- **pytree 扁平化**：`_pytree.py` 处理嵌套输入输出，与并发无关。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/fx/` 全部：IR、Tracer、Proxy、GraphModule、Interpreter、passes、experimental
- 依赖 `torch.utils._pytree` 做输入输出扁平化

**Out-of-Scope（不在本仓库源码内）**
- 具体算子实现：FX 只记录 `call_function` 到 `aten::*`，算子本体在 ATen
- Dynamo 的字节码追踪：见 `../dynamo/dynamo.md`（Dynamo 产出 FX Graph，但 FX 本身不解析 Python 字节码）
- 导出序列化格式：见 `../export/export.md`
- TorchScript C++ IR：见 `../torchscript-jit/torchscript-jit.md`

## 8. 与相邻子系统交互

- 上游 → 本叶子：`torch/export/` 的 AOT 导出、`torch/_dynamo/` 的 `aot_graph`、`torch/_inductor/` 的后端都以 `GraphModule` 为输入；`torch.onnx.export` 新版路径也基于 FX。
- 本叶子 → 下游：Interpreter 执行时调用 ATen 算子；CodeGen 产出的 `forward` 源码可被 `exec` 为模块方法。
- 与 TorchScript：两套独立 IR；FX 是 Dynamo/export/inductor 时代的图中间表示，TorchScript 是旧 JIT 路径。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，FX 叶子纯 Python 实现于 `torch/fx/`。适配口径：

- **按能力缝归组**：Tracer（记录）→ Graph/Node（IR）→ CodeGen/GraphModule（物化）→ Interpreter/Transformer（执行与改写），而非 Go 控制器模式。
- **图类型选择**：architecture 表达追踪期与变换期的组件关系；图变换本身是线性管道（记录→改写→执行），用 architecture 已足够，不再单列 dataflow（避免与 inductor 数据流重复）。
- **外部边界**：ATen 算子、Triton 标注为外部组件。
- **无并发特例**：FX 不设计 goroutine/协程，线程模型见第 6 节。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| FX 图 IR 与追踪/解释架构 | `fx-architecture.html` | architecture | **standard**（showcase 因 user→tracer 短边标签与节点间距不足降 standard；已缩短标签为"trace"，主流程语义不损失） |

JSON IR 源文件位于 `json/fx-architecture.json`。
