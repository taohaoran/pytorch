# TorchScript JIT（torchscript-jit）

> 本文是 `compile-graph` 域下的叶子子系统文档。域级总览见 `../compile-graph.md`。
> 本文只展开 TorchScript IR 与 script/trace 编译执行链路；FX Graph IR 与 Tracer/Interpreter 见 `../fx/fx.md`，
> 基于 FX 的导出见 `../export/export.md`，二者 IR 不同，不重复。
>
> 源码基准：pytorch（Python 前端 + C++ 核心），commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| TorchScript SSA IR | `Graph`/`Node`/`Value`/`Block` 构成的静态单赋值图，Node 持有 opname 与 `FunctionSchema` | `torch/csrc/jit/ir/ir.h`（`Graph`@1194、`Node`@306、`Value`@168、`Block`@1038）、`torch/csrc/jit/ir/ir.cpp` |
| Scripting 前端 | 解析 Python 子集源码，词法/语法分析后由 `ir_emitter` 把 AST 翻译成 IR | `torch/csrc/jit/frontend/parser.cpp`、`ir_emitter.cpp`、`sugared_value.cpp`、`tree.h`；Python 入口 `torch/jit/_script.py:script()`@1276、`script_method()`@357 |
| Tracing 前端 | 执行一次前向，把遇到的 ATen 调用记录成一张图，不控制流/数据依赖 | `torch/csrc/jit/frontend/tracer.cpp`；Python 入口 `torch/jit/_trace.py:trace()`@838、`trace_module()`@1043 |
| IR 打印与解析 | 图的可读文本格式（graph printer / parser） | `torch/csrc/jit/ir/irparser.cpp`、`graph_utils.cpp` |
| 图优化 Passes | 约 60+ 个独立 pass：DCE、常量传播、Common Subexpression Elimination、算子分解、conv-BN folding、freeze、融合等 | `torch/csrc/jit/passes/`（如 `dead_code_elimination.cpp`、`constant_propagation.cpp`、`common_subexpression_elimination.cpp`、`freeze_module.cpp`、`fold_conv_bn.cpp`、`create_autodiff_subgraphs.cpp`） |
| Profiling 图执行器 | 首次执行记录张量 shape/类型，按 profile 信息特化编译出执行 Plan，再下发解释器 | `torch/csrc/jit/runtime/profiling_graph_executor_impl.h/.cpp`、`graph_executor.h:GraphExecutor`@62 |
| 解释器执行 | 把 Graph 编译为线性 `Instruction` 指令流，基于 `Stack<IValue>` 逐条执行 `callOperator` | `torch/csrc/jit/runtime/interpreter.h`（`Code`@45、`InterpreterState`@91）、`interpreter.cpp`、`instruction.h` |
| 序列化/反序列化 | 模块保存为 `.pt`（pickle 常量 + flatbuffer/pickle 图），移动端 mobile bytecode 格式 | `torch/csrc/jit/serialization/export.cpp`、`import.cpp`、`pickler.cpp`、`unpickler.cpp`、`flatbuffer_serializer.cpp`、`export_bytecode.cpp`；Python 侧 `torch/jit/_serialization.py` |
| 模块封装 | `ScriptModule`/`ScriptFunction` 对外 Python 句柄，C++ 侧 `Module`/`Function` | `torch/csrc/jit/api/module.h`、`function_impl.h`；Python 侧 `torch/jit/__init__.py` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `Graph` | `torch/csrc/jit/ir/ir.h:1194` | 一张 TorchScript 图的根对象，持有顶层 Block、输入/输出 Value、插入点栈 |
| `Node` | `ir.h:306` | 图节点：opname（如 `aten::add`）、schema 引用、输入/输出 Value、所属 Block、scope |
| `Value` | `ir.h:168` | SSA 中的一条 SSA 值，携带类型（`TypePtr`）与唯一 debug 下标 |
| `Block` | `ir.h:1038` | 基本块：容纳一组 Node，对应 if/loop 子图 |
| `GraphExecutor` | `runtime/graph_executor.h:62` | 图执行器门面；`ProfilingGraphExecutorImpl` 负责 plan 选择与 profiling |
| `ExecutionPlan` | `runtime/graph_executor.h:25` | 一份"特化后可执行"的编译产物（含 `Code` 与 profiling 信息） |
| `Code` / `InterpreterState` | `runtime/interpreter.h:45/91` | 线性化指令流与其运行态（Stack<IValue>） |
| `Parser` | `frontend/parser.h` | 把 TorchScript 源码文本解析为 Tree，再由 `ir_emitter` 转 IR |
| `torch.jit.script` / `trace` | `torch/jit/_script.py:1276` / `_trace.py:838` | 对外编译入口 |

## 3. 关键调用链

**链 A：`torch.jit.script(fn)` 静态编译**
1. Python 层 `script()`（`torch/jit/_script.py:1276`）判断输入是函数还是 Module，调用 `torch._C._ScriptMethod`/`_script_from_make_module` 走 C++ 绑定。
2. C++ 前端用 `parser.cpp` 对注解后的 Python 子集做词法/语法分析得到 Tree。
3. `ir_emitter.cpp` 把 Tree 按作用域翻译成 `Graph`，`sugared_value.cpp` 做类型推导与 schema 匹配。
4. 图经 `passes/` 下各优化 pass（DCE、常量传播、autodiff 子图划分等）改写。
5. 交给 `ProfilingGraphExecutorImpl`：首次运行先 profiling，后续按 shape 特化出 `ExecutionPlan`，下发 `InterpreterState` 执行。

**链 B：`torch.jit.trace(fn, example_inputs)` 追踪**
1. `trace()`（`torch/jit/_trace.py:838`）以 example_inputs 实际跑一次 `fn`。
2. `torch/csrc/jit/frontend/tracer.cpp` 接管 ATen 算子调用（通过 dispatcher 的 tracer 模式），把每次调用追加为 `Node`，记录输入输出 Value。
3. 控制流（if/while/动态 shape）不被记录，按本次执行的具体路径固化为静态图；同样进入优化与执行器。

**链 C：`torch.jit.save(m)` 序列化**
1. `torch/jit/_serialization.py` 调 C++ `Module._save_for_export`。
2. `serialization/export.cpp` + `pickler.cpp` 把参数/缓冲区常量与图（IR 文本/flatbuffer）分块写入 `.pt`；`import.cpp`/`unpickler.cpp` 反向重建 `Module`。

## 4. 配置项

| 配置 | 默认/行为 | 位置 |
|------|-----------|------|
| `torch.jit.optimized_execution` | 控制是否启用执行期优化 | `torch/jit/__init__.py` |
| `torch._C._set_grad_enabled` / 编译期 autocast pass | 开启 autocast 图改写 | `torch/csrc/jit/passes/autocast.cpp` |
| 融合开关（fuser） | 开启 NVFuser/CPU fuser 子图融合 | `torch/csrc/jit/passes/canonicalize_graph_fuser_ops.cpp`、`torch/jit/_fuser.py` |
| `torch.jit.freeze` | 冻结模块（常量折叠、参数内化） | `torch/jit/_freeze.py`、`passes/freeze_module.cpp` |
| mobile bytecode 导出 | `torch.jit.mobile` 子命令 | `torch/jit/mobile/`、`serialization/export_bytecode.cpp` |

## 5. 错误与重试语义

- **编译期错误**：scripting 阶段类型不匹配/语法不支持时，`frontend/error_report.cpp` 给出带源码行号的诊断并抛出 Python 异常；不做重试，直接报错。
- **tracing 与 eager 语义偏差**：trace 把数据依赖固化为常量，运行时形状/分支变化不会重追踪；由用户用 `torch.jit.trace` 的 `check_trace` 校验。
- **运行期**：`InterpreterState` 执行算子抛错时，`runtime/jit_exception.cpp` 把 C++ 异常包装为 Python 异常并附带图节点的 source range 调试栈。
- **profiling 回退**：若特化 plan 与新输入 shape 不匹配，`ProfilingGraphExecutorImpl` 重新 profiling 生成新 plan，对调用方透明。

## 6. 并发细节

- **解释器线程模型**：`InterpreterState` 不内置并行；多线程推理靠 `GraphExecutor` 级别的并发调用（每个调用持有自己的 `Stack`），共享只读的 `Code`。
- **并行子图**：融合后的算子子图可通过 `parallel_launch` 在多线程执行（`runtime/interpreter` 中的 fork/wait 指令）。
- **编译缓存**：`GraphExecutor` 对同一 `Graph` 缓存已特化的 `ExecutionPlan`，按输入类型/shape 查表命中；无显式用户锁，由 GIL 与 C++ 构造期一次性完成。
- **无 goroutine 概念**：本叶子为 C++，不存在 Go 的 channel/workqueue；异步部分由 `torch/jit/_async.py` 与 future 机制支持。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/csrc/jit/ir/`：TorchScript SSA IR 定义
- `torch/csrc/jit/frontend/`：scripting 解析与 tracing
- `torch/csrc/jit/passes/`：全部图优化 pass
- `torch/csrc/jit/runtime/`：GraphExecutor 与解释器
- `torch/csrc/jit/serialization/`：save/load
- `torch/jit/`：Python 门面与 `torch.jit.script/trace/save`

**Out-of-Scope（不在本仓库源码内）**
- 底层 CUDA/cuDNN 算子实现：由 ATen 后端与第三方库提供，仅经 `callOperator` 调用
- Triton 代码生成：见 `../inductor/inductor.md`，TorchScript 自身不用 Triton
- FX 图的 Python 侧变换：见 `../fx/fx.md`
- ONNX 导出：见 `../onnx/onnx.md`

## 8. 与相邻子系统交互

- 上游 → 本叶子：用户代码直接调 `torch.jit.script/trace/save`；`torch/onnx/` 的 legacy ONNX 导出基于 TorchScript 图（`serialization/onnx.cpp`）。
- 本叶子 → 下游：`Interpreter` 通过 c10 dispatcher 调用 ATen 算子；`passes/freeze_module` 等产出可被移动端 `torch::jit::mobile` 消费。
- 与 FX 叶子的关系：TorchScript IR（C++ SSA 图）与 FX Graph（Python `GraphModule`）是两套独立 IR；`torch/export/` 与 Dynamo 走 FX 体系，不复用 TorchScript IR。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，不适用 Go 专项（无 goroutine/workqueue）也不适用 TS 专项。本叶子的适配口径：

- **编译栈以 C++ 为主**：IR、passes、解释器均在 `torch/csrc/jit/`；Python 侧仅作门面与少量工具（`torch/jit/_passes`、`_freeze.py`）。
- **按能力缝归组**：前端（scripting/tracing）→ IR 中间表示 → 优化 passes → 执行运行时 → 序列化，与 Go 控制器/informer 模式无关。
- **图类型选择**：用 architecture 表达"前端→IR→passes→执行器→算子"的编译流水线；执行期 plan 选择本质是配置查找而非状态机，不单列 lifecycle 图。
- **外部边界标注**：ATen 算子后端、CUDA/cuDNN 标注为"不在本仓库源码内"。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| TorchScript JIT 架构 | `torchscript-jit-architecture.html` | architecture | **standard**（showcase 因 graph_executor→interpreter、interpreter→ops 两条短边上标签与节点间距不足 4px 降 standard；已在 JSON 中移除这两条短边的文字标签，主流程语义不损失） |

JSON IR 源文件位于 `json/torchscript-jit-architecture.json`。
