# TorchDynamo（dynamo）

> 本文是 `dynamo-inductor` 域下的叶子子系统文档。域级总览见 `../dynamo-inductor.md`。
> 本文只展开字节码追踪、guard 与符号形状；后端代码生成见 `../inductor/inductor.md`，
> 编译入口与磁盘缓存见 `../compile-cache/compile-cache.md`。
>
> 源码基准：pytorch，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| eval_frame 钩子 | 替换 CPython 帧求值入口，在每帧执行前决定走编译还是 eager | `torch/_dynamo/eval_frame.py`（`DynamoStance`@202、`_create_wrapped_callback`@353） |
| 帧转换入口 | `convert_frame`：判断是否编译当前帧，构造编译上下文 | `torch/_dynamo/convert_frame.py:convert_frame_assert`@815、`trace_frame`@892、`ConvertFrameBox`@577 |
| 字节码符号执行 | 逐字节码指令把 Python 语义翻译为 FX 操作；支持内联、异常栈、推测日志 | `torch/_dynamo/symbolic_convert.py:InstructionTranslator`@5587、`InstructionTranslatorBase`@1465 |
| Variables 类型系统 | 包装 Python 值/张量/容器，决定该字节码对应什么 FX 节点 | `torch/_dynamo/variables/`（dtype/builder/tensor/torch_function 等） |
| Guards 守卫 | 把"输入必须满足的不变量"（类型、形状 id、取值范围）编译为可求值代码；失效即重编译 | `torch/_dynamo/guards.py:GuardManagerWrapper`@298 |
| 符号形状 | 用 SymInt/SymExpr 表达动态维度，传播 shape 约束 | `torch/fx/experimental/symbolic_shapes.py`（ShapeEnv/SymExpr，Dynamo 经 `eval_frame.py`/`convert_frame.py` 使用） |
| 图 break 提示 | 遇到不可追踪构造时优雅退出 eager，记录原因 | `torch/_dynamo/graph_break_hints.py`、`graph_break_registry.json` |
| Repro 日志 | 把失败/图 break 现场记录为可复现脚本 | `torch/_dynamo/repro/` |
| 编译配置 | 重编译上限、缓存大小、日志级别等 | `torch/_dynamo/config.py`（`accumulated_recompile_limit=256`@124、`cache_size_limit`@137） |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `set_eval_frame` / `DynamoStance` | `eval_frame.py:202` | 全局开关：开启/关闭/只读 stance |
| `convert_frame_assert` | `convert_frame.py:815` | 帧入口：尝试编译，失败抛错（debug 用） |
| `InstructionTranslator` | `symbolic_convert.py:5587` | 字节码解释主循环：维护 LocalState/BlockStack，把每条字节码分派到处理函数 |
| `LocalState` | `symbolic_convert.py:401` | 追踪期局部状态：当前 BytecodeInstruction 栈、输出图 |
| `GuardManagerWrapper` | `guards.py:298` | 收集并编译 guard 表达式；运行时求值 |
| `OutputGraph` | `torch/_dynamo/output_graph.py` | 追踪期累积 FX 节点，产出最终 GraphModule |
| `ReplayRecord` | `torch/_dynamo/replay_record.py` | 记录编译现场用于 repro |

## 3. 关键调用链

**链 A：首次 `f(x)` 触发 torch.compile**
1. 用户前向调用进入被装饰函数，CPython 调 `eval_frame` 钩子（`eval_frame.py`）。
2. 钩子查 guard 缓存：若该帧已编译且输入满足 guard，直接跳已编译代码。
3. 未命中则 `convert_frame_assert`（`convert_frame.py:815`）构造编译上下文。
4. `InstructionTranslator`（`symbolic_convert.py:5587`）按字节码顺序符号执行：遇到 `LOAD_FAST`/`BINARY_OP` 等，通过 Variables 系统把 Python 操作翻译为 FX 节点追加到 `OutputGraph`。
5. 遇到不可追踪的 Python 构造（如动态 `eval`、不支持的内建函数）触发 **graph break**：当前帧先以 eager 跑完，剩余帧再继续追踪。
6. 追踪完成，把 `OutputGraph` 的 FX GraphModule 交给 AOTAutograd（functorch 分片）→ inductor（见 `../inductor/inductor.md`）。
7. 编译产物连同 guard 一起写回缓存；下次同形输入直接命中。

**链 B：guard 失效重编译**
1. 下次 `f(x')` 输入形状/类型变化，`GuardManagerWrapper` 求值 guard 失败。
2. Dynamo 以新输入重新走链 A；`accumulated_recompile_limit`（默认 256，`config.py:124`）超过后停止重编译并告警。

## 4. 配置项

| 配置 | 默认/行为 | 位置 |
|------|-----------|------|
| `torch._dynamo.config.cache_size_limit` | 每帧最大重编译次数 | `config.py:137` |
| `torch._dynamo.config.accumulated_recompile_limit` | 全局累计重编译上限，默认 256 | `config.py:124` |
| `torch._dynamo.config.log_level` / `repro` | 日志与 repro 目录 | `config.py`、`logging.py` |
| `torch._dynamo.config.suppress_errors` | 编译失败时静默回退 eager | `config.py` |
| `dynamic_shapes` | 动态形状开关 | `config.py`、`torch/fx/experimental/symbolic_shapes.py` |
| `torch._dynamo.disable` / `torch.compile(backend=...)` | 禁用/选择后端 | `eval_frame.py`、`torch/_compile.py` |

## 5. 错误与重试语义

- **编译失败回退 eager**：`suppress_errors=True`（或默认生产模式）时，编译抛错被捕获，Dynamo 打印警告后回退到 eager 执行，不影响正确性。
- **guard 失效重编译**：guard 求值失败不报错，触发重新追踪；超过 `cache_size_limit` 后该帧不再编译，永久走 eager。
- **graph break**：不是错误，是预期的受控中断；`graph_break_hints.py` 给出原因，repro 目录记录现场。
- **repro 最小化**：`repro/` 与 `minifier` 把失败现场裁剪为最小可复现脚本。

## 6. 并发细节

- **全局 eval_frame 钩子**：进程级单例，由 GIL 保护；多线程下各线程独立追踪各自的帧。
- **编译缓存**：`eval_frame.py` 的缓存按 code object 键控，写缓存用线程局部状态；多线程同时触发同帧编译会串行化（首次编译完成前其他线程走 eager）。
- **子进程编译**：inductor 可在子进程中编译（见 `../inductor/inductor.md`），dynamo 本身不 fork。
- **无内部锁竞争**：追踪期是单线程顺序执行字节码，无共享可变图。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/_dynamo/` 全部：eval_frame、convert_frame、symbolic_convert、guards、variables、output_graph、repro、config

**Out-of-Scope（不在本仓库源码内）**
- AOTAutograd 前向/反向图分区：见 functorch 分片（dynamo 调用但不实现）
- 后端代码生成与融合：见 `../inductor/inductor.md`
- 编译入口装饰器与磁盘缓存：见 `../compile-cache/compile-cache.md`
- Python 解释器本身：CPython 不在本仓库

## 8. 与相邻子系统交互

- 上游 → 本叶子：用户调 `torch.compile(f)(x)`（见 `../compile-cache/compile-cache.md`）；CPython 帧求值钩子是入口。
- 本叶子 → 下游：产出 FX GraphModule 交给 AOTAutograd 与 inductor 后端；guard 表达式被缓存复用。
- 与 export：`torch/export/` 复用 dynamo 的字节码追踪能力（见 `../export/export.md`）。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，dynamo 叶子纯 Python（`torch/_dynamo/`）。适配口径：

- **按能力缝归组**：eval_frame 钩子 → convert_frame 入口 → InstructionTranslator 字节码符号执行 → Variables 类型系统 → OutputGraph 产出 FX → Guards 守卫。
- **图类型选择**：architecture 表达组件关系；sequence 表达"首次 torch.compile 调用链"（帧进入→追踪→guard→AOT→inductor→返回），这是 Dynamo 最核心的时序场景。
- **外部边界**：CPython 解释器、AOTAutograd、inductor 标注为相邻/外部组件。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| TorchDynamo 字节码追踪架构 | `dynamo-architecture.html` | architecture | **standard**（showcase 因垂直边标签与节点重叠降 standard；已移除垂直边文字标签，主流程语义不损失） |
| torch.compile 首次编译调用链 | `dynamo-sequence.html` | sequence | **standard**（showcase 因参与者标签宽度与自消息跨度不足降 standard；已缩短参与者名、改自消息为跨参与者消息，时序语义不损失） |

JSON IR 源文件位于 `json/dynamo-architecture.json`、`json/dynamo-sequence.json`。
