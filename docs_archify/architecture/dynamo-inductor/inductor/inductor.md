# TorchInductor（inductor）

> 本文是 `dynamo-inductor` 域下的叶子子系统文档。域级总览见 `../dynamo-inductor.md`。
> 本文只展开后端代码生成、IR、调度与融合；字节码追踪见 `../dynamo/dynamo.md`，
> 编译入口与缓存见 `../compile-cache/compile-cache.md`，算子分解原语见 `../decomposition/decomposition.md`。
>
> 源码基准：pytorch，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| compile_fx 入口 | 接收 dynamo/AUT 产出的 FX GraphModule，降级为 Inductor IR | `torch/_inductor/compile_fx.py:compile_fx_inner`@1018、`FxCompile`@1508、`_InProcessFxCompile`@1535 |
| Inductor IR | TVM 风格张量中间表示：Pointwise/Reduction/View/Constant/FusedKernel | `torch/_inductor/ir.py:IRNode`@592、`Pointwise`@1222、`Reduction`@1401、`View`@3888 |
| 算子降级（lowering） | 把 FX 节点（aten 调用）逐个翻译为 IRNode | `torch/_inductor/lowering.py`（`lower_*` 函数族）、`dependencies.py` |
| 算子分解 | 编译期对未直接支持的 aten 算子做分解 | `torch/_inductor/decomposition.py` |
| 算子融合调度 | 决定哪些 IRNode 融合进同一个内核，调度计算与内存 | `torch/_inductor/scheduler.py:SchedulerBuffer`@2488、`FusionResult`@128 |
| 代码生成 | 为融合后的子图生成 Triton 内核（GPU）或 C++ 内核（CPU）+ 包装器 | `torch/_inductor/codegen/triton.py`、`cpp.py`、`wrapper.py`、`cuda/`、`rocm/`、`mps/` |
| 模式匹配重写 | 模式匹配 + 替换子图（如融合 pattern） | `torch/_inductor/pattern_matcher.py` |
| 自动调优 | 对 Triton 内核做 autotune 选最优配置 | `torch/_inductor/autotune_process.py`、`autotune/`、`heuristics/` |
| Triton 模板 | 常用算子的 Triton 模板（attention/conv/mm） | `torch/_inductor/kernel/`、`codegen/` 下 `cpp_*_template.py` |
| AOT 编译 | 提前编译为可独立加载的 AOTI 包 | `compile_fx.py:compile_fx_aot`@2408、`aoti_eager.py` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `compile_fx_inner` | `compile_fx.py:1018` | 后端入口：FX 图 → 降级 → IR → 调度 → 代码生成 → 可执行 callable |
| `IRNode` | `ir.py:592` | 所有张量 IR 节点基类；`Pointwise`/`Reduction`/`View` 为其子类 |
| `Scheduler` / `SchedulerBuffer` | `scheduler.py:2488` | 融合调度：把 IRNode 分组为可融合的 kernel 子图，决定执行顺序 |
| `Lowering` 上下文 | `lowering.py` | 持有当前 FX 节点到 IRNode 的映射，注册各 aten op 的 lower 规则 |
| `CachingAutotuner` | `codecache.py` | 缓存已编译 Triton 内核 |
| `FxGraph` | `graph.py` | 一次编译的完整 IR 图容器 |
| `PatternMatcher` | `pattern_matcher.py` | 子图模式匹配替换 |

## 3. 关键调用链

**链 A：inductor 后端编译一张 FX 图**
1. dynamo 调用后端入口 `compile_fx_inner`（`compile_fx.py:1018`），传入 FX GraphModule。
2. **Lowering**：`lowering.py` 逐节点把 aten 调用翻译为 `ir.py` 中的 IRNode；不支持的算子先经 `decomposition.py` 分解。
3. **建图**：所有 IRNode 组成 `FxGraph`，记录张量依赖与内存关系。
4. **调度**：`scheduler.py` 的 Scheduler 把可融合的 Pointwise/Reduction 分组为 fusion 组，决定每个内核的 tile/reduction 配置。
5. **代码生成**：按设备选后端——GPU 走 `codegen/triton.py` 生成 Triton 内核 + `wrapper.py` 包装器；CPU 走 `codegen/cpp.py` 生成 C++ 内核。
6. **编译与加载**：Triton 源码经 Triton JIT（外部组件）编译为 cubin，加载为可调用；CPU C++ 经 cpp_builder 编译为共享库。
7. 返回可执行 callable，与输入约定绑定。

**链 B：自动调优**
1. 对 GEMM/attention 等内核，`autotune_process.py` 启动多个候选配置，benchmark 后选最快。
2. 选中配置写入缓存（见 `../compile-cache/compile-cache.md`）。

## 4. 配置项

| 配置 | 默认/行为 | 位置 |
|------|-----------|------|
| `torch._inductor.config.max_autotune` | 是否开启自动调优 | `config.py` |
| `torch._inductor.config.triton.cudagraphs` | 是否启用 CUDA Graph 复用 | `config.py`、`cudagraph_trees.py` |
| `torch._inductor.config.fx_graph_cache` | FX 图级缓存开关 | `config.py` |
| `torch._inductor.cpp.threads` | CPU 后端线程数 | `config.py` |
| `compile_threads` | 并发编译线程数 | `config.py`、`async_compile.py` |
| AOTI 配置 | AOT 编译产物路径 | `config.py`、`standalone_compile.py` |

## 5. 错误与重试语义

- **编译失败回退**：inductor 编译抛错时，dynamo 层按 `suppress_errors` 决定是否回退 eager（见 `../dynamo/dynamo.md`）；inductor 自身不做透明重试。
- **autotune 失败**：某候选配置编译失败被跳过，选次优；全部失败则抛错。
- **内核缓存未命中**：重新走编译流程；磁盘缓存命中则直接加载已编译共享库。
- **不支持的算子**：lowering 阶段查表失败，触发 decomposition；仍不支持则上报 dynamo graph break。

## 6. 并发细节

- **异步编译**：`async_compile.py` / `compile_worker/` 支持后台线程编译，首次调用阻塞，后续调用可 overlap。
- **子进程编译**：`compile_fx_subproc.py` 在子进程中编译，避免污染主进程状态；`autotune_process.py` 也用子进程隔离 benchmark。
- **缓存锁**：磁盘缓存用 `FileLock`（`torch/utils/_filelock.py`）防止多进程同时写同一缓存项；`codecache.py` 管理编译产物。
- **CUDA Graph 复用**：`cudagraph_trees.py` 管理 CUDA Graph 树，按输入结构复用已捕获图，减少 launch 开销。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/_inductor/` 全部：ir、lowering、scheduler、codegen、pattern_matcher、autotune、cache、triton 模板

**Out-of-Scope（不在本仓库源码内）**
- Triton 编译器本身：Triton 是独立项目，inductor 仅生成其源码并调用 JIT
- CUDA/cuBLAS/cuDNN 运行时：经 ATen 调用
- FX 图捕获：见 `../dynamo/dynamo.md`、`../fx/fx.md`
- 磁盘缓存格式与锁：见 `../compile-cache/compile-cache.md`

## 8. 与相邻子系统交互

- 上游 → 本叶子：dynamo（`../dynamo/dynamo.md`）把 FX GraphModule 交给 inductor 作为后端；`torch.compile(backend="inductor")` 是默认后端。
- 本叶子 → 下游：调用 Triton JIT（外部）编译 GPU 内核；调用 cpp_builder 编译 CPU 内核；产出可执行 callable 回给 dynamo。
- 与 decomposition：lowering 阶段调用 `torch/_decomp/` 分解表（见 `../decomposition/decomposition.md`）。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，inductor 叶子纯 Python（`torch/_inductor/`），生成的是 Triton/C++ 源码。适配口径：

- **按能力缝归组**：compile_fx 入口 → lowering/decomposition → IR 构造 → scheduler 融合 → codegen 后端 → 外部编译器。
- **图类型选择**：architecture 表达"FX 图→降级→IR→调度→代码生成→Triton/CPU 运行时"的编译流水线；融合调度本身是图算法，不单列 state machine。
- **外部边界**：Triton 编译器、CUDA/cuBLAS 标注为"不在本仓库源码内"。
- **并发**：异步/子进程编译与 FileLock 是本叶子的并发重点，见第 6 节。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| TorchInductor 代码生成架构 | `inductor-architecture.html` | architecture | **standard**（showcase 因边穿过无关组件与短边标签重叠降 standard；已重排为横向流水线并移除短边标签，主流程语义不损失） |

JSON IR 源文件位于 `json/inductor-architecture.json`。
