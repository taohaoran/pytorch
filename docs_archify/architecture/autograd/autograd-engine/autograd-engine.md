# C++ 反向计算引擎（autograd-engine）

> 本文是 `autograd` 域下的叶子子系统文档。域级总览见 `../autograd.md`。
> 本文只展开 **C++ 侧反向执行引擎、计算图节点（Node）、ReadyQueue 线程池调度**；不重复展开：
> - Python 侧 `torch.autograd.Function` 自定义反向 API 与 grad 模式开关 → 见 `../autograd-python/autograd-python.md`
> - 前向算子如何经 Dispatcher 选中并构造节点 → `core` 域
>
> 源码基准：PyTorch（C++ 核心），commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`，目录 `torch/csrc/autograd/`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| Engine 单例 | 反向计算的全局执行引擎；持有线程池与 device ready queue | `torch/csrc/autograd/engine.cpp`（`Engine::Engine` 第 273 行） |
| execute() 入口 | 启动一次反向：给定根节点与初始梯度，阻塞至完成 | `engine.cpp:1363`（`Engine::execute`） |
| evaluate_function() | 执行单个 Node：等待输入就绪→调 pre-hooks→调 `apply`→把输出分发到下游边 | `engine.cpp:1103`（`Engine::evaluate_function`） |
| thread_main() | worker 线程主循环：从 ready_queue 取 NodeTask 执行 | `engine.cpp:516`（`Engine::thread_main`） |
| ReadyQueue | 线程安全任务队列：push/pop/empty | `engine.cpp:230`（push）、`:258`（pop） |
| GraphTask | 一次反向任务的上下文：依赖计数、exec_info、captured vars | `engine.cpp`（`GraphTask` 类）、`graph_task.h` |
| NodeTask | 队列中的工作单元：GraphTask 弱引用 + Node* + 输入缓冲 | `engine.cpp` |
| Node（计算图节点） | 一个可微操作的反向；持有 next_edges_ 与 apply 虚函数 | `torch/csrc/autograd/node.h`（`class Node` 约第 66 行） |
| Edge | 有向边：(Node* 函数, 输入槽位 input_nr) | `torch/csrc/autograd/edge.h` |
| InputBuffer | 按 incoming edge 聚合梯度，全部到齐后标记节点就绪 | `torch/csrc/autograd/input_buffer.cpp` |
| Variable / AutogradMeta | Tensor 的 autograd 元数据：grad_fn_、requires_grad、is_view | `torch/csrc/autograd/variable.h`（`AutogradMeta` 第 100 行） |
| Hooks（pre/post/tensor hooks） | 前/后向钩子调用 | `engine.cpp:840`（`call_pre_hooks`）、`cpp_hook.*` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `torch::autograd::Engine` | `engine.cpp` | 反向引擎单例。`execute`/`evaluate_function`/`thread_main`；维护 `device_ready_queues_`、TLS `local_ready_queue`（第 139 行） |
| `torch::autograd::Node` | `node.h:66` | 计算图节点基类。`next_edges_`（下游边列表）、`apply(InputBuffer)->variable_list`（纯虚）、`add_input_metadata`（第 206 行） |
| `torch::autograd::Edge` | `edge.h` | `(Node* function, uint32_t input_nr)` |
| `torch::autograd::GraphTask` | `graph_task.h` | 一次反向任务的共享上下文；`not_ready_input_` 依赖计数、exec_info_、mutex_ |
| `torch::autograd::ReadyQueue` | `engine.cpp:230` | 线程安全任务队列 |
| `torch::autograd::InputBuffer` | `input_buffer.cpp` | 聚合某节点的所有入边梯度；全到齐则触发就绪 |
| `torch::autograd::AutogradMeta` | `variable.h:100` | Tensor 上的 autograd 元数据（grad_fn_、grad、requires_grad、is_view） |
| `Engine::evaluate_function` | `engine.cpp:1103` | 单节点执行核心：stream 同步、hooks、apply、下游入队 |

## 3. 关键调用链

**调用链一：torch.autograd.backward(root, grad) 反向**

1. Python 层（autograd-python）调用 `Engine::execute(root, grads, ...)`（`engine.cpp:1363`）。
2. execute 构造 `GraphTask`，把根节点加入 CPU ready_queue，调用 `thread_main`。
3. `thread_main`（`engine.cpp:516`）循环从 `local_ready_queue->pop()` 取 `NodeTask`。
4. 对每个 NodeTask 调 `evaluate_function(graph_task, func, inputs, ...)`（第 1103 行）：
   - 等待加速器输入 stream event 同步（第 1124-1140 行）；
   - 调 `call_pre_hooks`（第 1153 行）；
   - 调 `func->apply(inputs)` 得到梯度输出；
   - 对每个 `next_edge`：把梯度写入下游节点的 `InputBuffer`，若依赖计数归零则把下游 `NodeTask` push 到对应 device ready_queue（第 1256-1275 行附近 `queue->push(...)`）。
5. 所有节点完成后，GraphTask 标记完成，execute 返回。

**调用链二：多线程并行**

- 每个 device 有自己的 ReadyQueue；worker 线程（`init_local_ready_queue`）从各自队列取任务。CPU 任务由调用者线程兼处理（`should_run_in_cpu_ready_queue`，第 62 行）。
- 节点间通过 InputBuffer 的依赖计数同步，无需锁；GraphTask 完成用条件变量唤醒主线程。

## 4. 配置项

| flag / option | 默认 / 行为 | 位置 |
|---------------|-------------|------|
| 反向线程数 | 由 `torch.set_num_threads`/线程池决定 | `engine.cpp` 线程池 |
| `anomaly_mode` | 开启后额外记录每次反向历史，便于定位错误 | `anomaly_mode.cpp` |
| `functionalize`/`inference_mode` | 经 DispatchKey 体系影响是否记录节点（见 c10-core/aten-dispatch） | — |
| `Engine::stop()` | 关闭：向所有队列发 shutdown task | `engine.cpp:282` |

## 5. 错误与重试语义

- **节点 apply 抛异常**：在 worker 线程捕获，存入 GraphTask，广播到等待的调用线程并重新抛出；无重试。
- **图任务依赖计数**：InputBuffer 记录每节点入边数，全部到齐才就绪；计数错误触发 `TORCH_INTERNAL_ASSERT`。
- **异常时 GraphTask 取消**：异常发生后后续节点不再调度，已入队任务检测到异常快速退出。
- 无 IO 重试。

## 6. 并发细节

- **多 worker 线程**：`thread_main` 是 worker 主循环；每个 device 一个 ReadyQueue 队列。TLS `local_ready_queue`（`engine.cpp:139`）让 worker 知道自己的队列。
- **无锁节点调度**：节点间数据依赖经 InputBuffer 依赖计数传递，生产者把梯度写入后检查计数，归零时才 push 下游——典型"数据驱动的任务图"，而非共享工作队列竞争。
- **共享状态加锁**：`GraphTask::mutex_`（`engine.cpp:1158`）保护 captured_vars_ 等共享写。
- **stream/event 同步**：跨 device 梯度经 CUDA event 等待（第 1136-1138 行）。
- **GIL**：C++ 引擎执行期不持 GIL；与 Python 回调（hook）的桥接在 python_engine。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/csrc/autograd/`：Engine/Node/Edge/GraphTask/ReadyQueue/InputBuffer/Variable/AutogradMeta/hooks

**Out-of-Scope（不在本仓库源码内）**
- Python 侧 Function 自定义反向 API → `../autograd-python/autograd-python.md`
- 前向算子构造节点（VariableType kernel）→ `aten-dispatch` 包装层
- CUDA stream/event 实现 → 外部 CUDA runtime，不在本仓库源码内

## 8. 与相邻子系统交互

- 上游 → 本叶子：前向算子的 Autograd 包装 kernel（aten-dispatch 的 AutogradCPU 等）在执行时构造 Node 并连边。
- 本叶子 → 下游：
  - Node::apply 内部调用 `at::` 算子（aten-native）做梯度计算；
  - ReadyQueue → 线程池 worker；
  - 跨 device 经 stream/event 同步。
- 方向：Python backward → Engine::execute → ReadyQueue → evaluate_function → Node::apply → at 算子。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，不适用 Go/TS 专项。本叶子适配口径：
- **C++ 任务图并行**：这是 PyTorch 的"线程池 + ready queue + InputBuffer 依赖计数"模型——与 Go 的 workqueue 退避不同，这里是"数据就绪驱动"的 DAG 调度，无重试无退避。
- **虚函数 apply**：每个算子的反向是 `Node::apply` 的子类实现；类型擦除经 `variable_list`。
- **TLS 线程局部队列**：`local_ready_queue` 用 `C10_DEFINE_TLS_static` 做 per-thread 状态。
- **Python 桥接**：`python_engine.cpp` 把 C++ Engine 暴露给 Python；GIL 在 Python↔C++ 边界管理。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 反向引擎架构图 | `autograd-engine-architecture.html` | architecture | standard（降档披露：垂直边标签在 showcase 严格布局校验下贴节点，去标签后仍按 standard 渲染成功） |
| backward 执行时序图 | `autograd-engine-sequence.html` | sequence | showcase |

- JSON IR 源文件：`json/` 下对应两个文件。
- 说明：execute→evaluate→apply→下游入队是请求生命周期，适合 sequence；Engine/Node/ReadyQueue 结构用 architecture。
