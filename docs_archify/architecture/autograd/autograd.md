# autograd 域总览

> 本域包含以下叶子子系统；各叶子详情见对应文档。
> 源码基准：PyTorch（C++ 核心 + Python 前端），commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 域职责

`autograd` 域实现 PyTorch 的 tape-based 自动微分：前向执行时记录可微操作的反向节点，反向时由 C++ 引擎按计算图依赖关系调度执行梯度计算。本域分为 C++ 执行引擎与 Python 用户-facing API 两层。

核心代码路径：
- C++ 引擎与计算图：`torch/csrc/autograd/`（engine/node/edge/graph_task/input_buffer/variable）
- Python API：`torch/autograd/`（function/grad_mode/functional/graph）

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序图 / 数据流图 | 职责一句话 |
|------|------|--------|-------------------|-----------|
| autograd-engine | [autograd-engine.md](autograd-engine/autograd-engine.md) | [架构图](autograd-engine/autograd-engine-architecture.html) | [时序](autograd-engine/autograd-engine-sequence.html) | C++ 反向引擎、Node/ReadyQueue/GraphTask 线程池调度 |
| autograd-python | [autograd-python.md](autograd-python/autograd-python.md) | [架构图](autograd-python/autograd-python-architecture.html) | [数据流](autograd-python/autograd-python-dataflow.html) | Python Function 自定义反向、grad 模式开关、functional 线性求积 |

## 3. 域级机制细节

- **计算图结构**：前向每个可微操作产出一个 `Node`（`torch/csrc/autograd/node.h`），节点经 `Edge`（function+input_nr）连成 DAG；输出张量经 `AutogradMeta.grad_fn` 挂到首个反向节点。
- **反向调度**：`Engine::execute` 构造 `GraphTask`，把根节点 push 到 `ReadyQueue`；worker 线程 `thread_main` 取任务 → `evaluate_function` → `Node::apply` → 把梯度写入下游 `InputBuffer`，依赖计数归零则入队下游。这是"数据就绪驱动"的无锁 DAG 调度。
- **grad 模式**：`no_grad`/`inference_mode`/`set_grad_enabled`（Python）是 C++ `GradMode` TLS 的 RAII 包装；关闭模式下不构造反向节点。
- **C++↔Python 边界**：Python 自定义 `Function` 经 `python_cpp_function.cpp` 桥接到 C++ Node；引擎执行期释放 GIL，回调 Python 时获取。

## 4. 语言专项适配口径（域级）

本域为 **Python 前端 + C++ 核心**，不适用 Go/TS 专项。适配口径：
- C++ 侧：线程池 + ReadyQueue + InputBuffer 依赖计数的任务图调度（非 Go workqueue 退避模型）；TLS per-thread 队列；GraphTask mutex 保护共享态。
- Python 侧：按能力缝分组（Function 自定义反向 / grad_mode 开关 / functional 线性求积）；上下文管理器兼装饰器；GIL 在 Python↔C++ 边界管理。
