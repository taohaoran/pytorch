# 性能分析器（profiler）

> 本文是 `backends` 域下的叶子子系统文档。域级总览见 `../backends.md`。
> 本文聚焦 **torch.profiler 活动记录、内存分析、kineto 桥接与结果导出**；不涉及具体算子实现。
>
> 源码基准：PyTorch 主分支，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `profile` 上下文管理器 | `torch.profiler.profile` 主入口，选择活动类型与选项 | `torch/profiler/profiler.py:1078` |
| 活动配置 | `ProfilerActivityConfig`/`ProfilerActivity` 选择 CPU/CUDA/栈等 | `torch/profiler/profiler.py:201` |
| C++ 采集编排 | 活动事件采集与内存记录 | `torch/csrc/profiler/collection.{h,cpp}`、`orchestration/` |
| Kineto 桥接 | `kineto_shim`/`kineto_client_interface` 把事件喂给 libkineto | `torch/csrc/profiler/kineto_shim.{h,cpp}`、`kineto_client_interface.cpp` |
| 事件类型 | 算子/CUDA kernel/CPU 活动事件定义 | `torch/csrc/profiler/events.h` |
| 栈回溯合并 | 合并 Python/C++ 调用栈 | `torch/csrc/profiler/combined_traceback.{h,cpp}` |
| 内存分析 | 内存分配跟踪 | `torch/profiler/_memory_profiler.py` |
| Python 追踪 | Python 级执行追踪 | `torch/profiler/python_tracer.py` |
| Chrome Trace 导出 | 导出 chrome://tracing JSON | `torch/profiler/_chrome_trace_export.py` |
| Key Averages | 聚合每算子耗时统计 | `torch/profiler/profiler.py` |
| CUDA 快照工具 | cuspy 等辅助 | `torch/profiler/_cuspy/`、`csrc/profiler/cuspy/` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `profile` | `profiler.py:1078` | 主上下文管理器，start/stop/导出 |
| `ProfilerActivity`/`ProfilerConfig` | `profiler.py` | 选择采集的活动维度与配置 |
| `KinetoSession`（C++ 编排） | `csrc/profiler/orchestration/` | 启动/停止采集会话 |
| `kineto_shim` | `kineto_shim.h` | PyTorch 与 libkineto 的适配层 |

## 3. 关键调用链

1. **采集（见数据流图）**：用户 `with profile(activities=[...])` → Python API 调 C++ 编排启动会话 → 在算子 dispatch、CUDA kernel、Python 调用栈打点产生事件 → 事件进入 `collection` 缓冲 → 经 `kineto_shim` flush 到 libkineto。
2. **停止导出**：退出 with → 停止采集 → 合并调用栈 → 导出为 Chrome Trace JSON 与 Key Averages 表。

## 4. 配置项

| 配置 | 行为 | 位置 |
|------|------|------|
| `activities` | 选择 CPU/CUDA/软栈/内存等活动 | `profiler.py:1078` |
| `schedule` | 分阶段（warmup/active/skip）采样 | `profiler.py` |
| `record_shapes`/`with_stack` | 记录输入形状与调用栈 | `profiler.py` |
| `profile_memory` | 是否开启内存分析 | `profiler.py` |

## 5. 错误与重试语义

- 采集为旁路观测，失败不应影响训练；kineto 不可用时回退或报错提示。
- 无重试语义；导出失败抛 Python 异常。

## 6. 并发细节

- 事件在多线程/多 CUDA stream 下采集，按时间戳合并；栈回溯做去重/合并。
- 采集开销通过 schedule 分阶段控制。

## 7. 系统边界

**In-Scope**
- `torch/profiler/`：Python API、内存分析、导出
- `torch/csrc/profiler/`：采集编排、事件、kineto shim、栈回溯

**Out-of-Scope**
- libkineto 本体（外部组件，vendored 或系统库）
- 具体算子/后端实现：见 cuda-backend / vec-kernels / other-accelerators

## 8. 与相邻子系统交互

- **上游**：用户训练循环；dispatch 在算子执行时打点。
- **下游**：把事件交给外部 libkineto 做聚合与导出。
- 与各后端协作采集 CUDA/CPU kernel 时间。

## 9. 语言专项适配口径

本项目为 **Python + C++**，不适用 Go/TS 专项。
- Python API 是上下文管理器；C++ 侧做低开销打点与 kineto 适配。
- libkineto 明确标注为外部组件，本仓库只提供 shim。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|------|------|------|------|
| Profiler 架构图 | `profiler-architecture.html` | architecture | showcase |
| 事件采集数据流图 | `profiler-dataflow.html` | dataflow | standard（showcase 因管道节点布局自动回退） |

JSON IR 源位于 `json/`。

![架构图](profiler-architecture.html)

![事件采集数据流](profiler-dataflow.html)
