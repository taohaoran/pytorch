# device-context（device-context）

> 本文是 `python-frontend` 域下的叶子子系统文档。域级总览见 `../python-frontend.md`。
> 本文聚焦 **设备/流(Stream)/事件(Event)上下文、autocast 自动混合精度、设备管理**；
> CUDA 算子内核与 CUDACachingAllocator 见 cuda-backend 分片，不重复。
>
> 源码基准：PyTorch `main`，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| Stream/Event 公共基类 | `_StreamBase`/`_EventBase` 为各后端流/事件提供 Python 基类 | `torch/_streambase.py:11,19` |
| 流上下文切换 | `StreamContext`/`stream()` 把后续算子排队到指定流 | `torch/cuda/__init__.py:879,937` |
| 当前流查询 | `current_stream(device)` 返回当前设备流 | `torch/cuda/__init__.py:1358` |
| autocast 自动混合精度 | `autocast` 上下文按算子策略选择 FP16/BF16 dtype | `torch/amp/autocast_mode.py:52` |
| GradScaler | 缩放梯度防 FP16 梯度下溢 | `torch/amp/grad_scaler.py` |
| 设备管理 | CUDA 设备计数/当前设备/同步 | `torch/cuda/__init__.py` |
| C++ Stream/Event 绑定 | `torch._C.Stream`/`Event`，C++ 侧持有 cudaStream_t/事件 | `torch/csrc/Stream.{h,cpp}` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `class _StreamBase(torch.Stream)` | `_streambase.py:11` | 流句柄的 Python 基类，提供 wait_event/record_event 等 |
| `class _EventBase(torch.Event)` | `_streambase.py:19` | 事件基类，用于流间同步 |
| `class StreamContext` | `torch/cuda/__init__.py:879` | `with stream(s):` 上下文管理器 |
| `def stream(...)` | `torch/cuda/__init__.py:937` | 返回流上下文管理器 |
| `def current_stream(...)` | `torch/cuda/__init__.py:1358` | 查询当前流 |
| `class autocast` | `amp/autocast_mode.py:52` | 自动混合精度上下文 |
| `class GradScaler` | `amp/grad_scaler.py` | FP16 训练梯度缩放 |

## 3. 关键调用链

1. **切换流**：`with torch.cuda.stream(s):` → `stream()`（`__init__.py:937`）返回 `StreamContext`（`:879`）→ `__enter__` 把 `s` 设为当前流（`current_stream`，`:1358`）→ 块内算子排队到 `s` → `__exit__` 恢复原流。
2. **autocast**：`with torch.autocast(device_type, dtype):` → `autocast`（`autocast_mode.py:52`）进入 C++ autocast mode → 算子分发时按白名单把输入转 FP16/BF16，非白名单算子保持原 dtype。
3. **流间同步**：`event.record()` → `stream.wait_event(event)` 让下游流等待上游事件完成，底层经 C++ `Stream.cpp` 调 CUDA 运行时。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `autocast(dtype)` | 默认 dtype（CUDA 为 float16） | `autocast_mode.py:52` |
| `enabled` | autocast 是否启用 | autocast |
| GradScaler `enabled` | 是否启用梯度缩放 | `grad_scaler.py` |
| 设备索引 | `torch.cuda.set_device(i)` | `torch/cuda/__init__.py` |

## 5. 错误与重试语义

- 流/事件操作失败（如非法设备）由 C++ 抛 `RuntimeError`，Python 不重试。
- autocast 上下文异常退出时保证恢复原 dtype/mode（`__exit__` 复位）。
- GradScaler 在 inf/nan 梯度时自动跳过该步并下调 scale factor。

## 6. 并发细节

- **CUDA 流是设备侧异步执行队列**；Python 线程内 `current_stream` 是线程局部，多线程可在不同流上排队而不互相阻塞。
- `Event.record`/`wait_event` 实现跨流同步，避免主机端同步。
- autocast mode 为线程局部栈，不跨线程泄漏。
- GIL 在算子提交流期间释放，允许其他线程与设备并行。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/_streambase.py`、`torch/cuda/__init__.py`（流/事件/设备管理部分）、`torch/amp/`、`torch/csrc/Stream.{h,cpp}`。

**Out-of-Scope（相邻叶子）**
- CUDA 内存分配器 CUDACachingAllocator 与算子内核 → cuda-backend 分片。
- CUDA Graph（`torch.cuda.CUDAGraph`）相关不在本叶子展开。
- 多设备/分布式通信 → distributed 分片。

## 8. 与相邻子系统交互

- 用户 → `with stream/autocast` → 影响后续所有算子（torch-api 的 Tensor 算子分发）。
- 流/事件上下文 → C++ Dispatcher 选择目标流（下游设备内核，见 cuda-backend）。
- autocast → 在算子分发前插入 dtype 转换（与 functorch/dynamo 的 mode 栈并行存在）。

## 9. 语言专项适配口径

Python 前端 + C++ 核心，不适用 Go/TS 专项。能力缝：服务定义侧为 `torch.cuda.stream`/`autocast` 上下文协议；提供方为 `_streambase` 基类与 C++ Stream/Event 绑定；消费方为所有算子。本叶子是 PyTorch **设备并发语义**的 Python 暴露面，真正的流/事件对象与 CUDA 调用在 C++ 侧。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 设备/流/autocast 上下文结构 | `device-context-architecture.html` | architecture | **standard**（showcase 因横向标签间距未过 0 错误门槛，自动回退并如实披露） |

JSON IR 源文件：`json/device-context-architecture.json`。流切换/autocast 流程已在第 3 节文字化说明，不另出时序图。

![device-context 架构图](device-context-architecture.html)
