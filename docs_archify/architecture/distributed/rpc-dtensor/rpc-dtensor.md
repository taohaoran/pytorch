# RPC 与分布式张量（rpc-dtensor）

> 本文是 `distributed` 域下的叶子子系统文档。域级总览见 `../distributed.md`。
> 本文聚焦 **RPC 远程调用框架（RpcAgent/RRef）与 DTensor 分布式张量（DeviceMesh/Placement）**；
> 底层集合通信见 `../c10d/c10d.md`，数据并行策略见 `../ddp-fsdp/ddp-fsdp.md`。
>
> 源码基准：PyTorch 主分支，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| RPC 入口 | `rpc_sync`/`rpc_async`/`remote` 发起跨进程函数调用/远程创建 | `torch/distributed/rpc/api.py:772/846/555` |
| RPC 启停 | `init_rpc`/`shutdown` 初始化 worker 与关闭 | `torch/distributed/rpc/api.py:327` |
| RpcAgent 抽象 | C++ 抽象 agent，管理收发消息与 worker 表 | `torch/csrc/distributed/rpc/rpc_agent.h` |
| WorkerInfo | worker 名/标识，定位对端 | `torch/distributed/rpc/api.py:32`（C++ 绑定） |
| RRef 远程引用 | 跨进程远程对象引用，引用计数管理 | `torch/csrc/distributed/rpc/rref_impl.h`、`rref_context.cpp` |
| Python 调用处理 | 服务端接收 python_call 并在本地执行 | `torch/csrc/distributed/rpc/python_call.cpp`、`python_rpc_handler.cpp` |
| 传输后端 | TensorPipe（默认），进程间点对点传输 | `torch/distributed/rpc/options.py:50` |
| DTensor | 分布式张量 `DTensor(Tensor)`，携带分片 Spec | `torch/distributed/tensor/_api.py:356` |
| DeviceMesh | 描述设备/进程拓扑的 N 维网格 | `torch/distributed/device_mesh.py:188` |
| Placement | `Shard`/`Replicate` 描述某维度如何分片/复制 | `torch/distributed/tensor/placement_types.py:162/1700` |
| 分片传播 | 算子前向时推导输入/输出分片（ShardingPropagator） | `torch/distributed/tensor/_sharding_prop.py:362` |
| 重分布 | DTensorRedistributePlanner 决定 all-gather/重分片 | `torch/distributed/tensor/_redistribute.py:680` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `rpc_sync`/`rpc_async`/`remote` | `rpc/api.py` | 同步/异步/远程创建 RPC 原语 |
| `RpcAgent` | `rpc/rpc_agent.h` | 消息收发抽象，`send/recv`、后端可替换 |
| `RRef`/`RRefContext` | `rref_impl.h`、`rref_context.cpp` | 远程对象引用及其生命周期管理 |
| `DTensor` | `_api.py:356` | 分布式张量子类，持 `DTensorSpec` |
| `DTensorSpec` | `_dtensor_spec.py:78` | 记录 DeviceMesh 与各维 Placement |
| `DeviceMesh` | `device_mesh.py:188` | 设备拓扑 |
| `ShardingPropagator` | `_sharding_prop.py:362` | 算子分片推导 |
| `TensorPipeRpcBackendOptions` | `options.py:50` | TensorPipe 传输配置 |

## 3. 关键调用链

1. **RPC 异步调用**：`rpc_async(to, func, args)`（`api.py:846`）→ 解析 `WorkerInfo` → 当前 `RpcAgent` 把调用序列化为 `Message`（`message.cpp`）→ 经 TensorPipe 发到对端 → 对端 `python_rpc_handler.cpp` 反序列化并 `python_call.cpp` 执行 `func` → 结果沿原路返回，本地用 Future 等待。
2. **RRef**：`remote(to, ...)` 创建远程对象，返回本地 RRef 句柄；`RRefContext`（`rref_context.cpp`）在 owner 与持有者间维护引用计数，析构时通知 owner 回收。
3. **DTensor 前向**：用户在 `DeviceMesh` 上创建 DTensor（含 `DTensorSpec`）→ 调用算子 → `ShardingPropagator`（`_sharding_prop.py:362`）根据输入 Placement 推导输出 Placement → 若需要不同分布，`DTensorRedistributePlanner`（`_redistribute.py:680`）经 c10d 做 all-gather/重分片 → 结果仍是携带新 Spec 的 DTensor。

## 4. 配置项

| 配置 / 环境变量 | 默认 / 行为 | 位置 |
|------|------|------|
| `RpcBackendOptions` | 默认 TensorPipe；`num_worker_threads` 线程数 | `options.py:50` |
| `UNSET_RPC_TIMEOUT` | RPC 调用超时 | `rpc/api.py` |
| `device_mesh` 后端 | 由 `init_device_mesh`/DeviceMesh 工厂创建 | `device_mesh.py:188` |
| `_tensor/dim_map` | DTensor 分片维度映射 | `_dtensor_spec.py` |

## 5. 错误与重试语义

- **RPC 失败**：远端异常回传为本地异常；超时由 `timeout` 控制；RPC 层不自动重试。
- **RRef 生命周期**：owner 退出时未释放 RRef 会泄漏；`shutdown(graceful=True)`（`api.py:327`）优雅关闭等待 in-flight 调用。
- **DTensor 重分布**：ShardingPropagator 无法推导时抛分片不支持错误；死锁由 c10d 超时兜底。

## 6. 并发细节

- **RPC**：RpcAgent 用收发线程池处理并发 in-flight 调用；对端每个 worker 有独立处理队列。
- **DTensor**：在每个 rank 进程内本地张量运算，跨 rank 仅在重分布时经 c10d 通信，计算与通信分离。

## 7. 系统边界

**In-Scope**
- `torch/distributed/rpc/` 与 `torch/csrc/distributed/rpc/`：RPC 框架
- `torch/distributed/tensor/`、`torch/distributed/device_mesh.py`：DTensor/DeviceMesh/Placement

**Out-of-Scope**
- TensorPipe 传输库本体（外部组件）
- c10d 集合通信原语：`../c10d`
- DDP/FSDP 数据并行：`../ddp-fsdp`；torchrun 编排：`../elastic`

## 8. 与相邻子系统交互

- **DTensor → c10d**：重分布经 `distributed_c10d` 的 all-gather/all-reduce。
- **RPC → TensorPipe**：点对点消息走外部 TensorPipe；也可基于 ProcessGroup。
- **上层**：FSDP/并行层、流水线并行可复用 DTensor 与 RPC。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，不适用 Go/TS 专项。
- Python 侧按"RPC 调用原语 / 分布式张量抽象"能力缝组织。
- C++ 侧：RpcAgent 是消息收发抽象基类，RRef 用引用计数管理跨进程对象生命周期；DTensor 分片传播在 Python 用 dispatch mode 实现。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|------|------|------|------|
| RPC 与 DTensor 架构图 | `rpc-dtensor-architecture.html` | architecture | showcase |

JSON IR 源位于 `json/`。RPC 消息往返时序已在调用链文字化描述，未单独出时序图（避免与 RPC 调用链重复）。

![架构图](rpc-dtensor-architecture.html)
