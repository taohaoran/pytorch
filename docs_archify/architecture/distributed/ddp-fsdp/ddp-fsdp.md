# DDP 与 FSDP 数据并行（ddp-fsdp）

> 本文是 `distributed` 域下的叶子子系统文档。域级总览见 `../distributed.md`。
> 本文聚焦 **基于 c10d 的高层数据并行策略**：DistributedDataParallel（DDP）与 FullyShardedDataParallel（FSDP）；
> 其底层集合通信抽象见 `../c10d/c10d.md`，RPC/DTensor 见 `../rpc-dtensor/rpc-dtensor.md`。
>
> 源码基准：PyTorch 主分支，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| DDP 封装 | `DistributedDataParallel(nn.Module)` 包装模型，每卡一份完整副本 | `torch/nn/parallel/distributed.py:465` |
| DDP 前向 | `forward` / `_run_ddp_forward` 走 autograd 前向 | `distributed.py:1893`、`:1719` |
| 梯度分桶 | C++ Reducer 把梯度按 bucket 合批，bucket 满即 allreduce，与反向计算重叠 | `torch/csrc/distributed/c10d/reducer.{hpp,cpp}` |
| 通信钩子 | `register_comm_hook` 支持梯度压缩/自定义通信（如 powersgd） | `python_comm_hook.cpp`、`default_comm_hooks.{hpp,cpp}` |
| FSDP 封装 | `FullyShardedDataParallel` 按 rank 分片参数/梯度/优化器状态 | `torch/distributed/fsdp/fully_sharded_data_parallel.py:118` |
| 平展参数 | `FlatParameter` 把同组参数打包成连续张量，便于 all-gather/reduce-scatter | `torch/distributed/fsdp/_flat_param.py` |
| 分片状态字典 | `FullStateDictConfig`/`ShardedStateDictConfig` 决定保存全量或分片权重 | `torch/distributed/fsdp/api.py:293/340` |
| CPU Offload | `CPUOffload` 把分片参数卸载到 CPU | `torch/distributed/fsdp/api.py:230` |
| 分片梯度缩放 | `ShardedGradScaler` 适配 FSDP 分片的 AMP 梯度缩放 | `torch/distributed/fsdp/sharded_grad_scaler.py:46` |
| 自动封装 | `fully_shard`/`wrap` 按策略自动嵌套 FSDP 单元 | `torch/distributed/fsdp/_fully_shard/`、`wrap.py` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `DistributedDataParallel` | `distributed.py:465` | DDP 模块，持有 process group 与梯度 bucket 配置 |
| `Reducer` | `reducer.hpp` | C++ 侧梯度分桶与通信调度，注册 autograd 梯度就绪回调 |
| `FullyShardedDataParallel` | `fully_sharded_data_parallel.py:118` | FSDP 单元，前向 all-gather、反向 reduce-scatter |
| `_FlatParamHandle` / `FlatParameter` | `_flat_param.py` | 平展参数的分片/还原句柄 |
| `CPUOffload` / `StateDictConfig` | `api.py` | FSDP 配置与状态字典策略 |
| `ShardedGradScaler` | `sharded_grad_scaler.py:46` | 继承 `GradScaler`，按分片聚合缩放 |

## 3. 关键调用链

1. **DDP 前向**：用户调用 `model(input)` → `DistributedDataParallel.forward`（`distributed.py:1893`）→ `_run_ddp_forward`（`:1719`）执行包装模块前向，返回输出。
2. **DDP 反向梯度同步**（见时序图）：
   - `loss.backward()` 触发 autograd；每个参数梯度就绪时回调 C++ `Reducer`。
   - Reducer 把梯度累积进 bucket，bucket 满即调用 c10d `allreduce`（`reducer.cpp`）。
   - allreduce 在 NCCL 上完成后 `Work.wait()` 返回，Reducer 把梯度归一化写回参数，供优化器使用。通信与反向计算重叠以掩盖延迟。
3. **FSDP 前向**：进入一个 FSDP 单元 → `FlatParam` 句柄 `all-gather` 从各 rank 取本分片参数拼出完整参数（`_flat_param.py`）→ 执行该层前向 → 用完即释放分片。
4. **FSDP 反向**：梯度到达后 `reduce-scatter` 把梯度按 rank 分片写回，只保留本 rank 的分片梯度，配合 `ShardedGradScaler` 做 AMP。

## 4. 配置项

| 配置 / 构造参数 | 默认 / 行为 | 位置 |
|------|------|------|
| `device_ids` | DDP 绑定 GPU | `distributed.py:465` |
| `bucket_cap_mb` | 梯度 bucket 大小（MB），越大通信越少、内存越多 | `distributed.py` 构造参数 |
| `broadcast_buffers` | 前向是否同步 buffer | DDP 构造参数 |
| `gradient_as_bucket_view` | 梯度视图直接指向 bucket，省一次拷贝 | DDP 构造参数 |
| `sharding_strategy` | FSDP 分片策略（FULL_SHARD / SHARD_GRAD_OP 等） | `fully_sharded_data_parallel.py` |
| `cpu_offload=CPUOffload(...)` | 是否把分片卸载 CPU | `api.py:230` |
| `state_dict_type` | 全量/分片 state dict 策略 | `api.py:293/340` |

## 5. 错误与重试语义

- **通信失败**：底层 c10d/NCCL 错误经 `Work` 上抛；DDP/FSDP 不自行重试，训练进程失败交由 `../elastic` 重启。
- **FSDP 不一致**：world 内各 rank 必须一致地包裹 FSDP 单元，否则 all-gather 死锁；state_dict 前需统一 `state_dict_type`。
- **NCCL 超时**：桶同步阻塞超时会抛错并 `abort()`。

## 6. 并发细节

- **DDP**：反向在每个 rank 进程内单线程 autograd，通信由 Reducer 触发并与反向计算流水重叠；bucket 粒度决定重叠程度。
- **FSDP**：跨 rank 通过 all-gather/reduce-scatter 通信；前向/反向分别有 all-gather 与 reduce-scatter 调度，避免同时持有全量参数。
- 与 c10d 共享 CUDA stream 语义，`Work` 完成后才对梯度可见。

## 7. 系统边界

**In-Scope**
- `torch/nn/parallel/distributed.py`（DDP）
- `torch/distributed/fsdp/`（FSDP 全套）
- `torch/csrc/distributed/c10d/reducer.{hpp,cpp}`（DDP 梯度分桶）

**Out-of-Scope**
- 底层 ProcessGroup/集体通信原语：`../c10d`
- NCCL/Gloo 库本体：外部组件
- RPC/RRef 与 DTensor：`../rpc-dtensor`
- 多进程编排与容错重启：`../elastic`

## 8. 与相邻子系统交互

- **依赖 c10d**：DDP/FSDP 通过 `distributed_c10d` 拿到 ProcessGroup，调用 `allreduce`/`all_gather`/`reduce_scatter`。
- **被训练循环使用**：用户脚本把模型包成 DDP/FSDP 后正常 `forward/backward/step`。
- **下游**：最终落到 NCCL（GPU），见 `../../backends/cuda-backend/cuda-backend.md`。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，不适用 Go/TS 专项。
- Python 侧按"数据并行策略"能力缝组织：DDP 与 FSDP 是两种正交策略，均复用 c10d。
- C++ 侧：DDP 的 Reducer 是关键并发模块，用 autograd 回调 + bucket 合批实现计算/通信重叠；FSDP 主体为 Python，靠 c10d 集合原语完成分片。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|------|------|------|------|
| DDP/FSDP 架构图 | `ddp-fsdp-architecture.html` | architecture | showcase |
| DDP 梯度分桶时序图 | `ddp-fsdp-sequence.html` | sequence | standard（showcase 因多参与者时序段边界约束自动回退） |

JSON IR 源位于 `json/`。

![架构图](ddp-fsdp-architecture.html)

![DDP 梯度同步时序](ddp-fsdp-sequence.html)
