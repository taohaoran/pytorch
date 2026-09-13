# distributed 分布式训练域（distributed）总览

> 本域包含以下叶子子系统；各叶子详情见对应文档。
> 源码基准：PyTorch 主分支，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 域职责

`distributed` 域提供 PyTorch 的**分布式训练栈**，自底向上分四层抽象，四层职责互不重叠：

1. **通信原语层（c10d）**：ProcessGroup 抽象、NCCL/Gloo/MPI/UCC 后端、Store 键值协调、Work 异步句柄。
2. **数据并行策略层（ddp-fsdp）**：基于 c10d 的 DDP（梯度同步）与 FSDP（参数分片）。
3. **通用分布式抽象层（rpc-dtensor）**：RPC 远程调用/RRef，以及 DTensor/DeviceMesh/Placement 分布式张量。
4. **进程编排层（elastic）**：torchrun 启动、agent 多进程管理、rendezvous 集合点与容错重启。

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序 / 数据流 | 职责一句话 |
|------|------|--------|-------------------|-----------|
| c10d | [c10d.md](c10d/c10d.md) | [架构图](c10d/c10d-architecture.html) | — | ProcessGroup 与可插拔集合通信原语 |
| ddp-fsdp | [ddp-fsdp.md](ddp-fsdp/ddp-fsdp.md) | [架构图](ddp-fsdp/ddp-fsdp-architecture.html) | [时序图](ddp-fsdp/ddp-fsdp-sequence.html) | DDP 梯度同步与 FSDP 参数分片 |
| rpc-dtensor | [rpc-dtensor.md](rpc-dtensor/rpc-dtensor.md) | [架构图](rpc-dtensor/rpc-dtensor-architecture.html) | — | RPC/RRef 与 DTensor 分布式张量 |
| elastic | [elastic.md](elastic/elastic.md) | [架构图](elastic/elastic-architecture.html) | [时序图](elastic/elastic-sequence.html) | torchrun 启动与容错重启 |

## 3. 域级机制细节

- **分层调用方向**：elastic 拉起 worker → worker 内 `init_process_group` 经 rendezvous 建 c10d ProcessGroup → DDP/FSDP/RPC/DTensor 复用 ProcessGroup 做集合通信。
- **rendezvous 是关键交汇点**：elastic 与 c10d 都消费 rendezvous 产生的 Store/rank/world_size。
- **后端可插拔**：c10d 把 NCCL/Gloo/MPI/UCC 多态化，GPU 训练默认 NCCL（依赖 `../backends/cuda-backend/cuda-backend.md` 的 CUDA 流）。
- **容错边界**：只有 elastic 负责进程级重启；上层通信失败一律上抛，由 elastic 重启重会合。

## 4. 语言专项适配口径（域级披露）

本项目为 **Python 前端 + C++ 核心**，不适用 Go（goroutine/informer/Reconciler）或 TypeScript（capability seam）专项。本域按 Python+C++ 口径分析：C++ 侧为 c10d 通信库、RPC agent、Reducer（多态继承 + intrusive_ptr 生命周期 + CUDA stream 并发）；Python 侧按"通信原语 / 数据并行策略 / RPC 与张量抽象 / 进程编排"能力缝组织。

## 5. 质量档汇总

| 叶子 | 架构图 | 第二图 |
|------|--------|--------|
| c10d | showcase | — |
| ddp-fsdp | showcase | 时序 standard |
| rpc-dtensor | showcase | — |
| elastic | standard | 时序 standard |
