# 弹性训练启动与容错（elastic）

> 本文是 `distributed` 域下的叶子子系统文档。域级总览见 `../distributed.md`。
> 本文聚焦 **torchrun 入口、agent 多进程管理、rendezvous 集合点、events/metrics 与容错重启**；
> 启动后的进程间集合通信见 `../c10d/c10d.md`。
>
> 源码基准：PyTorch 主分支，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| torchrun 入口 | `torchrun`/`torchrunx` 命令拉起分布式训练脚本 | `torch/distributed/run.py`、`launch.py` |
| Launcher | 解析命令行参数、构造 agent 并运行 | `torch/distributed/elastic/` |
| LocalElasticAgent | 本地多进程 agent，管理 worker 生命周期 | `torch/distributed/elastic/agent/server/local_elastic_agent.py:118` |
| 进程监控/重启 | 监控 worker 退出码，失败后重新 rendezvous 并重启 | `local_elastic_agent.py`（`SimpleElasticAgent` 父类） |
| RendezvousHandler | 集合点抽象，决定 world_size/rank/store | `torch/distributed/elastic/rendezvous/api.py` |
| etcd 后端 | 基于 etcd 的动态 rendezvous | `rendezvous/etcd_rendezvous.py:91`、`etcd_store.py` |
| static/tcp 后端 | 静态地址 rendezvous | `rendezvous/static_tcp_rendezvous.py:31` |
| c10d 后端 | 内置 c10d rendezvous 后端 | `rendezvous/c10d_rendezvous_backend.py` |
| rendezvous 注册表 | 按 url 方案注册 handler | `rendezvous/registry.py` |
| Events | 生命周期事件记录与 handlers | `elastic/events/api.py`、`handlers.py` |
| Metrics | agent 指标采集 | `elastic/metrics/api.py` |
| 健康检查 | worker 健康检查 server | `agent/server/health_check_server.py` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `LocalElasticAgent` | `local_elastic_agent.py:118` | 本地 worker 进程拉起、监控、重启 |
| `RendezvousHandler` | `rendezvous/api.py` | `next_rendezvous` 协调 rank 分配 |
| `EtcdRendezvousHandler` | `etcd_rendezvous.py:91` | etcd 动态后端 |
| `StaticTCPRendezvous` | `static_tcp_rendezvous.py:31` | 静态后端 |
| `EventHandler` | `events/handlers.py` | 事件消费（写文件/日志） |

## 3. 关键调用链

1. **首次启动（见时序图）**：`torchrun` 解析参数 → Launcher 构造并启动 `LocalElasticAgent` → agent 调 `RendezvousHandler.next_rendezvous` → etcd/static 后端决定 world_size 与各 rank → 返回 store/rank → agent fork 出 `world_size` 个 worker 子进程并注入 `MASTER_ADDR/PORT/RANK/WORLD_SIZE/LOCAL_RANK` 等环境变量。
2. **运行监控**：agent 阻塞等待 worker 进程退出；正常退出收集退出码，异常退出进入重启分支。
3. **故障重启**：worker 失败 → agent 重新 `next_rendezvous`（动态调整 world_size）→ 拉起新一批 worker → 通过 events/metrics 上报。

## 4. 配置项

| 配置 / 环境变量 | 默认 / 行为 | 位置 |
|------|------|------|
| `--nnode` / `--nproc_per_node` | 节点数 / 每节点进程数 | `run.py` |
| `--rdzv_backend` | etcd/static/c10d | `run.py` |
| `--rdzv_endpoint` | rendezvous 地址 | `run.py` |
| `--max_restarts` | agent 最大重启次数 | elastic 配置 |
| `MASTER_ADDR/PORT/RANK/WORLD_SIZE/LOCAL_RANK` | 注入 worker 环境变量 | agent |
| `TORCHELASTIC_ERROR_FILE` | 错误信息落盘位置 | elastic |

## 5. 错误与重试语义

- **worker 失败**：agent 根据退出码与 `restart_count` 决定重启还是放弃；超过 `max_restarts` 则整体失败。
- **rendezvous 超时**：`next_rendezvous` 等待对端超时抛错，agent 据此重启。
- **错误落盘**：失败信息写入 `TORCHELASTIC_ERROR_FILE`，便于排查。
- 这是整个 distributed 域**唯一负责进程级重试/重启**的层；c10d/DDP 失败向上抛，由本层重启。

## 6. 并发细节

- **多进程模型**：agent 在父进程 fork/监控 worker 子进程；每个 worker 是独立 rank。
- **etcd 协调**：rendezvous 通过 etcd 键值做跨节点协调；static 后端无外部依赖。
- **事件/指标**：agent 通过 events/metrics 旁路上报，不阻塞训练主路径。

## 7. 系统边界

**In-Scope**
- `torch/distributed/elastic/`：torchrun、agent、rendezvous、events、metrics
- `torch/distributed/run.py`、`launch.py`：命令入口

**Out-of-Scope**
- etcd 服务本体（外部组件）
- 训练进程内的集合通信：`../c10d`
- 训练逻辑本身（用户脚本）

## 8. 与相邻子系统交互

- **上游**：用户命令行 `torchrun ...`。
- **下游**：拉起的 worker 进程内执行用户脚本，调用 `init_process_group`（`../c10d`）完成 rendezvous 后的进程组建立。
- **协调**：rendezvous 后端对接外部 etcd 或 TCP 主节点。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，不适用 Go/TS 专项。
- Python 侧按"启动编排 / 集合点 / 事件指标"能力缝组织，纯 Python 实现，依赖 `multiprocessing`。
- 本层不涉及 CUDA/算子，仅负责进程编排与容错。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|------|------|------|------|
| Elastic 架构图 | `elastic-architecture.html` | architecture | standard（showcase 因侧连虚线路由自动回退） |
| torchrun 启动/重启时序图 | `elastic-sequence.html` | sequence | standard（showcase 因多参与者时序段约束自动回退） |

JSON IR 源位于 `json/`。

![架构图](elastic-architecture.html)

![torchrun 启动时序](elastic-sequence.html)
