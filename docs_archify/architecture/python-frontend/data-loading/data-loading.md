# data-loading（data-loading）

> 本文是 `python-frontend` 域下的叶子子系统文档。域级总览见 `../python-frontend.md`。
> 本文聚焦 **DataLoader、Dataset（Map/Iterable）、Sampler、多进程 worker、collate/pin_memory**；
> 张量/算子见 `torch-api`，模型见 `nn-modules`。
>
> 源码基准：PyTorch `main`，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| DataLoader | 数据加载主类，组合 Dataset/Sampler/worker/collate | `torch/utils/data/dataloader.py:149` |
| 迭代器实现 | `_BaseDataLoaderIter`/单进程/多进程迭代器 | `dataloader.py:85`（`_DatasetKind`） |
| Map-style Dataset | `Dataset` 基类，支持 `__getitem__`/`__len__` | `torch/utils/data/dataset.py:39` |
| Iterable Dataset | `IterableDataset`，流式读取（文件/数据库） | `dataset.py:73` |
| 工具 Dataset | TensorDataset/ConcatDataset/ChainDataset/Subset/StackDataset | `dataset.py:189,299,356,386,212` |
| Sampler 体系 | Sequential/Random/WeightedRandom/Batch/Distributed/SubsetRandom | `torch/utils/data/sampler.py:97,116,213,286,191` |
| 多进程 worker | `_worker_loop` 子进程加载，`WorkerInfo`/`get_worker_info` | `torch/utils/data/_utils/worker.py:244,99` |
| 组批与锁页 | `collate.py` 默认组批、`pin_memory.py` 异步锁页 | `_utils/collate.py`、`_utils/pin_memory.py` |
| 信号/异常包装 | `signal_handling.py`、`_ExceptionWrapper`、`_ResumeIteration` | `_utils/signal_handling.py`、`worker.py:140` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `class DataLoader` | `dataloader.py:149` | 用户入口，`__iter__` 返回对应迭代器 |
| `_DatasetKind` | `dataloader.py:85` | 标记 map/iterable 两种 dataset 形态 |
| `class Dataset` | `dataset.py:39` | Map-style 基类 |
| `class IterableDataset` | `dataset.py:73` | 流式基类 |
| `class Sampler` / `BatchSampler` | `sampler.py:28,286` | 采样索引 / 把索引打包成 batch |
| `_worker_loop` | `worker.py:244` | 子进程主循环：取索引→读 dataset→collate→入队 |
| `get_worker_info()` | `worker.py:99` | worker 内获取本 worker 元信息（id/num_workers/seed） |
| `_IterableDatasetStopIteration` / `_ResumeIteration` | `worker.py:132,140` | 协调 iterable 数据集多 worker 终止/恢复的消息 |

## 3. 关键调用链

1. **单 batch 加载（num_workers>0）**：`for batch in dl` → `DataLoader.__iter__` 建 `_MultiProcessingDataLoaderIter` → 启动 N 个 `_worker_loop` 子进程（`worker.py:244`）→ 子进程按 sampler 取索引、`dataset[idx]` 读样本、`collate_fn` 组批 → 放入工作队列 → 主迭代器从队列取 batch、可选 `pin_memory` → `yield` 给训练循环（见时序图）。
2. **迭代结束**：主迭代器向 worker 发 `_IterableDatasetStopIteration`（`worker.py:132`）→ worker 退出 → 析构时 join 子进程、关队列（`_persistent_workers_atexit` 兜底，`dataloader.py:78-80`）。
3. **num_workers=0**：直接在主进程迭代 `_SingleProcessDataLoaderIter`，无 fork、无队列。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `batch_size` | 每批样本数，默认 1 | `dataloader.py:255` |
| `shuffle` | 每个 epoch 重排 | `dataloader.py` |
| `num_workers` | 子进程数，默认 0（主进程） | `dataloader.py:262` |
| `prefetch_factor` | 每个 worker 预取 batch 数，num_workers>0 默认 2 | `dataloader.py:201` |
| `persistent_workers` | 跨 epoch 保活子进程，默认 False | `dataloader.py:272` |
| `pin_memory` | 自动把张量锁页到 CUDA 可加速拷贝 | `dataloader.py` |
| `drop_last` | 丢弃末尾不完整 batch | `dataloader.py` |
| `worker_init_fn` | 子进程初始化钩子 | `dataloader.py:189` |

## 5. 错误与重试语义

- worker 内异常被 `_ExceptionWrapper` 包装后通过队列传回主进程，在主迭代 `next()` 时重新抛出，不会静默丢失。
- iterable 数据集 worker 终止用 `_IterableDatasetStopIteration` 控制；`_ResumeIteration`（`worker.py:140`）支持断点恢复。
- 不做自动重试；数据集损坏/读出错误直接中断训练。

## 6. 并发细节

- **多进程模型**：`num_workers>0` 时按 start method（fork/spawn）启动子进程；主进程与 worker 之间通过 `multiprocessing.Queue` 传递 batch，prefetch 形成流水线重叠计算与 IO。
- **RNG 隔离**：每个 worker 用 `_generate_state`（`worker.py:192`）派生独立随机种子，保证数据增广确定性。
- **pin_memory**：锁页拷贝在独立线程/CUDA 侧异步进行，与主计算重叠。
- **GIL**：主进程迭代在 GIL 下；子进程为独立解释器，不共享内存（fork 只读 copy-on-write）。
- **生命周期**：`persistent_workers=False` 每个 epoch 重建子进程；`True` 保活。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/utils/data/`：DataLoader、dataset.py、sampler.py、`_utils/`（worker/collate/pin_memory/fetch/signal_handling）、`graph.py` 等。

**Out-of-Scope（相邻叶子）**
- 分布式采样 `DistributedSampler` 属于 sampler.py 但与 distributed 集合通信配合，集合通信部分不在本叶子。
- 数据集内容/IO（如读取图片解码）由用户 Dataset 实现，不在本框架。
- `datapipes/` 为独立数据管道工具，本文仅列目录不深入。

## 8. 与相邻子系统交互

- 训练循环 → `DataLoader` → 产出 batch → 喂给 `nn.Module`（nn-modules 叶子）前向。
- `DataLoader` 内部使用 `Tensor`（torch-api）作为 batch 元素类型。
- 锁页张量 → 异步 `cuda()` 拷贝到设备（device-context / cuda-backend 下游）。

## 9. 语言专项适配口径

Python 前端 + C++ 核心，不适用 Go/TS 专项。本叶子是 PyTorch 中**并发模型最重**的 Python 子系统：采用多进程（非多线程）绕过 GIL 并行数据加载，主-从进程经队列通信。能力缝：服务定义侧为 `Dataset.__getitem__`/`__iter__` 与 `Sampler.__iter__` 协议；提供方为 DataLoader 迭代器与 worker 主循环；消费方为训练循环。C++ 边界仅在 pin_memory/异步拷贝处出现。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 组件结构 | `data-loading-architecture.html` | architecture | **standard**（showcase 未过 0 错误门槛，自动回退并如实披露） |
| 多进程加载流水线 | `data-loading-sequence.html` | sequence | **standard**（同上） |

JSON IR 源文件：`json/data-loading-architecture.json`、`json/data-loading-sequence.json`。

![data-loading 架构图](data-loading-architecture.html)

![数据加载时序图](data-loading-sequence.html)
