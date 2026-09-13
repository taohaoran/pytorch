# python-frontend 域总览（python-frontend）

> 本域包含 PyTorch 的 **Python 用户前端层**：从 `import torch` 入口、张量/算子面，到神经网络模块、优化器、数据加载、序列化与设备/流上下文。各叶子详情见对应文档。
> 源码基准：PyTorch `main`，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 域职责

python-frontend 域是用户与 PyTorch 交互的最上层 Python 面：负责包初始化、Tensor/算子 API 暴露、神经网络组合（Module）、训练算法（optim）、数据流水线（DataLoader）、权重持久化（save/load）以及设备/流/autocast 上下文。它把用户调用经 pybind11 下沉到 C++ 核心（libtorch），本域所有叶子均不实现设备算子内核本身。

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序/数据流图 | 职责一句话 |
|------|------|--------|---------------|-----------|
| torch-api | [torch-api.md](torch-api/torch-api.md) | [架构图](torch-api/torch-api-architecture.html) | — | 包入口、Tensor 类、算子命名空间与 C++ 绑定面 |
| nn-modules | [nn-modules.md](nn-modules/nn-modules.md) | [架构图](nn-modules/nn-modules-architecture.html) | — | Module 层级、Parameter、hooks 与 functional 算子 |
| optim | [optim.md](optim/optim.md) | [架构图](optim/optim-architecture.html) | — | Optimizer 基类、SGD/Adam 与 lr_scheduler |
| data-loading | [data-loading.md](data-loading/data-loading.md) | [架构图](data-loading/data-loading-architecture.html) | [时序图](data-loading/data-loading-sequence.html) | DataLoader/Dataset/Sampler 多进程数据流水线 |
| serialization | [serialization.md](serialization/serialization.md) | [架构图](serialization/serialization-architecture.html) | — | save/load、zipfile 格式、weights_only |
| device-context | [device-context.md](device-context/device-context.md) | [架构图](device-context/device-context-architecture.html) | — | 设备/流/事件与 autocast 上下文 |

## 3. 域级机制细节

- **Python↔C++ 边界**：所有叶子最终经 `torch._C`（pybind11）下沉；torch-api 定义该边界，其余叶子（nn/optim/data/serialization/device-context）消费它。
- **dispatcher mode 栈**：device-context 的 autocast、extensions 域的 functorch 都挂在 `_ops.py` 的 dispatch mode 栈上，本域叶子是这些模式的消费者而非定义者。
- **并发模型**：data-loading 是本域唯一以多进程绕过 GIL 的子系统；其余 Python 层单线程，真正设备并发在 C++ 侧。
- **训练闭环**：DataLoader 产出 batch → Module 前向 → loss → autograd 反传 → Optimizer 更新参数 → serialization 持久化，构成本域叶子间的主线数据流。

## 4. 语言专项适配口径

本项目为 Python 前端 + C++ 核心，不适用 Go（goroutine/informer）或 TS（workspace）专项。本域按能力缝归组：服务定义侧为各 Python 公共 API；提供方为 Python 实现 + `torch._C` pybind 绑定；消费方为用户脚本。C++ 设备内核/内存分配不在本域（见 cuda-backend 分片）。
