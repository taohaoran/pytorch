# optim（optim）

> 本文是 `python-frontend` 域下的叶子子系统文档。域级总览见 `../python-frontend.md`。
> 本文聚焦 **Optimizer 基类、SGD/Adam 等具体优化器、lr_scheduler、swa_utils**；
> 网络层与参数持有见 `nn-modules`，张量/算子见 `torch-api`。
>
> 源码基准：PyTorch `main`，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| Optimizer 基类 | 持有参数组 `param_groups` 与优化状态 `state`，定义 `step/zero_grad/state_dict` 模板 | `torch/optim/optimizer.py:368` |
| 具体优化器 | SGD、Adam、AdamW、Adamax、RAdam、NAdam、RMSprop、Adagrad、Adadelta、ASGD、Rprop、LBFGS、SparseAdam、Adafactor、Muon | `torch/optim/{sgd,adam,adamw,...}.py` |
| 函数式优化 | `_functional.py` 提供可组合的纯函数式更新（函数式 API） | `torch/optim/_functional.py` |
| 多张量融合 | `_multi_tensor/` 批量多张量融合内核 | `torch/optim/_multi_tensor/` |
| 学习率调度 | LambdaLR/StepLR/MultiStep/Cosine/OneCycle/ReduceLROnPlateau 等 | `torch/optim/lr_scheduler.py` |
| SWA 权重平均 | `swa_utils` 实现随机权重平均（SWA）与模型权重融合 | `torch/optim/swa_utils.py` |
| 无状态优化 | `_stateless.py` 提供无状态优化器变体（函数式训练） | `torch/optim/_stateless.py` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `class Optimizer` | `optimizer.py:368` | 优化器抽象基类，`step()` 为抽象方法由子类实现 |
| `_RequiredParameter` | `optimizer.py:48` | 标记子类必须由用户传入的占位 |
| `Optimizer.step` | `optimizer.py:1127` | 执行一步参数更新，接收可选 closure |
| `Optimizer.zero_grad` | `optimizer.py:1058` | 清空梯度，`set_to_none=True` 置 None |
| `Optimizer.add_param_group` | `optimizer.py:1137` | 向优化器追加参数组（可单独设 lr/weight_decay） |
| `Optimizer.state_dict/load_state_dict` | `optimizer.py:710,910` | 序列化/恢复优化器状态（动量等） |
| `class LRScheduler` | `lr_scheduler.py:95` | 所有调度器基类，包装 optimizer |
| `class ReduceLROnPlateau` | `lr_scheduler.py:1584` | 按指标 Plateau 调整 lr |

## 3. 关键调用链

1. **训练一步**：`loss.backward()` 算好 `.grad` → 用户 `optimizer.step()`（`optimizer.py:1127`）→ 具体优化器（如 `sgd.py`）遍历 `param_groups` 中每个参数，读取 `state[p]` 中的动量/二阶矩 → 按算法原地更新 `p.data` → `optimizer.zero_grad()`（`:1058`）清空梯度。
2. **学习率调度**：`scheduler.step()` → 依据 epoch/step 计算新 lr → 写回 `param_group['lr']`。
3. **断点续训**：`optimizer.state_dict()`（`:710`）导出 param_groups 与 state → 与 `model.state_dict()` 一同 `torch.save`；`load_state_dict`（`:910`）恢复。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `lr`（各优化器） | 学习率，如 Adam 默认 1e-3 | 各优化器 `__init__` |
| `weight_decay` | 权重衰减（L2 正则） | 各优化器 |
| `betas`（Adam） | 一阶/二阶矩衰减系数 (0.9, 0.999) | `adam.py` |
| `zero_grad(set_to_none=True)` | 梯度置 None 而非 0，省显存 | `optimizer.py:1058` |
| `amsgrad`/`foreach`/`capturable` | Adam 行为开关 | `adam.py` |

## 5. 错误与重试语义

- 优化器不做失败重试；参数未参与前向（无 grad）时按 `requires_grad` 与 `grad is None` 跳过。
- `step()` 若 closure 为 LBFGS 等需多次评估的算法，会在内部循环调用 closure。
- `load_state_dict` 对 state 形状/设备不匹配抛出 `RuntimeError`，不静默容错。

## 6. 并发细节

- 优化器在 Python 单线程训练循环中调用；参数原地更新，无锁。
- `_multi_tensor` 融合内核在 C++ 侧批量处理同设备张量，减少 Python 逐参数开销。
- 分布式训练下各 rank 独立优化器，梯度由 DDP 同步（不在本叶子）。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/optim/` 全部：基类、具体算法、lr_scheduler、swa_utils、_functional、_stateless。

**Out-of-Scope（相邻叶子）**
- 参数与 Module 持有 → `nn-modules`。
- 梯度反传（autograd 引擎）→ autograd 相关分片。
- 权重序列化 → `serialization`。
- 分布式梯度同步 → distributed 分片。

## 8. 与相邻子系统交互

- 训练循环 → `Optimizer.step()` ← 读取 `nn.Module.parameters()` 的 `.grad`。
- 优化器原地更新 `param.data`（设备张量，下游由 torch-api/C++ 内核写回）。
- `LRScheduler` 包装同一 `Optimizer`，调整其 `param_group['lr']`。
- `state_dict` ↔ `torch.save/load`（serialization 叶子）。

## 9. 语言专项适配口径

Python 前端 + C++ 核心，不适用 Go/TS 专项。能力缝：服务定义侧为 `Optimizer.step()` 模板方法；提供方为各 `*Optimizer` 子类与 `lr_scheduler`；消费方为用户训练循环。算法的融合内核（foreach/multi-tensor）下沉 C++，Python 仅负责遍历与状态记账。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 优化器结构 | `optim-architecture.html` | architecture | **standard**（showcase 未过 0 错误门槛，自动回退 standard 并如实披露） |

JSON IR 源文件：`json/optim-architecture.json`。训练一步流程已在第 3 节文字化说明，不另出时序图。

![optim 架构图](optim-architecture.html)
