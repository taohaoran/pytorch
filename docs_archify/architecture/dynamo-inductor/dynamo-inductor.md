# dynamo-inductor 域总览

> 本域包含以下叶子子系统；各叶子详情见对应文档。
> 源码基准：pytorch，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 域职责

dynamo-inductor 域覆盖 PyTorch 2 时代的 **torch.compile 编译栈**：在不改变 eager 语义的前提下，拦截 Python 帧执行，符号追踪出 FX 图，经 AOT 与算子分解，最终由 Inductor 后端生成 Triton/C++ 内核。本域解决"eager 训练图如何被自动图编译加速"这一问题，是 PyTorch 2.x 的核心性能路径。

域内分层：
- **追踪层**：Dynamo 字节码追踪与 guard（`torch/_dynamo/`）。
- **后端层**：Inductor IR、融合调度、代码生成（`torch/_inductor/`）。
- **入口与缓存层**：`torch.compile` 门面、配置、磁盘/内核缓存（`torch/_compile.py`、`torch/_inductor/cache.py`）。
- **原语层**：算子分解表与 prims/refs（`torch/_decomp/`、`torch/_prims/`、`torch/_refs/`）。

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序图 / 数据流图 | 职责一句话 |
|------|------|--------|-------------------|-----------|
| dynamo | [dynamo.md](dynamo/dynamo.md) | [架构图](dynamo/dynamo-architecture.html) | [时序图](dynamo/dynamo-sequence.html) | 字节码符号追踪、guard 守卫与图 break |
| inductor | [inductor.md](inductor/inductor.md) | [架构图](inductor/inductor-architecture.html) | — | IR、算子融合调度、Triton/C++ 代码生成 |
| compile-cache | [compile-cache.md](compile-cache/compile-cache.md) | [架构图](compile-cache/compile-cache-architecture.html) | — | torch.compile 入口、配置、多层缓存与文件锁 |
| decomposition | [decomposition.md](decomposition/decomposition.md) | [架构图](decomposition/decomposition-architecture.html) | — | 算子分解表、prims 原语与 refs 参考实现 |

## 3. 域级机制细节

- **torch.compile 全流程**：用户调 `torch.compile(f)(x)` → dynamo eval_frame 钩子 → 字节码符号追踪出 FX 图 → guard 记录输入不变量 → AOTAutograd 分前后/反向 → inductor 降级+融合+codegen → 缓存产物。首次编译慢，后续 guard 命中零开销。
- **guard 失效重编译**：输入形状/类型变化时 guard 求值失败，触发重编译；`cache_size_limit`/`accumulated_recompile_limit` 防止无限重编译。
- **三层缓存**：Guard 缓存（帧→产物）→ FxGraph 缓存（图内容哈希→CompiledFxGraph）→ 磁盘/内核缓存（Triton 内核、C++ 共享库）。FileLock 保证多进程并发写盘安全。
- **算子分解贯穿全栈**：decomposition 表在 dynamo/export/inductor 三处被复用，把高阶算子展开为 prims/核心 aten，降低后端 codegen 复杂度。
- **异步与子进程编译**：inductor 支持后台线程异步编译与子进程隔离 autotune；CUDA Graph 复用进一步降低 launch 开销。

## 4. 域级图

本域未单独产出域级架构图；各叶子已有图（含 dynamo 的 torch.compile 首次编译时序图），系统级架构图由组织者统一汇总。
