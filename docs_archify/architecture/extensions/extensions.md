# extensions 域总览（extensions）

> 本域包含 PyTorch 的 **扩展与函数变换能力**：用户自定义算子注册与 C++ 扩展构建（custom-ops），以及函数式变换 vmap/grad 与 AOTAutograd 入口（functorch）。各叶子详情见对应文档。
> 源码基准：PyTorch `main`，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 域职责

extensions 域为两类"把 PyTorch 能力向外/向编译栈延伸"的机制：custom-ops 让用户在统一 Dispatcher 注册自己的算子并用 JIT 构建 C++ 扩展；functorch 把神经网络当作纯函数做向量化/微分变换，并作为 AOTAutograd 入口连接 dynamo/inductor 编译栈。

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序/数据流图 | 职责一句话 |
|------|------|--------|---------------|-----------|
| custom-ops | [custom-ops.md](custom-ops/custom-ops.md) | [架构图](custom-ops/custom-ops-architecture.html) | — | torch.library 算子注册、_custom_op、cpp_extension JIT |
| functorch | [functorch.md](functorch/functorch.md) | [架构图](functorch/functorch-architecture.html) | — | vmap/grad/jvp/vjp、make_fx、AOTAutograd 入口 |

## 3. 域级机制细节

- **统一 Dispatcher 注册**：custom-ops 注册的算子与内置算子同列 `torch.ops.*`；functorch 经 Python key 在同一 dispatcher mode 栈上拦截这些算子。两者共享 torch-api 定义的算子命名空间与分发机制。
- **Python↔C++ 扩展缝**：custom-ops 是用户侧扩展该缝的官方途径；编译产物经 `torch._C` 加载，与内置算子一视同仁。
- **向编译栈过渡**：functorch/AOTAutograd 产出 FX Graph 交给 dynamo-inductor 分片，本域只负责变换与抓图入口，不做代码生成。

## 4. 语言专项适配口径

本项目为 Python 前端 + C++ 核心，不适用 Go/TS 专项。本域是 Python↔C++ 扩展缝与函数式变换缝的交汇：custom-ops 用声明式 DSL + ninja 工具链桥接 C++；functorch 用 dispatcher mode 栈（Python key）实现可组合变换。能力缝：服务定义侧为 `torch.library.*` 与 `torch.func.*`；提供方为本域实现；消费方为 Dispatcher 与下游编译栈。
