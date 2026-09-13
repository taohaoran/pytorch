# functorch（functorch）

> 本文是 `extensions` 域下的叶子子系统文档。域级总览见 `../extensions.md`。
> 本文聚焦 **vmap、grad/jvp/vjp、make_fx、函数变换组合器、AOTAutograd 入口**；
> dynamo 字节码追踪与 inductor 代码生成见 dynamo-inductor 分片，不重复。
>
> 源码基准：PyTorch `main`，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `vmap` API | 向量化映射批量维度，无需手写 batch 维 | `torch/_functorch/apis.py:68`、`vmap.py:286` |
| `grad` / `grad_and_value` | 函数式自动微分 | `apis.py:356,469`、`eager_transforms.py:1534` |
| `jvp` / `vjp` | 前向/反向模式雅可比向量积 | `eager_transforms.py:1094,239` |
| `make_fx` | 把可调用抓成 FX Graph | `torch/fx/`（消费侧）、`_functorch/` 集成 |
| `functional_call` / `make_functional` | 把 Module 转为纯函数（参数作输入） | `_functorch/functional_call.py`、`make_functional.py` |
| AOTAutograd | 提前追踪前向+反向，分离图 | `_functorch/aot_autograd.py`、`_aot_autograd/` |
| 变换组合 | vmap/grad 以 dispatcher mode 栈叠加 | `python_key.py`、`eager_transforms.py` |
| 分区/编译 | `partitioners.py`、`compilers.py` 供后端选择 | `_functorch/partitioners.py`、`compilers.py` |
| 旧包 functorch | 兼容包（现并入 torch） | `functorch/`（shim/文档/示例） |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `def vmap` | `apis.py:68` | 向量化组合器入口 |
| `vmap_impl` | `vmap.py:286` | vmap 实际实现（BatchedTensor + Python key） |
| `def grad` / `grad_and_value` | `apis.py:356,469` | 梯度组合器 |
| `grad_impl` / `jvp` / `vjp` | `eager_transforms.py:1534,1094,239` | eager 变换内核 |
| `grad_increment_nesting` / `jvp_increment_nesting` | `eager_transforms.py:358,380` | 嵌套变换的层数计数器 |
| `aot_autograd` | `aot_autograd.py` | AOTAutograd 入口，产出 forward/backward 图 |
| `python_key` | `python_key.py` | dispatcher Python key，承载 BatchedTensor 变换语义 |

## 3. 关键调用链

1. **组合 vmap(grad(f))(x)**：`vmap(grad(f))`（`apis.py:68,356`）→ 各变换以 dispatcher mode 栈入栈 → 每次算子调用经 pre-dispatch 栈（见 torch-api `_ops.py`）落到 Python key → `vmap_impl`（`vmap.py:286`）把批量维打包为 BatchedTensor → `grad_impl`（`eager_transforms.py:1534`）记录反向 → 组合出向量化梯度函数。
2. **AOTAutograd**：`aot_autograd`（`aot_autograd.py`）→ 用 make_fx 抓前向 → 跑一次反向得到 backward 图 → `partitioners.py` 把前反向分区 → 交给后端（如 inductor）编译。
3. **嵌套变换**：`grad_increment_nesting`（`eager_transforms.py:358`）维护嵌套层数，保证多个变换正确叠层。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `randomness`（vmap） | 采样性：error/different/same | `apis.py:68` |
| `chunk_size` | vmap 分块大小（省内存） | `vmap.py` |
| `has_aux`（grad） | 是否附带辅助输出 | `eager_transforms.py` |
| AOTAutograd 配置 |  tracing/fake tensor 开关 | `config.py` |

## 5. 错误与重试语义

- 变换叠加非法（如对非张量 grad）直接抛 `RuntimeError`，不重试。
- 旧 API（`deprecated.py`）触发 DeprecationWarning，引导迁移到 `torch.func.*`。
- AOTAutograd 抓图失败会回退到 eager 并抛出具体算子错误。

## 6. 并发细节

- 变换本身是纯函数式包装，无内部线程；运行在 dispatcher mode 栈上。
- BatchedTensor 在 Python key 下维护批量维元数据，真正并行计算下沉设备内核。
- AOTAutograd 抓图在单线程 tracing 阶段完成，产出的图可被后端异步编译。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/_functorch/` 全部（apis、vmap、eager_transforms、aot_autograd、functional_call、python_key 等）。

**Out-of-Scope（相邻叶子）**
- dynamo 字节码追踪/guard、inductor Triton 代码生成 → dynamo-inductor 分片。
- FX Graph 本身的定义与解释器 → compile-graph/fx 部分。
- autograd 引擎内核 → autograd 相关分片。

## 8. 与相邻子系统交互

- 用户纯函数 → functorch 变换 → 经 dispatcher Python key（torch-api `_ops.py` mode 栈）拦截算子。
- functorch/AOTAutograd 产出 FX Graph → 交给 dynamo/inductor（下游分片）编译。
- 自定义算子（custom-ops）需声明 alias/安全语义才能被 functorch 正确变换。

## 9. 语言专项适配口径

Python 前端 + C++ 核心，不适用 Go/TS 专项。本叶子是 PyTorch **函数式变换与编译栈入口**：Python 侧以 dispatcher mode 栈（Python key）实现可组合变换，C++ Dispatcher 负责把算子路由到对应 mode。能力缝：服务定义侧为 `torch.func.vmap/grad` 等纯函数接口；提供方为 `_functorch` 变换内核与 AOTAutograd；消费方为 dynamo/inductor 编译栈。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 函数变换与 AOTAutograd 结构 | `functorch-architecture.html` | architecture | **standard**（showcase 未过 0 错误门槛，自动回退并如实披露） |

JSON IR 源文件：`json/functorch-architecture.json`。变换叠加/AOT 流程已在第 3 节文字化说明，不另出时序图。

![functorch 架构图](functorch-architecture.html)
