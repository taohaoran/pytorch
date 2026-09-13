# 算子分解与原语（decomposition）

> 本文是 `dynamo-inductor` 域下的叶子子系统文档。域级总览见 `../dynamo-inductor.md`。
> 本文只展开算子分解表、prims 原语与 refs 参考实现；inductor 编译期如何调用分解见 `../inductor/inductor.md`，
> export 阶段如何用分解表见 `../../compile-graph/export/export.md`。
>
> 源码基准：pytorch，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| 分解注册表 | 装饰器 `@register_decomposition(aten.op)` 把高阶算子注册到分解表 | `torch/_decomp/__init__.py:register_decomposition`@196、`_add_op_to_registry`@65 |
| 分解表查询 | 按算子名查表，返回该算子的分解实现 | `torch/_decomp/__init__.py:get_decompositions`@246、`remove_decompositions`@282 |
| 核心 aten 分解表 | 一组默认启用的分解（如 `tanh_backward`/`sigmoid_backward`/`hardsigmoid`） | `torch/_decomp/decompositions.py`（`@register_decomposition(aten.tanh_backward)`@125 等）、`core_aten_decompositions()`@305 |
| JVP/RNG 分解 | 为前向模式自动微分与 RNG 提供的额外分解表 | `torch/_decomp/decompositions_for_jvp.py`、`decompositions_for_rng.py` |
| prims 原语 | 一组无渐异、最小的基本算子（`aten.prims.*`），作为分解的终点 | `torch/_prims/__init__.py`、`context.py`、`executor.py`、`rng_prims.py`、`debug_prims.py` |
| prims 公共工具 | prims 的形状/类型推断包装 | `torch/_prims_common/`（`wrappers.py`） |
| refs 参考实现 | 基于 prims 的 Python 参考实现，用于测试与导出规范化 | `torch/_refs/nn/functional/`、`torch/_refs/linalg/`、`torch/_refs/fft.py`、`torch/_refs/special/` |
| out 参数转换 | 分解函数的 `out=` 参数规范化 | `torch/_decomp/__init__.py:_convert_out_params`@109 |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `register_decomposition(op, fn)` | `_decomp/__init__.py:196` | 装饰器：把 fn 注册为 op 的分解实现 |
| `get_decompositions(ops)` | `_decomp/__init__.py:246` | 返回指定算子集合的分解字典 |
| `core_aten_decompositions()` | `_decomp/__init__.py:305` | 默认核心分解表（inductor/export 使用） |
| `DecompositionTable` / `CustomDecompTable` | `torch/_dispatch/`、`torch/_ops.py` | 分派时查表，命中则用分解替代原算子 |
| `prims.*` 算子 | `torch/_prims/__init__.py` | 基本算子原语（如 `primitive_convolution`） |
| `_refs.*` | `torch/_refs/` | prims 之上的 Python 参考实现 |

## 3. 关键调用链

**链 A：编译期算子分解**
1. inductor/export 构建 `FxGraph` 或导出图时，遍历 FX 节点。
2. 对每个 aten 算子，查 `get_decompositions` 表（`_decomp/__init__.py:246`）。
3. 命中分解：把该节点替换为分解函数展开的子图（分解函数内部调用 prims 或更基础的 aten 算子）。
4. 递归分解直到所有节点都是"核心 aten"或 prims，后端可直接 codegen。
5. 未命中分解表的算子由后端直接支持；若后端也不支持则上报 graph break（见 `../dynamo/dynamo.md`）。

**链 B：prims 与 refs 的关系**
1. `torch/_prims/` 定义无渐异的最小算子集。
2. `torch/_refs/` 用 Python 实现这些原语的参考语义，供数值正确性测试与导出规范化使用。
3. 分解表的终点通常是 prims 或核心 aten 算子。

## 4. 配置项

| 配置 | 默认/行为 | 位置 |
|------|-----------|------|
| `core_aten_decompositions()` 返回集 | 默认启用的分解算子列表 | `_decomp/__init__.py:305` |
| `torch._inductor.config.decomposition_override` | 允许用户覆盖分解 | `_inductor/config.py` |
| 导出阶段分解表 | `default_decompositions()`（export 叶子） | `torch/export/exported_program.py:322` |
| `torch._prims_common.wrappers` | prims 形状/类型推断 | `_prims_common/wrappers.py` |

## 5. 错误与重试语义

- **无分解的算子**：查不到分解时，后端直接支持；后端也不支持则 graph break，不做自动重试。
- **分解循环**：若分解函数又调回自身依赖的算子，`DecompositionTable` 检测循环依赖并报错。
- **不影响正确性**：分解是代数恒等变换，分解前后数值语义一致；测试通过 refs 对比验证。
- **不做运行期重试**：分解在编译期一次性完成。

## 6. 并发细节

- **分解表只读**：注册发生在模块导入期（装饰器执行），运行期表只读，多线程并发查表无竞争。
- **无锁**：分解函数本身是纯函数，不持有共享状态。
- **prims 执行**：prims 经 dispatcher 分派到后端实现，与普通 aten 算子并发语义一致。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/_decomp/`：注册表与分解函数集合
- `torch/_prims/`：基本算子原语定义
- `torch/_prims_common/`：原语公共工具
- `torch/_refs/`：参考实现

**Out-of-Scope（不在本仓库源码内）**
- 后端代码生成如何消费分解后图：见 `../inductor/inductor.md`
- dispatcher 机制：见 torch/_dispatch/（本叶子只提供表，不实现分派）
- TorchScript 旧路径的分解：见 `../../compile-graph/torchscript-jit/torchscript-jit.md`（`torch/csrc/jit/passes/decompose_ops.cpp` 是 C++ 侧独立分解）

## 8. 与相邻子系统交互

- 上游 → 本叶子：inductor 编译（`../inductor/inductor.md`）与 torch.export（`../../compile-graph/export/export.md`）在图规范化阶段调用分解表。
- 本叶子 → 下游：分解把高阶算子展开为 prims/核心 aten，后端直接 codegen；refs 用于测试与数值校验。
- 与 functorch：AOTAutograd 也消费分解表（见 functorch 分片）。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，decomposition 叶子纯 Python（`torch/_decomp/`、`_prims/`、`_refs/`）。适配口径：

- **按能力缝归组**：分解注册表 → 分解函数 → prims 原语 → refs 参考实现 → 后端消费。
- **图类型选择**：architecture 表达"高阶算子→查表→分解→prims/refs→后端"的结构；分解本身是线性查表替换，不单列 dataflow。
- **外部边界**：dispatcher、后端 codegen 标注为相邻组件。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 算子分解与原语架构 | `decomposition-architecture.html` | architecture | **standard**（showcase 因垂直边标签与节点重叠降 standard；已移除垂直边文字标签，主流程语义不损失） |

JSON IR 源文件位于 `json/decomposition-architecture.json`。
