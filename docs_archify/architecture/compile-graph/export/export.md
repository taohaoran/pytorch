# 导出（export）

> 本文是 `compile-graph` 域下的叶子子系统文档。域级总览见 `../compile-graph.md`。
> 本文只展开基于 FX 的 `ExportedProgram` 导出、动态形状与序列化；FX Graph IR 本身见 `../fx/fx.md`，
> 算子分解原语见 `../../dynamo-inductor/decomposition/decomposition.md`。
>
> 源码基准：pytorch，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| 非追踪导出入口 | `torch.export.export()`：以 Dynamo 字节码追踪方式捕获 nn.Module，产出无副作用、规范化的 FX 图 | `torch/export/__init__.py:export()`@59、`torch/export/_trace.py:ExportArtifact`@198、`ATenExportArtifact`@175 |
| ExportedProgram | 导出产物对象：封装 GraphModule、图签名（输入/输出/参数/缓冲区）、形状约束、模块调用图、状态字典 | `torch/export/exported_program.py:ExportedProgram`@1070 |
| 动态形状约束 | 用 `Dim`/`_Constraint` 表达 SymInt 维度的静态/动态/派生/相等约束，支持 `dynamic_dim` | `torch/export/dynamic_shapes.py:Dim`@109、`_Constraint`@383、`dims()`@360 |
| 图规范化 Passes | 移除 AOT 副作用 token、自动函数化清理、getitem 消除、requires_grad 保留等 | `torch/export/passes/`、`_remove_auto_functionalized_pass.py`、`_remove_effect_tokens_pass.py`、`_common_getitem_elimination_pass`@852 |
| 算子分解 | 导出时按 `default_decompositions()` 把高阶算子分解为 ATen 原语 | `exported_program.py:default_decompositions()`@322、`decomp_utils.py` |
| 序列化 pt2 包 | 把 ExportedProgram 打包为自包含归档（图、常量、权重、元信息） | `torch/export/pt2_archive/_package.py`、`constants.py`；`_safeguard.py` |
| 图签名 | 记录参数/缓冲区/输入/输出的名字与元信息，支持跨版本迁移 | `torch/export/graph_signature.py`、`_swap.py`、`_unlift.py` |
| 未导出检测 | 标记未被图使用的参数/缓冲区 | `torch/export/_state_dict_utils.py` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `export()` | `torch/export/__init__.py:59` | 对外门面：`export(f, args, kwargs=..., dynamic_shapes=...) -> ExportedProgram` |
| `ExportedProgram` | `exported_program.py:1070` | 产物对象；`graph_module`/`graph`/`graph_signature`/`module_call_graph` 属性；`module(check_guards=True)`@1479 转成可执行 `torch.fx.GraphModule` |
| `Dim` | `dynamic_shapes.py:109` | 用户侧动态形状句柄；`_StaticDim`/`_DerivedDim` 表达静态/派生维度 |
| `_Constraint` | `dynamic_shapes.py:383` | 对 SymInt 表达式的不等式约束（min/max），经 `_process_equalities`@532 合并相等关系 |
| `ATenExportArtifact` | `_trace.py:175` | 追踪中间产物：aten 级 FX GraphModule + 伪输入 + 形状环境 |
| `default_decompositions()` | `exported_program.py:322` | 导出阶段默认应用的分解表（见 decomposition 叶子） |

## 3. 关键调用链

**链 A：`torch.export.export(f, args, dynamic_shapes=...)`**
1. `export()`（`__init__.py:59`）规范化参数，把 `dynamic_shapes` 解析为 `_Constraint` 集合（`dynamic_shapes.py`）。
2. 进入 `_trace.py`：复用 Dynamo（见 `../dynamo/dynamo.md`）做字节码级追踪，配合 `ExportDynamoConfig`（`_trace.py:141`）关闭不必要的后端，产出 aten 级 FX GraphModule（`ATenExportArtifact`）。
3. 对追踪图跑规范化 passes：去副作用 token、函数化清理、getitem 消除、保留 requires_grad（`_preserve_requires_grad_pass`@487）。
4. 按 `default_decompositions()` 做算子分解，把高阶 op 降级为 ATen 原语。
5. 构造 `ExportedProgram`：绑定 GraphModule、`graph_signature`、`module_call_graph`、形状约束。
6. 用户可 `ep.save("file.pt2")` 走 `pt2_archive/_package.py` 序列化，或 `torch.export.load` 反序列化。

**链 B：`ep.module()` 拿回可执行图**
1. `ExportedProgram.module(check_guards=True)`（`exported_program.py:1479`）执行 guard 检查。
2. 通过 `_convert_guards_to_code`（@1797）把形状约束编译为运行时检查，返回可直接前向的 `torch.fx.GraphModule`。

## 4. 配置项

| 配置 | 默认/行为 | 位置 |
|------|-----------|------|
| `dynamic_shapes` 参数 | 传入 `Dim`/字典描述哪些维度动态及其范围 | `__init__.py:export()` |
| `strict` | 严格模式下追踪必须完整覆盖图（无图 break） | `_trace.py`、`_draft_export.py` |
| `pre_dispatch` | 是否预分发模式追踪 | `_trace.py:ExportDynamoConfig` |
| `custom_op` | 注册自定义算子的导出规则 | `torch/export/custom_ops.py` |
| pt2 包格式 | 归档布局与常量存储 | `pt2_archive/constants.py` |

## 5. 错误与重试语义

- **追踪失败（图 break）**：若 Dynamo 追踪遇到不支持的 Python 构造而 graph break，严格模式抛 `Unsupported` 错误并指出断点；不做自动重试，需用户改写代码或用 `@torch._dynamo.disable`。
- **动态形状冲突**：`_Constraint` 合并后若不等式无解（如同时要求 dim>10 与 dim<5），`dynamic_shapes.py` 在约束求解阶段抛 `AssertionError`。
- **guard 校验**：`ep.module()` 后运行时若输入违反形状约束，`_convert_guards_to_code` 抛出形状不匹配错误。
- **不做 eager 回退**：export 是离线产物，不参与运行期编译回退；运行期回退属于 Dynamo 层（见 `../dynamo/dynamo.md`）。

## 6. 并发细节

- **追踪期**：导出本身在单线程内完成（复用 Dynamo 的字节码追踪，见 dynamo 叶子）；多线程并发导出各自持有独立 `ExportedProgram`。
- **序列化**：`pt2_archive/_package.py` 写盘用文件锁/原子替换，避免半写文件；读取是只读。
- **无运行期锁**：`ExportedProgram.module()` 产出的 GraphModule 可被多线程并发执行，图在构造后只读。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/export/` 全部：`__init__.py`、`_trace.py`、`exported_program.py`、`dynamic_shapes.py`、`passes/`、`pt2_archive/`

**Out-of-Scope（不在本仓库源码内）**
- Dynamo 字节码追踪与 guard：见 `../dynamo/dynamo.md`（export 复用但不重写）
- 算子分解原语与参考实现：见 `../../dynamo-inductor/decomposition/decomposition.md`
- ONNX 格式转换：见 `../onnx/onnx.md`
- FX Graph IR 定义：见 `../fx/fx.md`

## 8. 与相邻子系统交互

- 上游 → 本叶子：用户调 `torch.export.export`；Dynamo（`../dynamo/dynamo.md`）提供字节码追踪能力，AOTAutograd 提供前向/反向图捕获。
- 本叶子 → 下游：ExportedProgram 可被 inductor 后端编译（`../../dynamo-inductor/inductor/inductor.md`）、ONNX 导出（`../onnx/onnx.md`）、或部署到移动端/推理引擎。
- 与 decomposition：导出调用 `default_decompositions()`，分解表来源见 decomposition 叶子。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，export 叶子纯 Python（`torch/export/`）。适配口径：

- **按能力缝归组**：追踪入口（复用 Dynamo）→ 动态形状约束 → 规范化 passes → 分解 → ExportedProgram 产物 → 序列化 pt2 包。
- **图类型选择**：architecture 表达"用户→追踪/约束→产物→持久化"的组件关系；导出本身是线性流水线，不再单列 dataflow。
- **外部边界**：Dynamo、AOTAutograd、第三方推理运行时标注为相邻叶子/外部组件。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| torch.export 导出架构 | `export-architecture.html` | architecture | **standard**（showcase 因短边标签间距不足降 standard；主流程语义不损失） |

JSON IR 源文件位于 `json/export-architecture.json`。
