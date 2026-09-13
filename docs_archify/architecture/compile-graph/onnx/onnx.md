# ONNX 导出（onnx）

> 本文是 `compile-graph` 域下的叶子子系统文档。域级总览见 `../compile-graph.md`。
> 本文只展开 ONNX 格式导出与算子翻译；通用 save/load 序列化不在本叶子（属 serialization 分片）。
>
> 源码基准：pytorch，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| ONNX 导出入口 | `torch.onnx.export(model, args, f, ...)`：支持 nn.Module / ExportedProgram / ScriptModule，`dynamo=True` 走新版 FX 路径 | `torch/onnx/__init__.py:export()`@65 |
| 新版（Dynamo/FX）导出器 | 基于 FX Graph 逐节点翻译为 ONNX IR；`_translate_fx_graph` 为核心 | `torch/onnx/_internal/exporter/_core.py:_translate_fx_graph`@767、`_handle_call_function_node_with_lowering`@512 |
| torchlib 算子翻译表 | aten 算子到 ONNX 算子的注册与 lowering 规则 | `torch/onnx/_internal/exporter/_torchlib/`（`_torchlib_registry.py`、`ops/`） |
| ONNXProgram 产物 | 持有 onnxscript `ir.Graph`，提供 `save`/`model_proto`/`sample_outputs` | `torch/onnx/_internal/exporter/_onnx_program.py:ONNXProgram`@204 |
| 旧版 TorchScript 导出 | 按 opset 分文件的 symbolic 函数（opset7~20），基于 TorchScript 图 | `torch/onnx/_internal/torchscript_exporter/symbolic_opset*.py`、`registration.py`、`jit_utils.py` |
| 算子注册/覆盖 | 用户自定义 `symbolic` 函数注册到特定 opset | `torch/onnx/_constants.py`、`torch/onnx/symbolic_helper.py`、`torch/onnx/operators.py` |
| FX passes | 导出前对 FX 图做的图级改写 | `torch/onnx/_internal/fx/passes/`、`_internal/exporter/_fx_passes.py` |
| 验证 | 用 onnxruntime 跑推理对比 | `torch/onnx/_internal/exporter/_verification.py`、`verification.py`（ORT 为外部组件） |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `export()` | `torch/onnx/__init__.py:65` | 对外门面；`dynamo` 开关选择新/旧路径 |
| `_translate_fx_graph` | `_internal/exporter/_core.py:767` | 把 FX GraphModule 逐节点翻译为 onnxscript ir.Graph |
| `_handle_call_function_node_with_lowering` | `_core.py:512` | 对单个 aten 节点查 torchlib 翻译表并生成 ONNX 节点 |
| `ONNXProgram` | `_onnx_program.py:204` | 导出产物：ir.Graph + 输入输出样例；`model_proto`/`save`/`__call__`（ORT 推理） |
| `TorchTensor` | `_core.py:112` | 包装 torch.Tensor 为 onnxscript 张量类型 |
| `symbolic_opsetN.py` | `_internal/torchscript_exporter/` | 旧版：每个 opset 一份 aten→ONNX 符号函数 |

## 3. 关键调用链

**链 A：新版 `torch.onnx.export(model, args, dynamo=True)`**
1. `export()`（`__init__.py:65`）根据 `dynamo` 选择路径；新版先把 model 经 Dynamo/export 捕获为 FX GraphModule（见 `../dynamo/dynamo.md`、`../export/export.md`）。
2. 对 FX 图跑 `_internal/fx/passes` 与 `_fx_passes.py` 做规范化。
3. `_translate_fx_graph`（`_core.py:767`）按拓扑序遍历节点：placeholder/get_attr/call_function/output 分别由 `_handle_*` 函数处理。
4. `_handle_call_function_node_with_lowering`（@512）查 torchlib 翻译表（`_torchlib/ops/`），把 `aten::*` 调用翻译为 ONNX 算子节点，写入 onnxscript ir.Graph。
5. 封装为 `ONNXProgram`（`_onnx_program.py:204`）；用户调 `program.save("model.onnx")` 序列化 proto；`verify=True` 时用 onnxruntime 推理对比（外部组件）。

**链 B：旧版 `dynamo=False`**
1. model 先 trace/script 为 TorchScript 图（见 `../torchscript-jit/torchscript-jit.md`）。
2. `torchscript_exporter/` 按 opset 选 `symbolic_opsetN.py` 中的符号函数，逐节点映射为 ONNX 节点。

## 4. 配置项

| 配置 | 默认/行为 | 位置 |
|------|-----------|------|
| `dynamo` | 默认 True，走新版 FX 导出；False 走旧版 TorchScript 导出 | `__init__.py:export()` |
| `opset_version` | ONNX 算子集版本 | `__init__.py`、`_core.py:_get_onnxscript_opset`@494 |
| `dynamic_shapes` | 动态形状描述 | `__init__.py`、`_internal/exporter/_dynamic_shapes.py` |
| `custom_translation_table` | 用户自定义 aten→ONNX 翻译覆盖 | `__init__.py` |
| `external_data` | 权重是否外置存储 | `__init__.py` |
| `verify` / `profile` | 是否 ORT 验证 / 性能剖析 | `__init__.py`、`_verification.py` |
| `report` | 导出报告生成 | `_internal/exporter/_reporting.py` |

## 5. 错误与重试语义

- **不支持的算子**：torchlib 翻译表查不到 aten 算子时，`_handle_call_function_node_with_lowering` 抛 `UnsupportedOperator` 并指出算子名；不做自动重试，需用户提供 `custom_translation_table` 或分解该算子。
- **不支持的动态行为**：FX 图中出现未被支持的控制流/副作用时，`_fx_passes` 或翻译阶段报错并给出 node 位置。
- **验证失败**：`verify=True` 时 ORT 输出与 PyTorch 前向不一致，`_verification.py` 抛数值/形状不匹配错误。
- **不做运行期回退**：ONNX 导出是离线工具，导出失败直接报错。

## 6. 并发细节

- **导出期**：单次导出在单线程内完成；多模型并发导出各自持有独立 ONNXProgram。
- **ORT 验证**：`_onnx_program.py:_ort_session_initializer` 懒加载 onnxruntime 会话，会话对象不跨线程共享。
- **无内部锁**：翻译过程无共享可变状态；写盘用原子替换。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/onnx/` 全部：`__init__.py`、`_internal/exporter/`（新版）、`_internal/torchscript_exporter/`（旧版）、`symbolic_helper.py`、`operators.py`

**Out-of-Scope（不在本仓库源码内）**
- onnxruntime：推理验证后端，仅作为外部库调用
- onnxscript：IR 定义与 torchlib 算子库（`onnxscript` 包，第三方依赖，不在本仓库源码内）
- FX 图捕获：见 `../fx/fx.md`、`../dynamo/dynamo.md`
- 通用 save/load：见 serialization 分片

## 8. 与相邻子系统交互

- 上游 → 本叶子：用户调 `torch.onnx.export`；新版路径消费 `torch.export.ExportedProgram`（见 `../export/export.md`）或 Dynamo 产出的 FX 图。
- 本叶子 → 下游：产出 `.onnx` proto 文件，供第三方推理引擎（ONNX Runtime、TensorRT 等）消费；验证阶段调用 onnxruntime（外部）。
- 与 torchscript-jit：旧版导出路径以 TorchScript 图为输入；新版不再依赖 TorchScript IR。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，onnx 导出叶子纯 Python（`torch/onnx/`）。适配口径：

- **按能力缝归组**：模型输入 → FX 图捕获 → torchlib 查表翻译 → ONNX IR → 序列化/验证，是典型 ETL 管道。
- **图类型选择**：用 **dataflow** 表达"模型→FX 图→翻译→ONNX IR→proto"的数据流，比 architecture 更贴合翻译管道语义。
- **外部边界**：onnxruntime、onnxscript 标注为"不在本仓库源码内"。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| ONNX 导出数据流 | `onnx-dataflow.html` | dataflow | **standard**（showcase 严格布局校验对多行管线斜线边不通过，降 standard 渲染；三条跨行边改用 `vertical-channel` 路由后 standard 校验通过（45/45 ok:true），主流程语义不损失） |

JSON IR 源文件位于 `json/onnx-dataflow.json`。
