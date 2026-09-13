# compile-graph 域总览

> 本域包含以下叶子子系统；各叶子详情见对应文档。
> 源码基准：pytorch，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 域职责

compile-graph 域覆盖 PyTorch 的**图中间表示与图式导出**体系：从旧的 TorchScript C++ IR，到新的 Python 级 FX Graph IR，再到基于 FX 的导出与 ONNX 格式转换。本域解决"把 eager 模式的模型转化为静态、可序列化、可跨部署的图"这一问题，向下衔接 ATen 算子，向上服务于移动端部署、导出与推理后端。

域内存在三代 IR，边界清晰：
- **TorchScript IR**（C++ SSA 图）：旧 JIT 路径，`torch/csrc/jit/`。
- **FX Graph IR**（Python `GraphModule`）：Dynamo/export/inductor 时代的统一图表示，`torch/fx/`。
- **ExportedProgram**（FX + 元数据）：可序列化的规范化导出产物，`torch/export/`。

ONNX 导出跨两代 IR：旧版基于 TorchScript，新版基于 FX。

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序图 / 数据流图 | 职责一句话 |
|------|------|--------|-------------------|-----------|
| torchscript-jit | [torchscript-jit.md](torchscript-jit/torchscript-jit.md) | [架构图](torchscript-jit/torchscript-jit-architecture.html) | — | TorchScript C++ IR、script/trace 编译、解释执行与序列化 |
| fx | [fx.md](fx/fx.md) | [架构图](fx/fx-architecture.html) | — | Python 级 FX Graph IR、符号追踪、Interpreter 与图变换 |
| export | [export.md](export/export.md) | [架构图](export/export-architecture.html) | — | 基于 FX 的 ExportedProgram 导出、动态形状与 pt2 包序列化 |
| onnx | [onnx.md](onnx/onnx.md) | — | [数据流图](onnx/onnx-dataflow.html) | ONNX 格式导出：FX/TorchScript 图翻译为 ONNX 算子 |

## 3. 域级机制细节

- **三代 IR 的演进**：TorchScript（C++，2018 前后）→ FX（Python，2020）→ ExportedProgram（FX + 元数据，2023+）。新代码应优先 FX/export 路径；TorchScript 进入维护模式。
- **图追踪两种范式**：tracing（按执行记录算子，无控制流）与 scripting（静态编译 Python 子集，支持控制流）。FX 的符号追踪是 scripting 思路在 Python 层的现代化。
- **导出的规范化**：无论 export 还是 onnx，都经过"图捕获 → 规范化 passes → 算子分解 → 序列化"流水线；分解表来源见 `../dynamo-inductor/decomposition/decomposition.md`。
- **外部边界**：ONNX Runtime、onnxscript、Triton 均不在本仓库源码内。

## 4. 域级图

本域未单独产出域级架构图；各叶子已有图，系统级架构图由组织者统一汇总。
