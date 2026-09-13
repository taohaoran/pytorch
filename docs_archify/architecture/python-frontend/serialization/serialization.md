# serialization（serialization）

> 本文是 `python-frontend` 域下的叶子子系统文档。域级总览见 `../python-frontend.md`。
> 本文聚焦 **torch.save/load、checkpoint、zipfile 序列化格式、Storage 序列化、weights_only 安全加载**；
> ExportedProgram 导出见 compile-graph 分片，不在本叶子。
>
> 源码基准：PyTorch `main`，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `torch.save` | 序列化任意对象（模型/张量/字典）到文件或流 | `torch/serialization.py:944` |
| `torch.load` | 反序列化，支持 map_location 跨设备加载 | `serialization.py:1315` |
| 新 zipfile 格式 | `_use_new_zipfile_serialization=True` 默认，zip 容器分离张量二进制与 pickle 元数据 | `serialization.py:1001-1002` |
| 文件/缓冲打开器 | `_open_file`/`_open_buffer_reader`/`_open_zipfile_reader`/`_open_zipfile_writer_*` | `serialization.py:774,782,805,810,839` |
| weights_only 安全加载 | 仅反序列化张量，禁用任意类实例化，防 pickle 反序列化漏洞 | `serialization.py:318`（`safe_globals`） |
| mmap 支持 | `set_default_mmap_options` 内存映射大文件 | `serialization.py:229` |
| skip_data | 只加载结构不加载张量数据 | `serialization.py:385` |
| 端序处理 | `LoadEndianness` 枚举处理跨字节序加载 | `serialization.py:132` |
| C++ 读写器 | `PyTorchFileReader`/`PyTorchFileWriter` 封装 zip 二进制 IO | `torch/csrc/serialization.{h,cpp}` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `def save` | `serialization.py:944` | 序列化入口，选 zip/legacy 格式 |
| `def _save` | `serialization.py:1183` | 实际写出逻辑 |
| `def load` | `serialization.py:1315` | 反序列化入口，处理 map_location/weights_only |
| `class safe_globals` | `serialization.py:318` | weights_only 模式下的临时安全白名单上下文 |
| `class _opener` 及子类 | `serialization.py:763-863` | 统一封装文件/缓冲的读写打开 |
| `SourceChangeWarning` | `serialization.py:110` | 源文件路径与加载时不一致告警 |

## 3. 关键调用链

1. **`torch.save(obj, f)`**：`save`（`serialization.py:944`）→ `_use_new_zipfile_serialization=True` 时 `_open_zipfile_writer(f)`（`:858`）建 zip → `_save`（`:1183`）把张量二进制经 `torch._C` 写入 `PyTorchFileWriter`，对象图经 `pickle_module` 序列化到 zip 的 data.pkl → 关闭。
2. **`torch.load(f, map_location=...)`**：`load`（`:1315`）→ 若为 zip 格式 `_open_zipfile_reader`（`:1572`）→ 用 unpickler 读 data.pkl，遇到张量时从 zip 惰性读 Storage 并按 map_location 重映射设备 → 重建对象图。
3. **weights_only**：走 `_weights_only_unpickler`，仅允许张量/原语等白名单类型，拒绝任意类调用。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `_use_new_zipfile_serialization` | True，新 zip 格式 | `save:949` |
| `pickle_protocol` | 默认最高协议 | `save` |
| `map_location` | 加载时重映射设备（cpu/cuda:0） | `load:1315` |
| `weights_only` | False；True 时安全加载 | `load` |
| `mmap` | `set_default_mmap_options` 控制大文件映射 | `serialization.py:229` |

## 5. 错误与重试语义

- 加载损坏文件抛出 pickle/zip 异常，不重试。
- `SourceChangeWarning`（`:110`）在加载的源路径与保存时不同（如模型类移动）时告警，不阻断。
- weights_only 遇白名单外类直接抛 `UnpicklingError`，拒绝执行任意代码。

## 6. 并发细节

- `_SerializationLocal`（`serialization.py:99`）为 `threading.local`，保存线程级状态，避免多线程 save 相互污染。
- 张量二进制 IO 在 C++ 侧完成；Python pickle 阶段受 GIL 约束。
- 加载时张量可惰性按需读取（mmap/zip 随机读），无需一次性全量入内存。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/serialization.py`、`torch/csrc/serialization.{h,cpp}`。

**Out-of-Scope（相邻叶子）**
- 模型 `state_dict` 生成/消费 → `nn-modules`。
- 优化器 `state_dict` → `optim`。
- `torch.package`/模型导出（ExportedProgram）→ compile-graph 分片。
- 分布式 checkpoint 聚合 → distributed 分片。

## 8. 与相邻子系统交互

- `nn.Module.state_dict()` / `optim.Optimizer.state_dict()` → `torch.save` 持久化。
- `torch.load` → 重建张量（torch-api 的 Tensor/Storage）→ 按 map_location 映射设备（device-context）。
- C++ 读写器 `PyTorchFileReader/Writer` ↔ `torch._C` 扩展（见 torch-api 绑定面）。

## 9. 语言专项适配口径

Python 前端 + C++ 核心，不适用 Go/TS 专项。能力缝：服务定义侧为 save/load 公共 API；提供方为 Python 门面 + C++ 二进制 IO；消费方为模型/优化器 checkpoint。张量大内存对象下沉 C++ 读写，Python 只处理对象图编排。安全面（weights_only）是本叶子特有的能力缝扩展点。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 序列化组件结构 | `serialization-architecture.html` | architecture | **standard**（showcase 因标签间距未过 0 错误门槛，自动回退并如实披露） |

JSON IR 源文件：`json/serialization-architecture.json`。save/load 流程已在第 3 节文字化说明，不另出时序图。

![serialization 架构图](serialization-architecture.html)
