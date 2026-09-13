# custom-ops（custom-ops）

> 本文是 `extensions` 域下的叶子子系统文档。域级总览见 `../extensions.md`。
> 本文聚焦 **torch.library 自定义算子注册、_custom_op Python API、cpp_extension JIT/setuptools 构建**；
> 内置算子命名空间见 `python-frontend/torch-api`，函数变换见 `functorch`。
>
> 源码基准：PyTorch `main`，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| torch.library.Library | 注册门面类：define 声明 schema、impl 绑定内核、fallback 兜底 | `torch/library.py:212` |
| `Library.define` | 声明算子 schema（含 alias analysis、tags） | `library.py:272` |
| `Library.impl` | 按 DispatchKey（CPU/CUDA/...）绑定实现函数 | `library.py:441` |
| `Library.fallback` | 未注册 key 的兜底内核 | `library.py:551` |
| 模块级便捷 API | `torch.library.define/impl` 装饰器式注册 | `library.py:695,776` |
| _custom_op Python API | 新一代 Python 自定义算子实现 | `torch/_custom_op/{impl,autograd}.py` |
| CppExtension / CUDAExtension | setuptools 扩展构建器 | `torch/utils/cpp_extension.py:1435,1505` |
| JIT load / load_inline | 现场 ninja 编译并加载 C++/CUDA 扩展 | `cpp_extension.py:1880,2162` |
| 编译流程 | `_jit_compile`→`_write_ninja_file_and_build_library` | `cpp_extension.py:2364,2549` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `class Library` | `library.py:212` | 算子命名空间注册句柄，生命周期内持有注册 |
| `Library.define` | `library.py:272` | 注册算子 schema |
| `Library.impl` | `library.py:441` | 把 Python/C++ 函数绑到某 DispatchKey |
| `torch.library.impl`（装饰器） | `library.py:776` | `@torch.library.impl(qualname, key)` 便捷装饰器 |
| `def CppExtension/CUDAExtension` | `cpp_extension.py:1435,1505` | setuptools Extension 子类 |
| `def load/load_inline` | `cpp_extension.py:1880,2162` | JIT 编译并 import 扩展 |
| `_clear_torch_ops_cache` | `library.py:616` | 重注册时清算子缓存 |

## 3. 关键调用链

1. **注册 Python 自定义算子**：`lib = torch.library.Library("mylib", "FRAGMENT")` → `lib.define("sin(Tensor x) -> Tensor")`（`library.py:272`）→ `lib.impl("sin", my_cpu_fn, "CPU")`（`:441`）→ 写入 C++ Dispatcher 注册表 → 用户经 `torch.ops.mylib.sin(x)` 调用。
2. **JIT 编译 C++ 扩展**：`torch.utils.cpp_extension.load(name, sources)`（`cpp_extension.py:1880`）→ `_jit_compile`（`:2364`）→ `_write_ninja_file_and_build_library`（`:2549`）调 ninja 编译为 `.so` → `dlopen` 加载 → 扩展内 `TORCH_LIBRARY` 块把算子注册进 Dispatcher。
3. **autograd**：`_custom_op/autograd.py` 注册 backward，使自定义算子可参与反向传播。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `verbose`（load） | 打印编译命令 | `cpp_extension.py:1880` |
| `is_python_module`（load_inline） | 包装为 Python 模块 | `cpp_extension.py:2162` |
| `with_cuda` | 是否包含 CUDA 头/库 | `cpp_extension.py` |
| `build_directory` | 编译产物目录 | `cpp_extension.py` |
| `alias_analysis`（define） | MUTATES_VIEW/VIEW 等别名分析标记 | `library.py:272` |

## 5. 错误与重试语义

- schema 不合法由 `_validate_out_schema`/`_validate_inplace_schema`（`library.py:67,138`）校验失败即抛错。
- 编译失败（缺编译器/ninja）直接抛 `RuntimeError`，不自动重试；用户需安装工具链。
- 重注册同名算子由 `_clear_torch_ops_cache`（`library.py:616`）刷新缓存，避免 stale packet。

## 6. 并发细节

- 算子注册在进程启动/import 阶段一次性完成，运行期只读，无锁。
- cpp_extension JIT 编译在独立子进程（ninja）中进行，与主 Python 进程隔离；加载 `.so` 经 RTLD 守卫（见 torch-api `dl_open_guard`）。
- 注册后的算子经 C++ Dispatcher 在设备侧多线程执行。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/library.py`、`torch/_custom_op/`、`torch/utils/cpp_extension.py`。

**Out-of-Scope（相邻叶子）**
- `torch.ops` 命名空间如何解析调用 → `python-frontend/torch-api`（本叶子只负责注册侧）。
- 算子分发到具体设备内核 → cuda-backend 分片。
- 编译缓存复用与 `TORCH_EXTENSIONS_DIR` 的环境细节属 cpp_extension 内部，不展开。

## 8. 与相邻子系统交互

- 算子作者 → `torch.library` → 注册到 C++ Dispatcher → 与内置算子同列于 `torch.ops.*`（torch-api 消费）。
- cpp_extension 编译产物（.so）→ 通过 `torch._C` 扩展机制加载（见 torch-api 绑定面）。
- 自定义算子的 autograd 语义 ↔ `_custom_op/autograd.py` ↔ autograd 引擎。

## 9. 语言专项适配口径

Python 前端 + C++ 核心，不适用 Go/TS 专项。本叶子是 **Python↔C++ 扩展缝**的核心：Python 侧 `torch.library` 提供声明式注册 DSL，C++ 侧 `TORCH_LIBRARY`/pybind 负责实际内核与编译。能力缝：服务定义侧为算子 schema 字符串；提供方为 Library/impl 与 cpp_extension 工具链；消费方为 Dispatcher 与 `torch.ops`。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 自定义算子注册与构建结构 | `custom-ops-architecture.html` | architecture | **standard**（showcase 未过 0 错误门槛，自动回退并如实披露） |

JSON IR 源文件：`json/custom-ops-architecture.json`。注册/JIT 流程已在第 3 节文字化说明，不另出时序图。

![custom-ops 架构图](custom-ops-architecture.html)
