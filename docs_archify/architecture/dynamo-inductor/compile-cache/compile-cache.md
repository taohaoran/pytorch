# 编译入口与缓存（compile-cache）

> 本文是 `dynamo-inductor` 域下的叶子子系统文档。域级总览见 `../dynamo-inductor.md`。
> 本文只展开 `torch.compile` 入口、编译配置与多层缓存（Guard/FxGraph/磁盘/内核）；
> 字节码追踪见 `../dynamo/dynamo.md`，后端代码生成见 `../inductor/inductor.md`。
>
> 源码基准：pytorch，commit `bb830660ca060cf2ca091b2bd2c01c0bcd96f2d1`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `torch.compile` 门面 | 用户侧装饰器/函数：选择后端、模式、动态形状等参数 | `torch/__init__.py:torch.compile`@3045；懒加载 `torch/_compile.py` |
| dynamo 开关 | 内部 `_disable_dynamo` 等辅助，避免循环 import | `torch/_compile.py:_disable_dynamo`@36 |
| 编译配置 | dynamo/inductor 全部可调配置项与环境变量别名 | `torch/_dynamo/config.py`（`accumulated_recompile_limit=256`@124、`cache_size_limit`@137）、`torch/_inductor/config.py` |
| 内存缓存抽象 | 泛型 `Cache`/`InMemoryCache`/`AsyncCache` 接口 | `torch/_inductor/cache.py:Cache`@36、`InMemoryCache`@64、`AsyncCache`@216 |
| 磁盘缓存 | 键值对持久化到磁盘，异步写回 | `torch/_inductor/cache.py:OnDiskCache`@249、`InductorOnDiskCache`@397 |
| FxGraph 缓存 | 缓存编译后的 FX 图（CompiledFxGraph），可跨进程复用 | `torch/_inductor/codecache.py:FxGraphCache`@2067、`CacheabilityValidator`@1080、`FxGraphCachePickler`@718 |
| 内核缓存 | GPU Triton 内核参数、CPU C++ 内核共享库 | `codecache.py:CudaKernelParamCache`@2562、`CpuTritonKernelCache`@2662、`CppCodeCache`@3947 |
| 本地/持久缓存 | 进程内 LocalCache 与跨进程 PersistentCache 两层 | `codecache.py:LocalCache`@426、`PersistentCache`@451 |
| 文件锁 | 多进程并发写盘的互斥锁 | `torch/utils/_filelock.py:FileLock`@9 |
| dynamo 帧缓存 | 按 code object 缓存 guard 与编译产物 | `torch/_dynamo/eval_frame.py`、`cache_size.py`、`funcname_cache.py` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|------|------|------|
| `torch.compile` | `torch/__init__.py:3045` | 对外门面：`compile(model, backend="inductor", dynamic=False)` |
| `Cache` / `OnDiskCache` | `cache.py:36/249` | 泛型键值缓存抽象；OnDisk 用 sha256 键 + pickle 序列化 |
| `FxGraphCache` | `codecache.py:2067` | 带 guard 的 FX 图缓存：键为图内容+配置哈希，值为 `CompiledFxGraph` |
| `CacheabilityValidator` | `codecache.py:1080` | 判断图是否可缓存（含不可缓存操作则跳过） |
| `CppCodeCache` | `codecache.py:3947` | 管理已编译 C++ 内核的共享库加载 |
| `FileLock` | `torch/utils/_filelock.py:9` | 文件系统锁，保护 OnDiskCache 并发写 |
| `GuardedCache` | `codecache.py:1871` | 带 guard 的缓存包装 |

## 3. 关键调用链

**链 A：`torch.compile(f)(x)` 缓存命中路径**
1. `torch.compile`（`torch/__init__.py:3045`）装饰 f，返回带编译逻辑的 callable。
2. 首次调用进入 dynamo eval_frame 钩子（见 `../dynamo/dynamo.md`）。
3. 后续调用先查 **Guard 缓存**：输入满足 guard 则直接执行已编译代码，零 Python 开销。
4. Guard 未命中时查 **FxGraphCache**（`codecache.py:2067`）：按图内容+配置哈希查磁盘，命中则加载 `CompiledFxGraph`，跳过后端编译。
5. 仍未命中走完整编译（dynamo 追踪→AOT→inductor），产物写入缓存。

**链 B：磁盘缓存写**
1. `OnDiskCache`（`cache.py:249`）把值 pickle 到 `~/.cache/torch/inductor/`。
2. 写盘前 `FileLock`（`torch/utils/_filelock.py:9`）获取互斥锁，防止多进程同时写同一文件。
3. `AsyncCache` 支持后台线程异步写回，不阻塞主路径。

## 4. 配置项

| 配置 | 默认/行为 | 位置 |
|------|-----------|------|
| `torch.compile(backend=...)` | 后端选择，默认 inductor | `torch/__init__.py:3045` |
| `torch._dynamo.config.cache_size_limit` | 每帧重编译上限 | `_dynamo/config.py:137` |
| `torch._dynamo.config.accumulated_recompile_limit` | 全局重编译上限，默认 256 | `_dynamo/config.py:124` |
| `torch._inductor.config.fx_graph_cache` | 启用 FxGraph 磁盘缓存 | `_inductor/config.py` |
| `TORCHINDUCTOR_CACHE_DIR` | 自定义缓存目录 | `codecache.py`、`cache.py` |
| `TORCHINDUCTOR_DISABLE_CACHE` | 禁用磁盘缓存 | `codecache.py` |
| `compile_threads` | 异步编译线程数 | `_inductor/config.py` |

## 5. 错误与重试语义

- **缓存不可用**：`CacheabilityValidator` 发现图含不可缓存操作（如副作用、未支持的自定义 op），抛 `BypassFxGraphCache`（`codecache.py:1073`），跳过缓存直接编译。
- **缓存损坏**：pickle 反序列化失败时抛 `CacheError`（`cache.py:30`），调用方删除损坏缓存项重新编译。
- **锁竞争**：`FileLock` 获取失败时等待；超时则视为缓存不可用，走在线编译。
- **不做自动重试**：缓存层失败即降级为在线编译，不反复重试缓存读。

## 6. 并发细节

- **异步写缓存**：`AsyncCache`（`cache.py:216`）用后台线程池写盘，主路径不阻塞。
- **多进程安全**：`FileLock` 基于 `fcntl`/`portalocker` 实现跨进程互斥；多训练进程共享同一缓存目录时不会写坏文件。
- **内核缓存加载**：`CppCodeCache`/`CudaKernelParamCache` 用进程级字典缓存已加载共享库，`dlopen` 幂等。
- **dynamo 帧缓存**：按 code object 键控，首次编译完成前其他线程走 eager，编译完成后无锁读取（不可变对象）。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `torch/_compile.py`、`torch/__init__.py:compile`
- `torch/_dynamo/config.py`、`torch/_inductor/config.py`
- `torch/_inductor/cache.py`、`codecache.py`、`cache_key.py`
- `torch/utils/_filelock.py`

**Out-of-Scope（不在本仓库源码内）**
- 字节码追踪与 guard 求值：见 `../dynamo/dynamo.md`
- 后端代码生成：见 `../inductor/inductor.md`
- 文件系统本身：缓存目录在 `~/.cache/`，不在本仓库

## 8. 与相邻子系统交互

- 上游 → 本叶子：用户调 `torch.compile`；配置项来自 `torch/_dynamo/config.py` 与 `torch/_inductor/config.py`。
- 本叶子 → 下游：缓存未命中时调 dynamo 追踪（`../dynamo/dynamo.md`）与 inductor 编译（`../inductor/inductor.md`）；缓存命中时直接返回已编译 callable。
- 与 onnx/export：export 不依赖本缓存；ONNX 导出是离线工具。

## 9. 语言专项适配口径

本项目为 **Python 前端 + C++ 核心**，compile-cache 叶子纯 Python（`torch/_inductor/cache.py`、`codecache.py`）。适配口径：

- **按能力缝归组**：入口门面 → 配置 → 三层缓存（Guard/FxGraph/磁盘）→ 文件锁 → 内核缓存。
- **图类型选择**：architecture 表达"入口→dynamo→多级缓存→后端"的分层结构；缓存查找是线性查表，不单列 dataflow。
- **外部边界**：文件系统、Triton 内核产物标注为外部/相邻。
- **并发重点**：FileLock 跨进程互斥与 AsyncCache 后台写是本叶子的并发核心，见第 6 节。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| torch.compile 入口与编译缓存分层 | `compile-cache-architecture.html` | architecture | **standard**（showcase 因短边标签间距不足降 standard；主流程语义不损失） |

JSON IR 源文件位于 `json/compile-cache-architecture.json`。
