# uv 环境管理

NORN 0.5.0 的环境由 `pyproject.toml` 定义，完整解析结果及包哈希保存在 `uv.lock`。使用 uv 0.9.17 或更新版本；锁定配置面向 Linux x86_64、Python 3.9–3.12。`.python-version` 默认选择 Python 3.11，uv 会按需安装该解释器。

## 选择运行环境

在仓库根目录执行其中一个配置：

| 用途 | 命令 |
|---|---|
| 数据准备和配置检查，不含 PyTorch | `uv sync --locked --no-dev` |
| CPU 训练和推理 | `uv sync --locked --extra cpu --no-dev` |
| A40／CUDA 12.1 训练和推理 | `uv sync --locked --extra cuda --no-dev` |
| A40、动态板块、NetCDF 和视频 | `uv sync --locked --extra cuda --extra kinematics --extra export --extra viz --no-dev` |
| CPU 开发和完整动态／GMT 测试 | `uv sync --locked --extra cpu --extra kinematics --extra export --extra viz` |

`cpu` 与 `cuda` 互斥，不能同时选中；不要使用 `--all-extras`。切换环境时重新执行对应的 `uv sync`，uv 会移除上一个配置独有的包。CUDA 配置需要可用的 NVIDIA 驱动；不需要为 NORN 的球谐变换编译自定义 CUDA 扩展。

基础依赖为 NumPy、pandas、SciPy 和 openpyxl。`cpu`／`cuda` 安装 PyTorch 2.5.1 与 torch-harmonics 0.8.0，分别从官方 CPU／cu121 索引获取 PyTorch。`kinematics` 安装 pygplates 1.0.0（动态几何和古位置均需要）；`export` 安装 xarray；`viz` 安装 matplotlib。MP4 另需系统 `ffmpeg`，GIF 使用 matplotlib 的 Pillow writer。开发依赖在 `dev` 组，运行环境可用 `--no-dev` 排除。

## 运行和检查

```bash
source .venv/bin/activate
norn --version
norn --help
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

不激活 shell 环境时，使用 `uv run --no-sync norn ...`。`--no-sync` 使用刚才选定的环境，不在执行训练时切换 CPU／CUDA 依赖。配置文件内的数据路径仍相对于配置文件解析，与 uv 的环境路径无关。

指定其他 Python 版本：

```bash
uv sync --locked --python 3.12 --extra cpu --extra export
uv run --python 3.12 --no-sync pytest -q
```

需要并存多个环境时，可分别设置 `UV_PROJECT_ENVIRONMENT`，例如：

```bash
UV_PROJECT_ENVIRONMENT=.venv-cpu uv sync --locked --extra cpu --extra export
UV_PROJECT_ENVIRONMENT=.venv-cpu uv run --no-sync norn --version
```

虚拟环境、数据、checkpoint 和预测产物不提交 Git。

## 依赖更新与构建

```bash
uv lock --check
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
uv build --no-sources
```

日常安装和 CI 使用 `--locked`，依赖不匹配时直接失败。更新依赖时编辑 `pyproject.toml` 后运行 `uv lock`，重新验证，再一起提交这两个文件。源码包包含锁文件和 Python 版本文件；wheel 保留标准 Python 包元数据。

CI 使用同一锁文件，检查 Python 3.9／3.12 的基础环境与 CPU SFNO，并在 Python 3.11 的 CPU／kinematics／viz 配置执行 GMT 和两标签完整回归；另行检查格式、构建和仓库目录外的 wheel 安装。所有测试输入在临时目录生成，不读取个人训练数据。
