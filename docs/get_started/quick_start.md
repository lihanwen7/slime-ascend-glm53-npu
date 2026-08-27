# 昇腾（NPU）快速使用

本文档从搭建环境开始，带您快速上手 slime-ascend，涵盖 NPU 环境配置、补丁构建、训练启动和常见问题排查。

## 硬件支持说明

**slime-ascend** 支持华为昇腾 NPU 硬件平台：

- **Atlas 800T A3**
- **Atlas 800T A2**

**重要说明**：
- 请确保已正确安装 CANN 套件（推荐 CANN 9.0.0 或更高版本）
- 推荐使用 Linux arm64 环境进行部署
- 本版本使用 MegatronAdaptor 和 TransformerEngineNPU 代替 MindSpeed 进行 NPU 上 Megatron-LM 的适配支持

## 组件版本

Python 3.11：

| 组件 | A2 版本 | A3 版本 |
| ---- | ------- | ------- |
| 基础镜像 | Ubuntu 22.04 | Ubuntu 22.04 |
| Python | 3.11 | 3.11 |
| CANN | 9.0.0 | 9.0.0 |
| torch | 2.10.0 | 2.10.0 |
| torch_npu | 2.10.0 | 2.10.0 |
| torchvision | 0.25.0 | 0.25.0 |
| Megatron-LM | 1dcf0dafa | 1dcf0dafa |
| Megatron-Bridge | dev_rl | dev_rl |
| triton-ascend | 3.2.1 | 3.2.1 |
| mbridge | 89eb1088 | 89eb1088 |
| SGLang | v0.5.13 | v0.5.13 |
| sgl-kernel-npu | 2026.6.1 | 2026.6.1 |
| nvidia-modelopt | >=0.37.0 | >=0.37.0 |
| slime (submodule) | v0.3.0 | v0.3.0 |

> 组件版本以 [scripts/quick_install.sh](../../scripts/quick_install.sh) 和 [patches/MANIFEST.toml](../../patches/MANIFEST.toml) 为准；
> Python 3.12 + CANN 9.1.0 组合暂未在本仓验证。

## 使用指南

模型与训练配置（训推分离 / colocate / 全异步）的支持情况见 [特性与模型支持](../supported_models.md)。

## 环境配置

### 方式一：Docker（推荐）

我们提供了针对 A3 硬件的 [Dockerfile](../../docker/Dockerfile.a3.ubuntu22.04.cann90.latest)，可快速构建完整训练环境。

#### 镜像构建

```bash
# 进入项目根目录
cd {slime-ascend-root-path}

# 构建 A3 镜像（构建上下文为仓库根目录，Dockerfile 会 COPY 整个仓库）
docker build -f docker/Dockerfile.a3.ubuntu22.04.cann90.latest \
  -t slime-ascend:9.0.0-a3-ubuntu22.04-py3.11-dev .
```

> A2(910b) 请改用 `cann:9.0.0-910b-*` 基础镜像，并通过 `--build-arg SGL_KERNEL_NPU_URL=<910b whl 地址>` 覆盖。

#### 启动容器

```bash
# 启动容器（以 A3 为例）
docker run -it --rm \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
  -v $(pwd):/workspace \
  -w /workspace \
  slime-ascend:9.0.0-a3-ubuntu22.04-py3.11-dev
```

镜像构建过程会自动完成：依赖安装（等价于 quick_install.sh 的步骤 1-6）、
third_party 补丁应用（`git am`）、以及 `bash build.sh` 的 slime 补丁构建与安装。

> 声明：slime-ascend 中提供的 Ascend 相关 Dockerfile、镜像皆为参考样例，可用于尝鲜体验，如在生产环境中使用请通过官方正式途径沟通。

### 方式二：本地安装脚本

我们提供了[一键安装脚本](../../scripts/quick_install.sh)，可快速搭建 NPU 环境：

```bash
conda create -n slime-ascend python=3.11
conda activate slime-ascend

# 克隆仓库（含 slime submodule）
git clone --recursive https://gitcode.com/Ascend/slime-ascend.git
cd slime-ascend

# CANN 安装路径，请根据实际情况修改
export CANN_INSTALL_PATH=/usr/local/Ascend
# NPU 设备类型：A3 或 A2
export NPU_DEVICE=A3

# 运行安装脚本（脚本内部会加载 CANN 环境变量）
bash scripts/quick_install.sh
```

**安装脚本会自动完成以下工作**：

1. 从源码安装 SGLang（v0.5.13，NPU 配置）
2. 安装 torch、torch_npu、torchvision 等基础依赖包
3. 安装 sgl-kernel-npu（按 `NPU_DEVICE` 区分 A2/A3）
4. 安装 mbridge、Megatron-Bridge、Megatron-LM
5. 安装 MegatronAdaptor 和 TransformerEngineNPU
6. 安装 triton_ascend
7. 对 sglang / Megatron-LM / Megatron-Bridge / mbridge / TE-NPU 应用 `patches/third_party/` 补丁（`git am`）
8. 执行 `bash build.sh`：重置 slime submodule 并应用全部 slime 补丁

> 注意：请确保使用本仓提供的 quick_install.sh 并从上表指定的分支/commit 源码安装。

### 补丁构建（build.sh）

`build.sh` 是本仓的核心：上游 slime 以 submodule 引入，所有 NPU 改动以 patch 管理。

```bash
bash build.sh                  # 重置 submodule + 按序应用全部补丁 + 验证
bash build.sh --skip-submodule # 离线/快速重建：跳过重置，幂等重打补丁
bash build.sh --clean          # 删除 slime/ 全量重建
```

构建完成后安装：

```bash
cd slime && pip install -e .
```

## 训练脚本与参数概览

完成上述准备工作后，即可运行 NPU 训练脚本，示例见 `scripts/run-*.sh`（对应关系见
[特性与模型支持](../supported_models.md)）。参数说明与调优见 [开发指南](../development.md)。

## 常见问题排查

### NPU 相关问题

1. **CANN 环境未正确加载**
   - 确保已执行 `source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh`
   - 检查 `ASCEND_TOOLKIT_HOME` 等环境变量是否正确设置

2. **NPU 设备不可见**
   - 检查 `ASCEND_RT_VISIBLE_DEVICES` 环境变量
   - 使用 `npu-smi info` 命令验证 NPU 设备状态

3. **显存不足**
   - 调整 `--sglang-mem-fraction-static` 参数（建议 0.6-0.8）
   - 减少 batch size 或并行度

4. **HCCL 通信错误**
   - 检查 `HCCL_HOST_SOCKET_PORT_RANGE` 和 `HCCL_NPU_SOCKET_PORT_RANGE` 配置
   - 确保端口范围内的端口未被占用

5. **补丁应用失败 / 离线构建**
   - 网络不可达时 `build.sh` 的 `git fetch` 会失败，可改用 `bash build.sh --skip-submodule`
   - 补丁冲突时进入 `slime/` 检查 `*.rej` 文件，解决后重新生成对应 patch

更多常见问题请参考 slime [Q&A](https://github.com/THUDM/slime/blob/main/docs/zh/get_started/qa.md)
