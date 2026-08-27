# slime-ascend

slime 的昇腾适配开发仓（patch 管理版）

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

**slime-ascend** 是为 RL scaling 设计的 LLM post-training 框架 [slime](https://github.com/THUDM/slime) 的华为昇腾 NPU 适配版本。
上游 slime 以 git submodule 形式引入（`slime/`），所有 NPU 侧改动一律以 **patch 文件**管理，
由 `build.sh` 一键重置 submodule 并按序应用全部补丁，不直接提交任何上游代码改动。
关于 slime 本身的特性与介绍，请参考 [slime 原版文档](https://thudm.github.io/slime/)。

## 目录

- [硬件支持](#硬件支持)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [文档](#文档)
- [欢迎贡献](#欢迎贡献)
- [安全声明](#安全声明)
- [免责声明](#免责声明)
- [常见 Q&A 与致谢](#常见-qa-与致谢)

## 硬件支持

- **Atlas 800T A3**
- **Atlas 800T A2**
- 推荐配置：CANN 9.0.0 或更高版本；Ascend HDK 25.5.0 或更高版本
- 推荐环境：Linux arm64，Python 3.11

完整组件版本矩阵见 [快速使用指南 - 组件版本](docs/get_started/quick_start.md#组件版本)。

## 目录结构

```
slime-ascend/
├── build.sh              # 构建脚本：重置 submodule + 按序应用全部补丁 + 验证
├── docker/               # NPU Dockerfile（A3；A2 换基础镜像与 whl 即可）
├── docs/                 # 项目文档（快速开始、模型支持、开发指南、安全声明）
├── slime/                # THUDM/slime submodule，构建产物（补丁直接应用于此工作区）
├── patches/              # 全部改动，按性质分类（详见 patches/README.md）
│   ├── npu-compat/         # NPU 硬件差异适配（永久维护）：设备标识、环境变量、资源分配
│   ├── upstream-fixes/     # 上游 bug 的临时 workaround（上游修复后删除）
│   ├── features/           # NPU 侧新增能力
│   ├── third_party/        # 对依赖组件的补丁（sglang / Megatron-LM / Megatron-Bridge / mbridge / TE-NPU）
│   └── MANIFEST.toml       # 补丁清单：逐 patch 说明与上游跟踪信息
└── scripts/              # 安装脚本与运行示例
    ├── quick_install.sh    # 依赖环境安装 + third_party 补丁应用
    └── run-*.sh            # 各模型 NPU 运行示例（Qwen3、GLM4.7 等）
```

## 快速开始

```bash
# 1. 克隆（含 submodule）
git clone --recursive <this-repo>
# 或已克隆时：
git submodule update --init --recursive

# 2. 构建：重置 slime submodule 到 pin 的 commit 并应用全部补丁
bash build.sh

# 3. 安装
cd slime && pip install -e .
```

`build.sh` 选项：

| 选项 | 说明 |
|---|---|
| `--clean` | 删除 slime/ 目录后全量重建 |
| `--skip-submodule` | 跳过 submodule 重置，快速重打补丁（幂等，已应用的补丁自动跳过；也可用于离线构建） |
| `--dry-run` | 只打印将执行的操作 |

完整的 NPU 环境搭建（Docker / 一键安装脚本 quick_install.sh）与常见问题排查，
见 [昇腾（NPU）快速使用](docs/get_started/quick_start.md)。

## 文档

- [昇腾（NPU）快速使用](docs/get_started/quick_start.md) — 环境配置、组件版本、训练启动、问题排查
- [特性与模型支持](docs/supported_models.md) — 训练配置与各模型示例脚本、发布状态
- [开发指南](docs/development.md) — 参数说明、补丁维护、调试
- [patches/README.md](patches/README.md) — 补丁分类原则与维护流程（补丁维护的权威文档）
- [patches/MANIFEST.toml](patches/MANIFEST.toml) — 逐 patch 说明与上游跟踪信息

## 欢迎贡献

若有功能建议、性能调优或使用体验反馈，欢迎提交 Issue / PR。

调试技巧请参考 slime [debug 指南](https://github.com/THUDM/slime/blob/main/docs/zh/developer_guide/debug.md)。

## 安全声明

详见 [slime-ascend 安全声明](docs/SECURITYNOTE.md)，内容包括系统安全加固、运行用户建议、文件权限控制、运行/数据安全声明、远程代码执行安全声明、公开接口声明与通信安全加固等。

## 免责声明

### 致 slime-ascend 使用者

1. slime-ascend 提供的模型仅供您用于非商业目的。
2. 对于各模型，slime-ascend 平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的 License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用 slime-ascend 模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在 GitCode 提交 issue，我们将及时审视并解决。
4. MindSpeed 功能依赖的 Megatron 等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，MindSpeed 仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。

### 致数据集所有者

如果您不希望您的数据集在 slime-ascend 中的模型被提及，或希望更新 slime-ascend 中的模型关于您的数据集的描述，请在 GitCode 提交 issue，我们将根据您的 issue 要求删除或更新您的数据集描述。衷心感谢您对 slime-ascend 的理解和贡献。

### License 声明

slime-ascend 提供的模型，如模型目录下存在 License 的，以该 License 为准。如模型目录下不存在 License 的，以 Apache 2.0 许可证许可，对应许可证文本见 [slime/LICENSE](slime/LICENSE)。

## 常见 Q&A 与致谢

- 常见问题请见 slime [Q&A](https://github.com/THUDM/slime/blob/main/docs/zh/get_started/qa.md)
- 特别感谢以下项目 & 社区：SGLang、Megatron-LM、Megatron-Bridge、mbridge、OpenRLHF、veRL、Pai-Megatron-Patch、MindSpeed、昇腾社区等
