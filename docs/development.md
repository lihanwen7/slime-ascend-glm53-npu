# 开发指南

## 参数说明

slime 的参数分为三类：

1. **megatron 参数**：slime 会读取 `PYTHONPATH` 中 megatron 的所有参数，可通过 `--tensor-model-parallel-size 2` 等方式配置；
2. **sglang 参数**：支持环境中安装的 sglang 的所有参数，需以 `--sglang-` 起始传入，例如 `--sglang-mem-fraction-static`；
3. **slime 自身的参数**：见 [slime/utils/arguments.py](../slime/slime/utils/arguments.py)。

训练、保存与评估的具体指令和参数设定请直接参考 slime 主仓文档
[快速开始指南](https://github.com/THUDM/slime/blob/main/docs/zh/get_started/quick_start.md)。

## 补丁维护

本仓不直接修改提交上游 slime 代码，所有改动以 patch 文件管理。
补丁分类原则、添加/更新补丁的标准流程、上游同步策略，**以 [patches/README.md](../patches/README.md) 为准**；
逐 patch 的用途说明与上游跟踪信息见 [patches/MANIFEST.toml](../patches/MANIFEST.toml)。

要点速记：

```bash
bash build.sh                      # 构建出打满补丁的 slime/ 工作区
cd slime && git diff -- <file> > ../patches/<category>/xxx.patch
# 在 MANIFEST.toml 登记 [[patches.files]] 条目，并在 patch 头部加一行注释
bash build.sh                      # 重建验证全部补丁应用成功
```

## 调试

调试技巧请参考 slime [debug 指南](https://github.com/THUDM/slime/blob/main/docs/zh/developer_guide/debug.md)。
