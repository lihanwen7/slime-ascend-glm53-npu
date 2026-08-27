# slime-ascend 补丁目录

本目录包含 slime-ascend 相对于上游 THUDM/slime 的所有改动，按性质分类。

## 目录结构

```
patches/
├── npu-compat/       # NPU 硬件差异性适配（永久性）
├── upstream-fixes/   # 上游 bug 的临时 workaround（预期上游修复后删除）
├── features/         # NPU 侧新增能力
└── MANIFEST.toml     # 补丁清单和元数据
```

## 补丁分类

### npu-compat/ - NPU 硬件兼容性适配

**特点**：
- 永久性维护
- 与上游版本无关的硬件差异
- 主要涉及设备标识、环境变量、资源分配

**示例**：
- Ray 资源分配从 `num-gpus` 到 NPU 资源
- 环境变量从 `CUDA_VISIBLE_DEVICES` 到 `ASCEND_RT_VISIBLE_DEVICES`
- 设备同步从 `torch.cuda` 到 `torch.npu`

### upstream-fixes/ - 上游临时修复

**特点**：
- 临时性维护
- 预期上游修复后删除
- 包含模型支持扩展、bug 修复等

**示例**：
- `model_provider.patch` - 扩展模型支持

**维护策略**：
- 每次同步上游时检查是否已修复
- 已修复的补丁从此目录删除
- 在 `MANIFEST.toml` 中跟踪上游 issue

### features/ - NPU 新增能力

**特点**：
- 长期维护
- NPU 特有的功能增强
- 可能未来贡献回上游

**示例**：
- FSDP backend for NPU
- NPU 特定的优化特性

## 应用补丁

补丁由 `build.sh` 自动应用：

```bash
# 构建流程
./build.sh
```

构建脚本会：
1. Checkout `slime` submodule 到指定 commit
2. 按顺序应用 `npu-compat`、`upstream-fixes`、`features` 补丁（新增文件也以 new-file patch 形式管理）
3. 生成可用的 slime-ascend

## 维护补丁

### 添加新补丁

```bash
# 1. 在 slime 工作区中修改文件
cd slime
# ... 修改代码 ...
git diff > ../patches/npu-compat/new_feature.patch

# 2. 更新 MANIFEST.toml
# 在对应分类下追加 [[patches.files]] 条目：file + summary（临时补丁加 upstream 跟踪字段）

# 3. 在 patch 文件头部加一行注释，格式：
# # [分类] 用途说明。详见 patches/MANIFEST.toml
```

### 更新现有补丁

```bash
# 1. 应用所有补丁到 slime
./build.sh

# 2. 修改文件
cd slime
# ... 修改代码 ...

# 3. 重新生成补丁
git diff > ../patches/npu-compat/existing.patch
```

### 同步上游

```bash
# CI 会自动执行，也可手动测试
cd slime
git fetch origin
git checkout v0.4.0  # 新版本

# 尝试应用补丁
cd ..
./build.sh

# 解决冲突并更新补丁
```
