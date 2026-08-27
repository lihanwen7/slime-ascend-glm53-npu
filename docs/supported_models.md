# 特性与模型支持

## 训练配置

目前在 NPU 上支持以下训练配置：

- **标准训推分离**
- **colocate**（训推同卡）
- **全异步（fully async）**

## 模型支持

| 训练算法 | 支持模型 | 示例脚本 | 发布状态 |
| -------- | ------- | -------- | -------- |
| GRPO | GLM-4.7-Flash (30B-A3B) | [run-glm4.7-30B-A3B-npu.sh](../scripts/run-glm4.7-30B-A3B-npu.sh) / [128k 长序列](../scripts/run-glm4.7-30b-A3B-npu-128k.sh) | Preview |
| GRPO | Qwen3-8B | [run-qwen3-8B-npu.sh](../scripts/run-qwen3-8B-npu.sh) / [colocate](../scripts/run-qwen3-8B-npu-colocate.sh) / [全异步](../scripts/run-qwen3-8b-fully_async-npu.sh) | Preview |
| GRPO | Qwen3-30B-A3B | [run-qwen3-30B-A3B-npu.sh](../scripts/run-qwen3-30B-A3B-npu.sh) | Preview |

注意："Preview" 发布状态表示预览非正式发布版本，"Released" 发布状态表示正式发布版本。

新验证一个模型后，请在此表中补充对应行，并保持与 `scripts/` 下实际示例脚本一致。
