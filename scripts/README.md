当前正式入口在内层Slime：

- 单机：`/mnt/share/l00804615/glm5.3_flash_native/slime-ascend/slime/scripts/run-glm5.3-flash-300B-npu.sh`
- 八机：`/mnt/share/l00804615/glm5.3_flash_native/slime-ascend/slime/scripts/run-glm5.3-flash-300B-8nodes-npu.sh`

旧单/双机入口及历史辅助体系已归档，位置见项目cache/archive目录下retired_launchers归档。

本目录目前只暂留scripts/lib/glm53/runtime中的4个兼容/分析模块：25号机现有进程还在引用这些路径，不能直接移走。
功能代码迁往slime_plugins/glm53_runtime的候选已在仓库外准备，尚未部署。当前目录组织仍有这一项AGENTS.md未完成事项。
