# GLM5.3 Flash native

本目录使用原有 Conda 环境和 CANN 9.2，不安装另一套环境。

```bash
cd /mnt/share/l00804615/glm5.3_flash_native
source ./activate.sh
```

`activate.sh` 完成 Conda 激活及 CANN、ATB、DeepEP 环境加载。激活本身不运行模型或探测设备。

## 运行入口

- 单机：`/mnt/share/l00804615/glm5.3_flash_native/slime-ascend/slime/scripts/run-glm5.3-flash-300B-npu.sh`
- 八机：`/mnt/share/l00804615/glm5.3_flash_native/slime-ascend/slime/scripts/run-glm5.3-flash-300B-8nodes-npu.sh`

单机以用户指定的21号脚本为参数标准；RAY_NODE_IP默认值仍为197.13.37.1，换主机时应明确设置。
八机可用--print-command和--print-env只读查看配置；单机--check会激活SDK并导入依赖。
日志和profiling默认输出在项目cache/work，历史日志/调试材料归档在cache/archive。
当前运行任务仍引用外层scripts/lib/glm53/runtime，模块迁移候选尚未部署，不能把目录合规性判为全部完成。

## SGLang 源码与 Git

`sglang/` 是普通目录，保留真实私仓 `.git`，不使用 `sglang-copy`、`sglang-bootstrap` 或另一份源码链接。

- 私仓：`pengxingchen0810/glmx`
- 分支：`glmx-lwm-0826-A5`，跟踪 `glmx/glmx-main-a3-a5-merge`
- 基点：`872f1721cc6d4679f008167386ec6699a6580102`
- 本地 Ascend 适配差异保留为工作区修改，可用 `git diff` 检查，没有伪造提交。

```bash
git -C sglang status --short
git -C sglang log -8 --oneline
git -C sglang diff
```

Git 历史和本地操作已验收。私仓网络认证按机器独立管理；不随环境分发私钥或访问令牌。
