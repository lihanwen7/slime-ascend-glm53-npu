八机入口：`/mnt/share/l00804615/glm5.3_flash_native/slime-ascend/lh_scripts/run_glm53_8node_full.sh`。

本配方参考 [Slime 官方 GLM-5.3-Flash 脚本](https://github.com/THUDM/slime/blob/main/scripts/run-glm5.3-flash-300B.sh) 的模型配置复用和参数组织，并参考当前 Ascend slime-ascend 基线 `282b6eb6409a0821293400fba3ddfe52fb5f9e01` 的 NPU 参数分组。官方 GLM 脚本面向已有 H100 Ray 集群；本地脚本保留同一 run ID 在八台机器上分别启动、21 号机提交的运维接口。只参考结构，没有套用官方 GPU 参数或重设 RL 算法。

**文件职责**

| 文件 | 职责 |
|---|---|
| `/mnt/share/l00804615/glm5.3_flash_native/slime-ascend/lh_scripts/run_glm53_8node_full.sh` | 节点表、路径；CKPT / ROLLOUT / PERF / OPTIMIZER / SGLANG / MISC 参数 |
| `/mnt/share/l00804615/glm5.3_flash_native/slime-ascend/slime/scripts/models/glm5.3-flash-300B.sh` | 复用现有 GLM 模型架构配置，不再在八机入口复制一份 |
| `/mnt/share/l00804615/glm5.3_flash_native/slime-ascend/runtime/lh-eight-node-env.sh` | 公共平台环境、SDK 初始化、每节点 SDK 版本发现视图 |
| `/mnt/share/l00804615/glm5.3_flash_native/slime-ascend/runtime/lh-eight-node-launch.sh` | 参数解析、数据/权重检查、Ray 集群生命周期、精确节点检查、自有进程组收尾 |

八机入口不再 source 单机的 lh-ascend-runtime.sh、lh-single-node-launch.sh 或 host.env。单机入口及其 runtime 没有修改。已有 Ray SIGINT/TMS 兼容模块继续通过原 PYTHONPATH 引用，本次没有移动或重写它们。

**查看和检查**

```bash
cd /mnt/share/l00804615/glm5.3_flash_native/slime-ascend
bash lh_scripts/run_glm53_8node_full.sh --help
bash lh_scripts/run_glm53_8node_full.sh --print-command
bash lh_scripts/run_glm53_8node_full.sh --print-env
bash lh_scripts/run_glm53_8node_full.sh --check
```

`--help`、`--print-command`、`--print-env` 不激活 Conda/CANN、不导入 torch、不扫描数据、不启动 Ray、不执行 npu-smi。`--check` 用系统 Python 检查 full288 配置、索引分片存在性和全量 JSONL prompt/label 字段；它会读取完整数据，不能替代权重哈希、数值或训练验收。实际运行也保留这一检查。

路径可在入口顶部直接阅读。HF_CHECKPOINT、PROMPT_DATA、LOG_ROOT、NUM_ROLLOUT、RM_TYPE 可由同名变量设置；八台必须使用一致设置。DEPS_ROOT/CONDA_ENV/CANN_ROOT 仅用于安装位置适配，实际 activate.sh 必须与配置指向同一环境。本配方只接受45层、288专家模型；top16另行调整模型配方，不会自动按目录名猜测。

**实际启动接口**

在确认资源可用后，八台各执行同一条命令、使用同一个新 run ID；以下只是用法示例，不表示已授权当前资源窗口：

```bash
bash lh_scripts/run_glm53_8node_full.sh --run-id 20261028_120000_full288
```

节点表已经改为本组通信 IP `197.13.21.1`—`197.13.28.1`，网卡 enp39s0f0。原来指向197.13.37—44的表不再保留。21号机等待精确8节点/64 NPU，发现其他节点直接失败；各节点有NPU进程时拒绝启动，不清理外部任务。Ray停止只针对本次启动的进程组。

默认产物路径为 `/mnt/share/l00804615/glm5.3_flash_native/cache/work/glm53-eight-node/<run-id>/node21` 至 node28。与原来写入仓库内 lh_logs_rh 的默认位置相比，这里改为项目 cache/work。run ID 推荐时间戳开头。每节点本地临时路径使用run ID的12位SHA前缀，避免Ray Unix socket路径超长；同一个run ID的已有目录仍会被拒绝，不会覆盖或自动清除。

**CANN 私有清单：显式参数**

默认使用共享CANN原清单。2026-10-27 的成功通信/训练启动还依赖一份排除optional HYBM包的私有清单，原因是21号机的包签名检查失败。需要复用该条件时，八台都显式追加：

```bash
--cann-package-manifest /mnt/share/l00804615/glm5.3_flash_native/cache/work/20261027_comm64_exclusive_lhw/b237_comm_config_v2/conf/ascend_package_load.ini
```

该参数只在本进程的私有 mount namespace 中只读绑定指定清单，挂载失败则终止，不改共享清单、不改设备验签设置。它没有修复或验收HYBM。历史实验日期路径没有硬编码为默认配置。已有私有 mount 包装器仍可继续使用；也可以直接使用此显式参数。此次验证没有实际执行unshare或mount。

SDK shallow view保留了原来的NFS导入兼容行为，但由原先两次创建改成每次运行一次。目录位于任务的本地TMPDIR，其中链接仅引用原SDK库，并复制ascend_toolkit_install.info供版本发现；没有用软链接组织项目源码。基础activate.sh仍只负责原有SDK激活。

**训练配置：保持旧默认值**

full288、45层；训练TP/PP/CP/EP=1/4/2/16；4个TP16推理引擎；16题×8响应/global128/micro1；prompt1024/response32768/context65536；动态packing预算2048；SGLang静态内存0.55、prefill256、max-running64；Adam lr1e-6、CPU offload、no-pin-cpu-grads；MoE分块512/1024；routing replay。仍未完成完整训练验收，本次整理不改变该结论。

**公共环境变量：值、消费方与原因**

这些值集中在lh-eight-node-env.sh的显式数组中，每行有注释。它们是当前环境兼容配置，不等于每项都已通过单独消融证明不可删除。

| 变量 | 值 / 作用 | 消费方或依据 |
|---|---|---|
| SLIME_ACCELERATOR | npu，明确设备后端 | Slime accelerator |
| ASCEND_RT_VISIBLE_DEVICES | 0—7，每节点8卡 | torch_npu / Slime |
| RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES | 1，保留Slime的设备分配 | Ray / slime.ray.utils |
| RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO | 0，zero-GPU actor不改写设备映射 | Ray；Slime NPU command_utils同值 |
| USE_LIBUV | 0，保留TCPStore旧后端 | PyTorch TCPStore；沿用已有配置，未单独消融 |
| HCCL_BUFFSIZE | 256 MiB，通信buffer | Slime command_utils、SGLang parallel_state、native DeepEP |
| HCCL_HOST_SOCKET_PORT_RANGE | 9000—9099，rollout主机端口 | HCCL；训练另用9200—9299 |
| HCCL_NPU_SOCKET_PORT_RANGE | 9100—9199，rollout设备端口 | HCCL；训练另用9300—9399 |
| HCCL_CONNECT_TIMEOUT / HCCL_EXEC_TIMEOUT / ACL_DEVICE_SYNC_TIMEOUT | 各3600秒，保留首步JIT时间预算 | CANN/HCCL |
| TASK_QUEUE_ENABLE | 0，关闭Ascend host task queue | torch_npu；训练actor也显式传入 |
| PYTORCH_NPU_ALLOC_CONF | expandable_segments:False，保留TMS/offload分配模式 | torch_npu allocator；没有重新做True/False消融 |
| TMS_HOOK_MODE | preload，native allocator拦截 | torch_memory_saver.entrypoint |
| TMS_PRESERVE_PRELOAD_MODE | 1，保留preload模式 | 已修改的torch_memory_saver.entrypoint |
| SGLANG_MEMORY_SAVER_CUDA_GRAPH | true，在NPU图路径使用TMS pool | SGLang NPU graph backend；名字含CUDA但NPU代码确实读取 |
| SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT | 1，保留ND权重格式便于更新 | SGLang NPU utils |
| ASCEND_USE_FIA | 1，Ascend fused attention路径 | SGLang ascend_backend / NPU graph / memory_pool |
| SGLANG_ENABLE_JIT_DEEPGEMM | False，避免CUDA DeepGEMM JIT | SGLang environ / deep_gemm_wrapper |
| SGLANG_OPT_DEEPGEMM_HC_PRENORM | 0，沿用现有mHC选择 | SGLang mhc.py；未在本次重构切换算子路径 |
| SGLANG_OPT_USE_TILELANG_MHC_PRE / POST | 各0，沿用mHC前后处理配置 | SGLang mhc.py；A5代码有自身分派，但本次不以静态分派代替完整消融 |
| SGLANG_RUST_BUILD_MODE | never，使用已安装扩展 | SGLang Rust扩展加载 |
| GLM53_R3_DIRECT_WORKERS | 1，保留routed_experts响应字段 | 已修改的sglang_rollout.py |
| GLM53_RAY_SIGINT_NONE_COMPAT | 1，已有Ray SIGINT None兼容 | glm53_ray_sigint_compat.py，保留owner/cwd保护 |
| TORCH_DEVICE_BACKEND_AUTOLOAD | 0，避免SDK未准备时自动导入torch_npu | PyTorch backend初始化 |
| FLA_NPU_DISABLE_PTH | 1，改为显式激活FLA vendor | 安装环境的fla_npu_opp_env.py |
| PYTHONNOUSERSITE | 1，避免混入用户site包 | CPython |
| PYTHONDONTWRITEBYTECODE | 1，避免并发写NFS源码缓存 | CPython |
| OMP_NUM_THREADS | 4，每进程CPU线程预算 | OpenMP；官方GLM脚本同值 |
| TOKENIZERS_PARALLELISM | false，避免嵌套tokenizer线程池 | tokenizers |
| CUDA_DEVICE_MAX_CONNECTIONS | 1，保留Megatron通用配置 | 官方Slime/Ascend脚本有同值；NPU单独必要性未证明 |

GLOO_SOCKET_IFNAME/HCCL_SOCKET_IFNAME来自唯一的FULL_COMM_IFNAME；SLIME_HOST_IP只设置为本节点通信IP，移除了127.0.0.1的中间值。代理设置、PYTHONPATH/LD_LIBRARY_PATH和SDK变量由激活过程处理，不把head节点的IP、临时目录或cache路径作为全局Ray runtime-env分发给其他节点。

HCCL_OP_EXPANSION_MODE显式unset，使用驱动默认模式；去掉了先设置AICPU_TS再unset的往返覆盖。

**只作用于训练actor的环境**

以下6项在入口的TRAIN_ENV_JSON里完整列出，不再通过Python逐项修改隐藏JSON：

| 变量 | 值 | 原因 |
|---|---|---|
| HCCL_HOST_SOCKET_PORT_RANGE | 9200—9299 | 和rollout分离 |
| HCCL_NPU_SOCKET_PORT_RANGE | 9300—9399 | 和rollout分离 |
| GLM53_TMS_POST7_RELEASE_COMPAT | 1 | 已有精确版本/文件SHA保护的post7 pool release兼容，只给训练worker |
| GLM53_MOE_DISPATCH_CHUNK_SIZE | 512 | moe_chunking.py的dispatch workspace限制 |
| GLM53_EXPERT_TOKEN_CHUNK_SIZE | 1024 | moe_chunking.py的expert token分块 |
| TASK_QUEUE_ENABLE | 0 | 明确传给训练worker |

**移除、合并与接口变化**

| 旧内容 | 处理与依据 |
|---|---|
| GLM53_NATIVE_HC_SHIM / SGLANG_NPU_HC_PRE_TAIL_GUARD / GLM53_NATIVE_ROOT | 移除旧launcher声明；在当前tracked模型代码、runtime、custom_ops及cann-custom-ops检索未找到读取点；不把名字当作已经生效的修复 |
| NATIVE_MIN_HBM_FREE_MB / NATIVE_ALLOW_WARNING_HEALTH | 移除；旧八机/被引用的单机启动逻辑没有使用它们做判定；占卡拒绝检查保留 |
| PHYSICAL_LOG_ROOT / ATB_SET_ENV / GLM53_RUNTIME_DIR / EXPERIMENT_NAME | 移除八机链路中的闲置别名或未使用变量 |
| DEEPEP_HCCL_BUFFSIZE=256 | 合并到HCCL_BUFFSIZE=256；所查SGLang parallel_state和native DeepEP读取显式值否则回退到HCCL_BUFFSIZE，当前buffer数值不变 |
| CP_SIZE / MAX_CONTEXT_LEN等脚本内只读别名 | 参数组直接写旧数值，避免多层转发 |
| GLM53_TMS_POST7_RELEASE_COMPAT=train-only | 移除全局哨兵；只保留TRAIN_ENV_JSON中的1 |
| GLM53_MOE_*两个shell覆盖开关 | 改成TRAIN_ENV_JSON中直接可见的512/1024；要调实验值直接改训练配方，默认值未变 |
| TRAIN_ENV_JSON来自调用者时的隐式合并 | 改为配方内显式完整JSON，避免继承不明actor环境 |
| 模型架构重复块 | 复用已有models/glm5.3-flash-300B.sh |
| --print-command先激活SDK并全扫数据 | 改成只打印；完整输入检查仍在--check与实际run中 |
| 默认日志位置 | 从仓库内lh_logs_rh改为项目cache/work |

因此本次保证的是默认120项训练参数与保留的公共变量数值一致，不承诺任意旧shell环境覆盖方式继续有效。删除的选择性变量只做了静态消费点检查，没有新的NPU消融证据。

验证：bash语法；120项Bash实际展开训练参数等价（含训练JSON）；保留的32项公共环境值等价；模拟head成功/失败、worker成功、意外节点拒绝、有NPU占用拒绝、退出状态传播和自有Ray组退出；检查模式无SDK/Ray/NPU副作用。真实SDK导入、CANN私有mount、实际Ray多机、算子数值、msSanitizer和完整训练均没有在本次执行。
