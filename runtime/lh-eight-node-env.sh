#!/usr/bin/env bash
# Eight-node environment only. No single-node launcher or host.env is sourced.
# Source defines functions/data only; --help/--print-* never activate the SDK.

GLM53_EIGHT_COMMON_ENV=(
    SLIME_ACCELERATOR=npu                         # Slime accelerator selection.
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     # The eight physical NPUs per node.
    RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1  # Slime assigns NPU IDs.
    RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0          # Keep IDs for zero-GPU Ray actors.
    USE_LIBUV=0                                  # TCPStore: keep the legacy backend.
    HCCL_BUFFSIZE=256                            # HCCL communicator buffer, MiB.
    HCCL_HOST_SOCKET_PORT_RANGE=9000-9099          # Rollout host ports; train uses 9200.
    HCCL_NPU_SOCKET_PORT_RANGE=9100-9199           # Rollout NPU ports; train uses 9300.
    HCCL_CONNECT_TIMEOUT=3600                     # Seconds, including first-step JIT.
    HCCL_EXEC_TIMEOUT=3600
    ACL_DEVICE_SYNC_TIMEOUT=3600
    TASK_QUEUE_ENABLE=0                          # Ascend host task queue, same as baseline.
    PYTORCH_NPU_ALLOC_CONF=expandable_segments:False # Existing TMS/offload allocation mode.
    TMS_HOOK_MODE=preload                         # Native allocator interception mode.
    TMS_PRESERVE_PRELOAD_MODE=1                   # Do not silently switch to torch hooks.
    SGLANG_MEMORY_SAVER_CUDA_GRAPH=true           # Manage graph pools through TMS on NPU.
    SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1        # Keep transferable weights in ND format.
    ASCEND_USE_FIA=1                             # Ascend fused attention path.
    SGLANG_ENABLE_JIT_DEEPGEMM=False              # Do not build CUDA DeepGEMM on NPU.
    SGLANG_OPT_DEEPGEMM_HC_PRENORM=0              # Keep the previous native mHC selection.
    SGLANG_OPT_USE_TILELANG_MHC_PRE=0
    SGLANG_OPT_USE_TILELANG_MHC_POST=0
    SGLANG_RUST_BUILD_MODE=never                  # Use installed Rust extension, no build.
    GLM53_R3_DIRECT_WORKERS=1                     # Preserve routed_experts at the gateway.
    GLM53_RAY_SIGINT_NONE_COMPAT=1                # Existing guarded Ray SIGINT hook.
    TORCH_DEVICE_BACKEND_AUTOLOAD=0               # SDK setup precedes torch_npu import.
    FLA_NPU_DISABLE_PTH=1                        # Explicit FLA vendor activation below.
    PYTHONNOUSERSITE=1                            # Avoid mixing user-site packages.
    PYTHONDONTWRITEBYTECODE=1                    # Avoid concurrent writes to NFS source.
    OMP_NUM_THREADS=4                            # Bound CPU threads per worker.
    TOKENIZERS_PARALLELISM=false                  # Avoid nested tokenizer thread pools.
    CUDA_DEVICE_MAX_CONNECTIONS=1                # Existing Megatron ordering setting.
)

glm53_eight_activate() {
    local item
    for item in "${GLM53_EIGHT_COMMON_ENV[@]}"; do export "$item"; done
    # The selected node supplies these; never forward head-local paths/IPs to peers.
    export GLOO_SOCKET_IFNAME="$FULL_COMM_IFNAME" HCCL_SOCKET_IFNAME="$FULL_COMM_IFNAME"
    export SLIME_HOST_IP="$node_ip"
    export no_proxy="localhost,127.0.0.1,$(IFS=,; echo "${NODE_IPS[*]}")"
    export NO_PROXY="$no_proxy"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy RAY_ADDRESS
    # Ray starts non-interactive bash workers. SSH markers can make bash source
    # the host bashrc and reactivate its base Conda environment after setup.
    unset SSH_CLIENT SSH_CONNECTION SSH_TTY
    # The baseline uses the driver's default HCCL mode, not single-node AICPU_TS.
    unset HCCL_OP_EXPANSION_MODE GLM53_TMS_POST7_RELEASE_COMPAT
    # Remove stale SDK/Python paths before activating this one shared environment.
    unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD ASCEND_CUSTOM_OPP_PATH ASCEND_OPP_PATH
    unset ASCEND_AICPU_PATH ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_CANN_PACKAGE_PATH
    unset ASCEND_HOME ASCEND_ATB_HOME ATB_HOME_PATH ASDSIP_HOME_PATH
    set +u
    source "$DEPS_ROOT/activate.sh"
    source "$CONDA_ENV/lib/python3.12/site-packages/fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash"
    set -u
    # Exactly one shallow SDK view, instead of the previous two. This preserves
    # torch_npu version discovery without recursively walking the SDK over NFS.
    # Links are SDK library references in the task's /tmp, not source organization.
    /usr/bin/python3 -I -S - "$CANN_ROOT" "$TMPDIR/cann" <<'PY_CANN'
import pathlib, shutil, sys
source, view = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
view.mkdir()
for entry in source.iterdir():
    (view / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
shutil.copyfile(source / 'aarch64-linux/ascend_toolkit_install.info', view / 'ascend_toolkit_install.info')
PY_CANN
    export ASCEND_HOME_PATH="$TMPDIR/cann"
    export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
    # Preserve the SDK Python packages exported by CANN activation.
    export PYTHONPATH="$DEPS_ROOT/slime-ascend/scripts/lib/glm53/runtime:$SLIME_ROOT:$DEPS_ROOT/Megatron-LM:$DEPS_ROOT/MegatronAdaptor:$DEPS_ROOT/TransformerEngineNPU:$DEPS_ROOT/mbridge:$DEPS_ROOT/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
}
