
# Prevent Conda/Python startup hooks from initializing a backend before SDK setup.
export TORCH_DEVICE_BACKEND_AUTOLOAD FLA_NPU_DISABLE_PTH PYTHONNOUSERSITE
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD ASCEND_CUSTOM_OPP_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_CANN_PACKAGE_PATH ASCEND_HOME ASCEND_ATB_HOME ATB_HOME_PATH ASDSIP_HOME_PATH
set +u
source "$DEPS_ROOT/activate.sh"
source "$FLA_PYTHON_ROOT/fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash"
# FLA 26.10 embeds Conv1d. Registering the legacy private vendor first
# mixes its kernels with the new op_api and fails workspace creation (561002).
set -u
# TorchNPU walks ASCEND_HOME_PATH recursively during import. Keep an equivalent
# local directory view so version discovery does not walk the SDK over NFS.
ASCEND_HOME_PATH=$(/usr/bin/python3 -I -S - "$CANN_ROOT" <<'PY_CANN_VIEW'
import pathlib,shutil,sys,tempfile
source=pathlib.Path(sys.argv[1]).resolve()
view=pathlib.Path(tempfile.mkdtemp(prefix='glm53-cann-'))
for entry in source.iterdir():
    (view/entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
shutil.copyfile(source/'aarch64-linux/ascend_toolkit_install.info',view/'ascend_toolkit_install.info')
print(view)
PY_CANN_VIEW
)
export ASCEND_HOME_PATH
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$DEPS_ROOT/slime-ascend/scripts/lib/glm53/runtime:$SLIME_ROOT:$DEPS_ROOT/Megatron-LM:$DEPS_ROOT/MegatronAdaptor:$DEPS_ROOT/TransformerEngineNPU:$DEPS_ROOT/mbridge:$DEPS_ROOT/sglang/python:${PYTHONPATH:-}"
export GLM53_NATIVE_ROOT="$DEPS_ROOT"
# New native training workers use the validated CANN mHC path; torch remains an A/B override.
export GLM53_MHC_BACKEND="${GLM53_MHC_BACKEND:-ascendc}"
export GLM53_NATIVE_HC_SHIM SGLANG_NPU_HC_PRE_TAIL_GUARD
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy RAY_ADDRESS
export no_proxy="localhost,127.0.0.1,$RAY_NODE_IP" NO_PROXY="localhost,127.0.0.1,$RAY_NODE_IP"
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Keep optional Rust extensions from triggering a toolchain build at startup.
export SGLANG_RUST_BUILD_MODE=never
export TMS_HOOK_MODE TMS_PRESERVE_PRELOAD_MODE
export SLIME_ACCELERATOR USE_LIBUV SGLANG_MEMORY_SAVER_CUDA_GRAPH GLM53_RAY_SIGINT_NONE_COMPAT SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT SLIME_HOST_IP
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO
export HCCL_OP_EXPANSION_MODE HCCL_BUFFSIZE DEEPEP_HCCL_BUFFSIZE HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT ACL_DEVICE_SYNC_TIMEOUT
export PYTORCH_NPU_ALLOC_CONF SGLANG_OPT_DEEPGEMM_HC_PRENORM SGLANG_OPT_USE_TILELANG_MHC_PRE SGLANG_OPT_USE_TILELANG_MHC_POST SGLANG_ENABLE_JIT_DEEPGEMM
export ASCEND_USE_FIA PYTHONDONTWRITEBYTECODE OMP_NUM_THREADS TOKENIZERS_PARALLELISM ASCEND_RT_VISIBLE_DEVICES
# Values are defined once in host.env; export them to Ray and its workers.
export GLOO_SOCKET_IFNAME HCCL_SOCKET_IFNAME
export HCCL_HOST_SOCKET_PORT_RANGE HCCL_NPU_SOCKET_PORT_RANGE
# This compatibility flag is injected into training workers via TRAIN_ENV_JSON only.
unset GLM53_TMS_POST7_RELEASE_COMPAT

glm53_submit() {
    if [[ ${1:-} == --help || ${1:-} == -h ]]; then
        echo 'Usage: bash lh_scripts/run_glm53_single_node.sh [--check]'; return
    fi
    if [[ ${1:-} == --check ]]; then
        "$CONDA_ENV/bin/python" - <<'PY_CHECK'
import os,importlib
for key in ('CONDA_PREFIX','PYTHONPATH','ASCEND_HOME_PATH','ASCEND_CUSTOM_OPP_PATH','GLOO_SOCKET_IFNAME','HCCL_SOCKET_IFNAME'):
    print(key+'='+os.environ.get(key,''),flush=True)
for name in ('torch','torch_npu','ray','slime','sglang','megatron','custom_ops'):
    m=importlib.import_module(name)
    print(name,getattr(m,'__version__',''),getattr(m,'__file__',''),flush=True)
print('Environment imports passed',flush=True)
PY_CHECK
        return
    fi
    [[ $# == 0 ]] || { echo 'Unexpected launcher arguments' >&2; return 2; }
    ulimit -c 0
    ulimit -n 65535
    mkdir -p "$LOG_ROOT" "/tmp/glm53-lh-$UID"
    RUN_DIR=$(mktemp -d "$LOG_ROOT/$(date +%Y%m%d_%H%M%S)_$(hostname -s)_XXXXXX")
    local run_id="$(basename "$RUN_DIR")"
    exec > >(tee -a "$RUN_DIR/stdout.log") 2>&1
    echo "[glm53] Run directory: $RUN_DIR"
    exec 9>"/tmp/glm53-lh-$UID/single-node.lock"
    flock -n 9 || { echo 'Another lh single-node run holds the lock'; return 75; }
    npu-smi info > "$RUN_DIR/npu-before.txt"
    if grep -Eq '\|[[:space:]]*[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+[[:space:]]*\|' "$RUN_DIR/npu-before.txt"; then
        echo 'NPU processes are present; refusing to overlap another job'; return 75
    fi
    export TMPDIR=$(mktemp -d /tmp/g53lh.XXXXXX)
    export RAY_TMPDIR="$TMPDIR/ray" TRITON_CACHE_DIR="$RUN_DIR/triton" TORCH_EXTENSIONS_DIR="$RUN_DIR/torch-extensions"
    export HF_MODULES_CACHE="$RUN_DIR/hf_modules"
    mkdir -p "$RAY_TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$HF_MODULES_CACHE"
    printf '%q ' "$CONDA_ENV/bin/python" -u "$SLIME_ROOT/train.py" "${ARGS[@]}" > "$RUN_DIR/command.sh"
    printf '\n' >> "$RUN_DIR/command.sh"
    "$CONDA_ENV/bin/python" - <<'PY_ENV' > "$RUN_DIR/runtime-env.json"
import os,json
keys={'PATH','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD','SLIME_ACCELERATOR','SLIME_HOST_IP','CUDA_DEVICE_MAX_CONNECTIONS','USE_LIBUV','no_proxy','NO_PROXY','OMP_NUM_THREADS','TOKENIZERS_PARALLELISM','PYTHONDONTWRITEBYTECODE','PYTHONNOUSERSITE','TORCH_DEVICE_BACKEND_AUTOLOAD','TMPDIR','HF_MODULES_CACHE','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR'}
prefix=('ASCEND','HCCL','GLOO','RAY_','PYTORCH_','SGLANG_','TMS_','DEEPEP_','CANN','ATB','FLA_','GLM53_')
print(json.dumps({'worker_process_setup_hook':'glm53_ctor_setup_v5.install','env_vars':{k:v for k,v in os.environ.items() if k in keys or k.startswith(prefix)}}))
PY_ENV
    # Ray workers inherit the head process working directory.
    cd "$SLIME_ROOT"
    setsid "$CONDA_ENV/bin/ray" start --head --block --node-ip-address="$RAY_NODE_IP" --port="$RAY_GCS_PORT" \
        --dashboard-host=127.0.0.1 --dashboard-port="$RAY_DASHBOARD_PORT" --object-manager-port="$RAY_OBJECT_MANAGER_PORT" --node-manager-port="$RAY_NODE_MANAGER_PORT" --ray-client-server-port="$RAY_CLIENT_SERVER_PORT" --dashboard-agent-grpc-port="$RAY_DASHBOARD_AGENT_GRPC_PORT" --dashboard-agent-listen-port="$RAY_DASHBOARD_AGENT_LISTEN_PORT" --runtime-env-agent-port="$RAY_RUNTIME_ENV_AGENT_PORT" --metrics-export-port="$RAY_METRICS_EXPORT_PORT" --min-worker-port="$RAY_WORKER_MIN_PORT" --max-worker-port="$RAY_WORKER_MAX_PORT" --num-cpus=48 --resources='{"NPU":8}' \
        --temp-dir="$RAY_TMPDIR" --disable-usage-stats > "$RUN_DIR/ray-start.log" 2>&1 &
    local ray_pid=$!
    trap 'kill -TERM -- -'"$ray_pid"' 2>/dev/null || true; sleep 3; kill -KILL -- -'"$ray_pid"' 2>/dev/null || true' EXIT
    local ready=0
    for ((i=0; i<300; i++)); do
        if env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/curl --noproxy '*' --max-time 3 -fsS "http://127.0.0.1:$RAY_DASHBOARD_PORT/api/jobs/" >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$ray_pid" 2>/dev/null || { cat "$RUN_DIR/ray-start.log"; return 1; }
        sleep 2
    done
    (( ready )) || { cat "$RUN_DIR/ray-start.log"; return 1; }
    "$CONDA_ENV/bin/ray" job submit --submission-id="lh-$run_id" --address="http://127.0.0.1:$RAY_DASHBOARD_PORT" \
        --runtime-env-json="$(cat "$RUN_DIR/runtime-env.json")" -- bash "$RUN_DIR/command.sh"
}
