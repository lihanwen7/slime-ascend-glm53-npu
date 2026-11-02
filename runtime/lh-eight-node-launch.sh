#!/usr/bin/env bash
# Ray lifecycle only. The recipe owns model/training args; env.sh owns platform setup.
source "${REPO_ROOT}/runtime/lh-eight-node-env.sh"

glm53_eight_usage() {
    local entrypoint
    entrypoint=$(realpath -- "${BASH_SOURCE[-1]}")
    cat <<USAGE
Usage: same script on each node, with the SAME NEW --run-id (timestamp recommended).
  bash "$entrypoint" --check
  bash "$entrypoint" --print-command
  bash "$entrypoint" --print-env
  bash "$entrypoint" --run-id 20261028_120000_full288
Optional: --cann-package-manifest /absolute/path/to/approved/ascend_package_load.ini
The manifest is read-only mounted in a private namespace; no shared SDK file changes.
--help/--print-* do not activate Python/CANN, scan data, start Ray or touch NPUs.
--check validates checkpoint/data with system Python; it does not import torch.
Logs: $LOG_ROOT/<run-id>/node21 ... node28 (default: project cache/work/glm53-eight-node).
Training settings are in the recipe; platform settings and reasons are in env.sh.
USAGE
}

glm53_eight_submit() {
    local mode=run run_id='' manifest='' arg
    local -a original_args=("$@")
    if (( $# == 0 )); then glm53_eight_usage; return 0; fi
    while (( $# )); do
        arg=$1; shift
        case "$arg" in
            --help|-h) glm53_eight_usage; return 0 ;;
            --check|--print-command|--print-env)
                [[ $mode == run ]] || { echo 'Choose one inspection mode' >&2; return 2; }
                mode=${arg#--} ;;
            --run-id) [[ $# -gt 0 ]] || return 2; run_id=$1; shift ;;
            --cann-package-manifest) [[ $# -gt 0 && $1 == /* ]] || return 2; manifest=$1; shift ;;
            *) echo "Unknown argument: $arg" >&2; return 2 ;;
        esac
    done
    if [[ $mode == print-command ]]; then
        printf '%q ' "$CONDA_ENV/bin/python" -u "$SLIME_ROOT/train.py" "${ARGS[@]}"
        printf '\n'; return 0
    fi
    if [[ $mode == print-env ]]; then
        printf '%s\n' '[all nodes: explicit platform environment]' "${GLM53_EIGHT_COMMON_ENV[@]}"
        printf '%s\n' "GLOO_SOCKET_IFNAME=$FULL_COMM_IFNAME" "HCCL_SOCKET_IFNAME=$FULL_COMM_IFNAME"
        printf '%s\n' 'SLIME_HOST_IP=<this node communication IP>'
        printf '%s\n' 'HCCL_OP_EXPANSION_MODE=<unset: driver default>'
        printf '%s\n' '[training actors only]' "$TRAIN_ENV_JSON"
        printf '%s\n' "CANN package manifest: ${manifest:-shared default (no private override)}"
        return 0
    fi
    if [[ $mode == run ]]; then
        [[ $run_id =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$ ]] || {
            echo 'Supply --run-id NAME (a new, identical name on all eight nodes).' >&2; return 2;
        }
    fi
    local rank=-1 i address
    local addresses
    addresses=$(ip -o -4 addr show dev "$FULL_COMM_IFNAME")
    for i in "${!NODE_IPS[@]}"; do
        address=${NODE_IPS[$i]}
        if awk '{print $4}' <<< "$addresses" | cut -d/ -f1 | grep -Fxq "$address"; then
            [[ $rank == -1 ]] || { echo 'Multiple configured node IPs found' >&2; return 2; }
            rank=$i
        fi
    done
    (( rank >= 0 )) || { echo "No configured cluster IP on $FULL_COMM_IFNAME" >&2; return 2; }
    local node_ip=${NODE_IPS[$rank]} master_ip=${NODE_IPS[0]}
    local gcs_port=25040 dashboard_port=25041
    echo "[glm53-8node] rank=$rank node=$node_ip master=$master_ip interface=$FULL_COMM_IFNAME"

    # Optional workaround must be explicit. Mount only inside this launch process.
    if [[ $mode == run && -n $manifest ]]; then
        local shared_manifest="$CANN_ROOT/aarch64-linux/conf/ascend_package_load.ini"
        [[ -f $manifest && -f $shared_manifest ]] || { echo 'CANN manifest file missing' >&2; return 2; }
        if [[ ! $manifest -ef $shared_manifest ]]; then
            local entrypoint
            entrypoint=$(realpath -- "${BASH_SOURCE[-1]}")
            exec unshare --mount --propagation private bash -euc '
                mount --bind "$1" "$2"
                mount -o remount,bind,ro "$2"
                shift 2
                exec bash "$@"
            ' -- "$manifest" "$shared_manifest" "$entrypoint" "${original_args[@]}"
        fi
    fi

    /usr/bin/python3 -I -S - "$HF_CHECKPOINT" "$PROMPT_DATA" "${ARGS[@]}" <<'PY_CHECK'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
args = sys.argv[3:]
def option(name, default=None):
    if name not in args:
        assert default is not None, f'Missing {name}'
        return default
    assert args.count(name) == 1, f'Duplicate {name}'
    return args[args.index(name) + 1]
world = int(option('--actor-num-nodes')) * int(option('--actor-num-gpus-per-node'))
tp = int(option('--tensor-model-parallel-size'))
pp = int(option('--pipeline-model-parallel-size'))
cp = int(option('--context-parallel-size'))
ep = int(option('--expert-model-parallel-size'))
experts = int(option('--num-experts'))
batch = int(option('--global-batch-size'))
micro = int(option('--micro-batch-size'))
assert world == 64 and min(tp, pp, cp, ep, experts, batch, micro) > 0
assert world % (tp * pp * cp) == 0 and world % (tp * pp * ep) == 0
assert experts % ep == 0
assert batch % ((world // (tp * pp * cp)) * micro) == 0
cfg = json.loads((root / 'config.json').read_text())['text_config']
expected = {'num_hidden_layers': 45, 'hidden_size': 4096, 'n_routed_experts': experts,
            'num_experts_per_tok': 8, 'vocab_size': 154880}
for key, value in expected.items():
    assert cfg[key] == value, (key, cfg[key], value)
index = json.loads((root / 'model.safetensors.index.json').read_text())
shards = set(index['weight_map'].values())
missing = [name for name in shards if not (root / name).is_file()]
assert not missing, missing
input_key = option('--input-key', 'prompt')
label_key = option('--label-key', 'label')
multimodal = json.loads(option('--multimodal-keys', '{}'))
count = 0
with open(sys.argv[2]) as f:
    for line in f:
        if line.strip():
            sample = json.loads(line)
            assert input_key in sample and label_key in sample
            if 'image' in multimodal:
                images = sample.get(multimodal['image'])
                assert isinstance(images, list) and images, f'No images in sample {count}'
                for image in images:
                    path = image.get('image') if isinstance(image, dict) else image
                    assert isinstance(path, str) and pathlib.Path(path).is_absolute(), path
                    assert pathlib.Path(path).is_file(), path
            count += 1
assert count > 0
print(f'CHECK PASSED: {world} NPUs, TP/PP/CP/EP={tp}/{pp}/{cp}/{ep}, '
      f'DP={world // (tp * pp * cp)}, {experts} experts, {len(shards)} shards, {count} samples')
PY_CHECK
    [[ $mode != check ]] || return 0

    ulimit -c 0
    ulimit -n 65535
    mkdir -p "/tmp/glm53-lh-$UID"
    exec 9>"/tmp/glm53-lh-$UID/single-node.lock"
    flock -n 9 || { echo 'This node already has an lh training job' >&2; return 75; }
    local node_log cluster_dir
    cluster_dir="$LOG_ROOT/$run_id"
    node_log="$cluster_dir/node$((21+rank))"
    mkdir -p "$cluster_dir"
    # A run ID must never reuse an earlier node session or completion marker.
    mkdir "$node_log" || { echo "Run ID already used on this node: $run_id" >&2; return 2; }
    exec > >(tee -a "$node_log/launcher.log") 2>&1
    npu-smi info > "$node_log/npu-before.txt"
    if grep -Eq '\|[[:space:]]*[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+[[:space:]]*\|' "$node_log/npu-before.txt"; then
        echo 'NPU processes are present; stop the existing workload first' >&2; return 75
    fi
    [[ ! -e "$cluster_dir/head-exit-code" ]] || { echo 'Run ID has already completed' >&2; return 2; }
    local short_id
    short_id=$(printf '%s' "$run_id" | sha256sum | cut -c1-12)
    export TMPDIR="/tmp/g53e-$UID/$short_id"
    mkdir -p "$(dirname "$TMPDIR")"
    mkdir "$TMPDIR" || { echo "Local session already exists: $TMPDIR" >&2; return 2; }
    export RAY_TMPDIR="$TMPDIR/ray"
    export TRITON_CACHE_DIR="$node_log/triton" TORCH_EXTENSIONS_DIR="$node_log/torch-extensions"
    export HF_MODULES_CACHE="$node_log/hf_modules"
    mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$HF_MODULES_CACHE"
    glm53_eight_activate
    cd "$SLIME_ROOT"
    local -a ray_args=(start --block --node-ip-address="$node_ip" --num-cpus=48
        --resources='{"NPU":8}' --disable-usage-stats
        --object-manager-port=25042 --node-manager-port=25043
        --dashboard-agent-grpc-port=25045 --dashboard-agent-listen-port=25046
        --runtime-env-agent-port=25047 --metrics-export-port=25048
        --min-worker-port=28600 --max-worker-port=28999)
    if (( rank == 0 )); then
        ray_args+=(--head --port="$gcs_port" --dashboard-host="$master_ip"
            --dashboard-port="$dashboard_port" --ray-client-server-port=25044 --temp-dir="$RAY_TMPDIR")
    else
        local ready=0
        for ((i=0; i<450; i++)); do
            [[ ! -e "$cluster_dir/head-exit-code" ]] || { echo 'Head has exited'; return 1; }
            if env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/curl --noproxy '*' --max-time 2 -fsS \
                "http://$master_ip:$dashboard_port/api/jobs/" >/dev/null 2>&1; then ready=1; break; fi
            sleep 2
        done
        (( ready )) || { echo 'Timed out waiting for head'; return 1; }
        ray_args+=(--address="$master_ip:$gcs_port")
    fi
    setsid "$CONDA_ENV/bin/ray" "${ray_args[@]}" > "$node_log/ray-start.log" 2>&1 &
    local ray_pid=$!
    # Values captured in the trap survive this function returning. Only our process group is stopped.
    trap 'rc=$?; trap - EXIT; if (( '"$rank"' == 0 )); then printf "%s\n" "$rc" > '"\"$cluster_dir/head-exit-code\""'; fi; kill -TERM -- -'"$ray_pid"' 2>/dev/null || true; sleep 3; kill -KILL -- -'"$ray_pid"' 2>/dev/null || true; exit "$rc"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    printf '%s\n' "$ray_pid" > "$node_log/ray.pid"
    if (( rank != 0 )); then
        while [[ ! -e "$cluster_dir/head-exit-code" ]]; do
            kill -0 "$ray_pid" 2>/dev/null || { tail -60 "$node_log/ray-start.log"; return 1; }
            sleep 5
        done
        return "$(cat "$cluster_dir/head-exit-code")"
    fi
    local ready=0
    for ((i=0; i<300; i++)); do
        if env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/curl --noproxy '*' --max-time 2 -fsS \
            "http://$master_ip:$dashboard_port/api/jobs/" >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$ray_pid" 2>/dev/null || { tail -60 "$node_log/ray-start.log"; return 1; }
        sleep 2
    done
    (( ready )) || { echo 'Head dashboard startup timed out'; return 1; }
    "$CONDA_ENV/bin/python" - "$master_ip:$gcs_port" "${NODE_IPS[@]}" <<'PY_NODES'
import ray, sys, time
ray.init(address=sys.argv[1])
expected = set(sys.argv[2:])
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    nodes = [n for n in ray.nodes() if n['Alive']]
    found = {n['NodeManagerAddress']: n['Resources'].get('NPU', 0) for n in nodes}
    if set(found) - expected:
        raise RuntimeError(f'Unexpected nodes in cluster: {found}')
    if set(found) == expected and all(v == 8 for v in found.values()):
        print('All 8 nodes / 64 NPUs ready:', found, flush=True)
        break
    print('Waiting for nodes:', found, flush=True)
    time.sleep(10)
else:
    raise RuntimeError('Timed out waiting for all 8 nodes / 64 NPUs')
ray.shutdown()
PY_NODES
    printf '%q ' "$CONDA_ENV/bin/python" -u "$SLIME_ROOT/train.py" "${ARGS[@]}" > "$cluster_dir/command.sh"
    printf '\n' >> "$cluster_dir/command.sh"
    # Every raylet already has its own initialized environment and local fast SLIME_HOST_IP.
    # Do not distribute the head node's IP, cache paths or temporary SDK view to other nodes.
    printf '%s\n' '{"worker_process_setup_hook":"glm53_ctor_setup_v5.install"}' > "$cluster_dir/runtime-env.json"
    "$CONDA_ENV/bin/ray" job submit --submission-id="glm53-full-$run_id" \
        --address="http://$master_ip:$dashboard_port" \
        --runtime-env-json="$(cat "$cluster_dir/runtime-env.json")" \
        -- bash "$cluster_dir/command.sh"
}
