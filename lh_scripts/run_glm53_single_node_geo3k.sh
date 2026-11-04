#!/usr/bin/env bash
# GLM-5.3-Flash, native single-node recipe. Edit the parameters below.
# --check performs static validation; a normal launch retains exclusive resource gates.
set -euo pipefail
export TASK_QUEUE_ENABLE=0
# Validated fused operators and packed KDA (override via environment for A/B tests).
export GLM53_KPOOL_BACKEND=${GLM53_KPOOL_BACKEND:-ascendc}
export GLM53_MHC_BACKEND=${GLM53_MHC_BACKEND:-ascendc}
export GLM53_SWIGLU_BACKEND=${GLM53_SWIGLU_BACKEND:-ascendc}
export GLM53_CONV_BACKEND=${GLM53_CONV_BACKEND:-ascendc}
export GLM53_KDA_SEQUENCE_MODE=${GLM53_KDA_SEQUENCE_MODE:-packed}
export GLM53_KDA_FWD_HEAD_CHUNK_SIZE=${GLM53_KDA_FWD_HEAD_CHUNK_SIZE:-64}
export GLM53_KDA_BWD_HEAD_CHUNK_SIZE=${GLM53_KDA_BWD_HEAD_CHUNK_SIZE:-64}
# Keep model offload for colocated online rollout/training.
# CPU packed boundaries and ViT alignment use the model's existing implementation.

# Ray address on this machine's communication NIC; change when moving hosts.
RAY_NODE_IP=${RAY_NODE_IP:-127.0.0.1}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GLM53_NUM_NPUS=16
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
source "${SCRIPT_DIR}/../runtime/lh-ascend-runtime.sh"

export HCCL_EXEC_TIMEOUT=3600 HCCL_CONNECT_TIMEOUT=3600 ACL_DEVICE_SYNC_TIMEOUT=3600

HF_CHECKPOINT=${HF_CHECKPOINT:?Set HF_CHECKPOINT to the HF model directory}
PROMPT_DATA=${PROMPT_DATA:?Set PROMPT_DATA to the JSONL dataset}
EXPERIMENT_NAME=glm53_geo3k
NUM_ROLLOUT=${NUM_ROLLOUT:-100}
RM_TYPE=${RM_TYPE:-deepscaler}
CP_SIZE=1

# Prompt limit includes the chat template. Total context capacity includes prompt and response.
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-2048}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-1024}
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-4096}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}

# Model architecture (top16 checkpoint).
MODEL_ARGS=(
    --spec slime_plugins.models.glm5_next.model get_glm5_next_model_provider
    --model-name glm5_next
    --num-layers 45
    --hidden-size 4096
    --ffn-hidden-size 12288
    --num-attention-heads 64
    --num-query-groups 64
    --kv-channels 256
    --q-lora-rank 1536
    --kv-lora-rank 512
    --vocab-size 154880
    --make-vocab-size-divisible-by 64
    --disable-bias-linear
    --swiglu
    --untie-embeddings-and-output-weights
    --position-embedding-type none
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --rotary-base 800000
    --num-experts 16
    --moe-layer-freq '[0]*3+[1]*42'
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-topk 8
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk-scaling-factor 2.5
    --moe-aux-loss-coeff 0
    --moe-router-dtype fp32
    --moe-grouped-gemm
    --moe-permute-fusion
    --freeze-indexer
    --enable-experimental
)

# Native MoE workspace limits. Set both to 0 for the original A/B baseline.
GLM53_MOE_DISPATCH_CHUNK_SIZE=${GLM53_MOE_DISPATCH_CHUNK_SIZE:-0}
GLM53_EXPERT_TOKEN_CHUNK_SIZE=${GLM53_EXPERT_TOKEN_CHUNK_SIZE:-0}
TRAIN_ENV_JSON=$(/usr/bin/python3 -I -S - "$TRAIN_ENV_JSON" "$GLM53_MOE_DISPATCH_CHUNK_SIZE" "$GLM53_EXPERT_TOKEN_CHUNK_SIZE" <<'PY_CHUNKS'
import json
import sys
values = json.loads(sys.argv[1])
for key, raw in zip(("GLM53_MOE_DISPATCH_CHUNK_SIZE", "GLM53_EXPERT_TOKEN_CHUNK_SIZE"), sys.argv[2:]):
    size = int(raw)
    assert size >= 0, key
    values[key] = str(size)
print(json.dumps(values))
PY_CHUNKS
)

# Rollout, optimizer, parallelism and SGLang settings.
ARGS=(
    --actor-num-nodes 1 --actor-num-gpus-per-node 16 --num-gpus-per-node 16 --colocate
    --train-env-vars "$TRAIN_ENV_JSON"
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT" --megatron-to-hf-mode raw
    --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --apply-chat-template
    --multimodal-keys '{"image":"images"}'
    --rm-type "${RM_TYPE:-deepscaler}" --seed 522
    --num-rollout "${NUM_ROLLOUT:-2}" --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
    --global-batch-size "$GLOBAL_BATCH_SIZE" --micro-batch-size 1
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" --rollout-temperature 1 --rollout-top-p 1
    --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" --rollout-max-context-len "$MAX_CONTEXT_LEN"
    --advantage-estimator grpo --pg-loss-type reinforce
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11
    --context-parallel-size "${CP_SIZE:-1}" --expert-model-parallel-size 4 --expert-tensor-parallel-size 1
    --allgather-cp --moe-token-dispatcher-type alltoall
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --seq-length "$MAX_CONTEXT_LEN" --max-position-embeddings "$MAX_CONTEXT_LEN"
    # Packing budget only: an oversized individual sample is scheduled alone.
    --use-dynamic-batch-size --max-tokens-per-gpu 2048 --data-pad-size-multiplier 1024
    --log-probs-chunk-size 1024
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --no-load-optim --no-load-rng --no-save-optim --freeze-params-name-list 'model.visual.*'
    --optimizer-cpu-offload --use-precision-aware-optimizer
    --overlap-cpu-optimizer-d2h-h2d
    --rollout-num-gpus-per-engine 16
    --sglang-device npu --sglang-attention-backend ascend --sglang-tp-size 16
    --sglang-dp-size 1 --sglang-ep-size 1
    # Keep each image encoder unsharded to match training ViT numerics.
    --sglang-mm-enable-dp-encoder
    --sglang-mem-fraction-static 0.55 --sglang-context-length "$MAX_CONTEXT_LEN"
    --sglang-max-running-requests 32 --sglang-chunked-prefill-size 1024
    --sglang-disable-overlap-schedule --sglang-cuda-graph-bs 1 2 4 8 16 24 32
    --sglang-page-size 64 --sglang-kv-cache-dtype bfloat16 --sglang-watchdog-timeout 1200
    --attention-dropout 0 --hidden-dropout 0 --bf16
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --distributed-timeout-minutes 20 --update-weight-buffer-size 268435456
    --rollout-data-transport object-store
    --check-weight-update-equal
    --use-rollout-routing-replay
)

# Optional fixed-input numerical diagnostics.
if [[ -n ${SAVE_DEBUG_ROLLOUT_DATA:-} ]]; then
    # Durable rollout capture enables recovery; keep optimizer/RNG checkpoint state.
    diagnostic_args=()
    for arg in "${ARGS[@]}"; do
        [[ "$arg" == --no-save-optim || "$arg" == --no-save-rng ]] || diagnostic_args+=("$arg")
    done
    ARGS=("${diagnostic_args[@]}" --ckpt-format torch_dist --save-debug-rollout-data "$SAVE_DEBUG_ROLLOUT_DATA")
    unset diagnostic_args
fi
[[ -z ${SAVE_DEBUG_TRAIN_DATA:-} ]] || ARGS+=(--save-debug-train-data "$SAVE_DEBUG_TRAIN_DATA")

glm53_submit "$@"
