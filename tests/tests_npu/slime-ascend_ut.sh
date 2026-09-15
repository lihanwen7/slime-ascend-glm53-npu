#!/bin/bash
set -euo pipefail

# Standalone NPU CI entrypoint. Slime source patches are replayed from the
# official v0.3.2 submodule; dependency patches are installed by the matching
# quick_install/Docker build and are not applied a second time here.
WORKSPACE=${WORKSPACE:-/home/slime-ascend_CI/slime-ascend}
DEPS_ROOT=${SLIME_ASCEND_DEPS_ROOT:-/workspace}

cd "${WORKSPACE}"
bash build.sh

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH="${WORKSPACE}/slime:${DEPS_ROOT}/mbridge:${DEPS_ROOT}/Megatron-LM:${DEPS_ROOT}/sglang/python:${DEPS_ROOT}/Megatron-Bridge/src:${DEPS_ROOT}/MegatronAdaptor:${DEPS_ROOT}/TransformerEngineNPU:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
export HCCL_ENABLE_SL=${HCCL_ENABLE_SL:-1}
export HCCL_SHM_SIZE=${HCCL_SHM_SIZE:-2147483648}
export HCCL_POLLING_MODE=${HCCL_POLLING_MODE:-1}
export HCCL_COMBINE_MODE=${HCCL_COMBINE_MODE:-1}
export HCCL_ALGO_TYPE=${HCCL_ALGO_TYPE:-0}
export HCCL_IF_IP=${HCCL_IF_IP:-127.0.0.1}

python "${WORKSPACE}/tests/tests_npu/ci_check.py"
