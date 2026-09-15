#!/bin/bash
set -euxo pipefail
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend"}
NPU_DEVICE=${NPU_DEVICE:=A3}
python -c 'import sys; assert sys.version_info[:2] == (3, 12), f"Python 3.12 is required, got {sys.version}"'
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DEPS_ROOT=${SLIME_ASCEND_DEPS_ROOT:-"$(dirname "${REPO_ROOT}")"}

# Pinned dependency contract for Slime v0.3.2 / Ascend validation.
CANN_VERSION=9.0.0
SGLANG_COMMIT=0.5.15.post1
SGLANG_BASE_COMMIT=0b3bb0cbe31873994c9f989fddfe2f87ca839fdd
SGLANG_ROUTER_PACKAGE=sglang-router==0.3.2
# Slime v0.3.2 uses the private prebuilt CANN 9.0/Python 3.12 bundle.
SGL_KERNEL_NPU_CP312_URL=https://gitcode.com/gcw_8jFUVqc5/sglang-kernel-npu-builds/releases/download/py312-cann9.0-a3/sgl-kernel-npu-torch2.10.0-py312-cann9.0-a3-aarch64.zip
SGL_KERNEL_NPU_FIX_COMMIT=9765e27532b8f854717eba1d3fb9a0bb97a0c887
MEMFABRIC_HYBRID_VERSION=1.0.8
TORCH_VERSION=2.10.0
TORCH_NPU_VERSION=2.10.0.post4
TORCHVISION_VERSION=0.25.0

MEGATRON_COMMIT=1dcf0dafa
MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
MEGATRON_ADAPTOR_COMMIT=f707a3b6eb9981532c3419c08ea40a8bb5962af5
MEGATRON_BRIDGE_COMMIT=35b4ebfc486fb15dcc0273ceea804c3606be948a
TRANSFORMER_ENGINE_NPU_COMMIT=2b8529f94c3fd10497bf772bdc5b448b7ed4659a
MegatronAdaptor_COMMIT=${MEGATRON_ADAPTOR_COMMIT}
TransformerEngineNPU_COMMIT=${TRANSFORMER_ENGINE_NPU_COMMIT}
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh
cd "${DEPS_ROOT}"

echo "1. install pinned torch packages"
pip install \
    "torch==${TORCH_VERSION}" \
    "torch_npu==${TORCH_NPU_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    setuptools==81.0.0

echo "2. install SGLang from source"
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout v${SGLANG_COMMIT}
test "$(git rev-parse HEAD)" = "${SGLANG_BASE_COMMIT}"
git am "${REPO_ROOT}"/patches/third_party/sglang/*
mv python/pyproject.toml python/pyproject.toml.backup
mv python/pyproject_npu.toml python/pyproject.toml
pip install -e "python[all_npu]" \
    --constraint <(printf '%s\n' \
        "torch==${TORCH_VERSION}" \
        "torch_npu==${TORCH_NPU_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}")
cd ..

echo "2.1 install other basic packages"
pip install "memfabric-hybrid==${MEMFABRIC_HYBRID_VERSION}"
pip install "${SGLANG_ROUTER_PACKAGE}" --force-reinstall
python -c "import sglang_router"

echo "3.install private CANN 9.0/Python 3.12 sgl-kernel-npu bundle"
KERNEL_ARCH=$(uname -m)
if [ "${KERNEL_ARCH}" != "aarch64" ]; then
    echo "the private cp312 sgl-kernel-npu bundle only provides linux_aarch64 wheels (got ${KERNEL_ARCH})" >&2
    exit 1
fi
case "${NPU_DEVICE}" in
    A3) KERNEL_DEVICE=a3 ;;
    A2) KERNEL_DEVICE=910b ;;
    *) echo "unsupported NPU_DEVICE: ${NPU_DEVICE}" >&2; exit 1 ;;
esac
KERNEL_BUNDLE_TMP=$(mktemp -d)
wget --no-check-certificate "${SGL_KERNEL_NPU_CP312_URL}" -O "${KERNEL_BUNDLE_TMP}/sgl-kernel-npu-cp312.zip"
unzip -q "${KERNEL_BUNDLE_TMP}/sgl-kernel-npu-cp312.zip" -d "${KERNEL_BUNDLE_TMP}"
KERNEL_BUNDLE_DIR=$(find "${KERNEL_BUNDLE_TMP}" -type f -name 'sgl_kernel_npu-*-cp312-cp312-linux_aarch64.whl' -printf '%h\n' | head -n 1)
test -n "${KERNEL_BUNDLE_DIR}"
test -f "${KERNEL_BUNDLE_DIR}"/torch_memory_saver-*-cp312-cp312-linux_aarch64.whl
test -f "${KERNEL_BUNDLE_DIR}"/deep_ep-*-cp312-cp312-linux_aarch64.whl
pip install "${KERNEL_BUNDLE_DIR}"/torch_memory_saver-*-cp312-cp312-linux_aarch64.whl \
    "${KERNEL_BUNDLE_DIR}"/sgl_kernel_npu-*-cp312-cp312-linux_aarch64.whl \
    "${KERNEL_BUNDLE_DIR}"/deep_ep-*-cp312-cp312-linux_aarch64.whl
# Link the DeepEP extension from the extracted private bundle.
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
DEEP_EP_CPP=$(find "${KERNEL_BUNDLE_DIR}" -type f -name 'deep_ep_cpp.cpython-312-aarch64-linux-gnu.so' -print -quit)
test -n "${DEEP_EP_CPP}"
ln -sf "${DEEP_EP_CPP}" \
    "${SITE}/deep_ep_cpp.cpython-312-aarch64-linux-gnu.so"

# (Optional) Confirm whether the import can be successfully
python -c "import deep_ep; print(deep_ep.__path__)"

echo "3.1 rebuild torch-memory-saver with the registered lifecycle patch"
git clone https://github.com/sgl-project/sgl-kernel-npu.git sgl-kernel-npu-src
cd sgl-kernel-npu-src
git checkout ${SGL_KERNEL_NPU_FIX_COMMIT}
git am "${REPO_ROOT}"/patches/third_party/sgl-kernel-npu/0001-fix-torch-memory-saver-allocation-lifecycle.patch
git am "${REPO_ROOT}"/patches/third_party/sgl-kernel-npu/0002-fix-tms-sglang-cuda-graph-api-and-preload-mode.patch
cd contrib/torch_memory_saver/python
python3 setup.py clean --all
python3 setup.py bdist_wheel
pip install --force-reinstall dist/torch_memory_saver*.whl
cd ../../../..

echo "4. install mbridge & Megatron-Bridge"
git clone https://github.com/ISEEKYAN/mbridge.git 
cd mbridge && git checkout ${MBRIDGE_COMMIT}
git am "${REPO_ROOT}"/patches/third_party/mbridge/*
pip install -e . && cd ..

git clone https://github.com/fzyzcjy/Megatron-Bridge.git
cd Megatron-Bridge && git checkout ${MEGATRON_BRIDGE_COMMIT}
git am "${REPO_ROOT}"/patches/third_party/megatron-bridge/*
cd ..
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
cd Megatron-LM/ && git checkout ${MEGATRON_COMMIT} && \
git am "${REPO_ROOT}"/patches/third_party/megatron/* && \
pip install -e . && cd ..

echo "5. install MA&TE"
git clone https://gitcode.com/Ascend/MegatronAdaptor.git && \
cd MegatronAdaptor/ && git checkout ${MegatronAdaptor_COMMIT} && \
pip install -e . && cd ..
git clone https://gitcode.com/Ascend/TransformerEngineNPU.git&& \
cd TransformerEngineNPU/ && git checkout ${TransformerEngineNPU_COMMIT} && \
pip install -e . && cd ..

echo "6. install triton_ascend"
pip uninstall triton triton_ascend -y
pip install triton_ascend==3.2.1

echo "7. install slime-ascend"
# Keep the submodule on the official v0.3.2 commit and replay the registered
# format-patch series before installing the Python project.
cd "${REPO_ROOT}"
bash build.sh
cd slime/
pip install -e .

echo "8. installation complete"
cd "${REPO_ROOT}"
# Create folder to store logs
mkdir -p logs
