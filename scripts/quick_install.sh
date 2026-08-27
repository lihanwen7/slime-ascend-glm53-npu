#!/bin/bash
set -euxo pipefail
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend"}
NPU_DEVICE=${NPU_DEVICE:=A3}

MEGATRON_COMMIT=1dcf0dafa
MegatronAdaptor_COMMIT=f707a3b6
TransformerEngineNPU_COMMIT=47d60449
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh

echo "1. install SGLang from source"
git clone -b v0.5.13 https://github.com/sgl-project/sglang.git
cd sglang
mv python/pyproject.toml python/pyproject.toml.backup
mv python/pyproject_npu.toml python/pyproject.toml
pip install -e python[all_npu]
cd ..

echo "2. install torch & torch_npu & other basic packages"
pip install torch==2.10.0 torch_npu==2.10.0 torchvision==0.25.0 setuptools==81.0.0

echo "3.install sgl-kernel-npu from release whl"
if [ "$NPU_DEVICE" = "A3" ]; then
    wget --no-check-certificate https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.6.1/sgl-kernel-npu-2026.6.1-torch2.10.0-py311-cann9.0.0-a3-aarch64.zip
fi
if [ "$NPU_DEVICE" = "A2" ]; then
    wget --no-check-certificate https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.6.1/sgl-kernel-npu-2026.6.1-torch2.10.0-py311-cann9.0.0-910b-aarch64.zip
fi
unzip sgl-kernel-npu*.zip
pip install torch_memory_saver*.whl
pip install sgl_kernel_npu*.whl
pip install deep_ep*.whl
# For using deep_ep: Link to the deep_ep_cpp.*.so file
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so
# Avoid "ln -s" failure due to existing deep_ep symlink which breaks "cd -", hence split it out separately.
cd -

# (Optional) Confirm whether the import can be successfully
python -c "import deep_ep; print(deep_ep.__path__)"

echo "4. install mbridge & Megatron-Bridge"
git clone https://github.com/ISEEKYAN/mbridge.git 
cd mbridge && git checkout 89eb10887887bc74853f89a4de258c0702932a1c
pip install -e . && cd ..

git clone https://github.com/fzyzcjy/Megatron-Bridge.git
cd Megatron-Bridge && git checkout dev_rl && cd ..
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
cd Megatron-LM/ && git checkout ${MEGATRON_COMMIT} && \
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
cd slime-ascend/
pip install -e . 

echo "8. apply npu patches"
cd ../sglang
git am ../slime-ascend/patches/third_party/sglang/*
cd ../Megatron-LM/
git am ../slime-ascend/patches/third_party/megatron/*
cd ../Megatron-Bridge/
git am ../slime-ascend/patches/third_party/megatron-bridge/*
cd ../mbridge/
git am ../slime-ascend/patches/third_party/mbridge/*
# Apply TransformerEngineNPU patches (fixed path - was TransformerEngineNPU, should be transformer_engine_npu)
cd ../TransformerEngineNPU/
git am ../slime-ascend/patches/third_party/transformer_engine_npu/*
# Note: MegatronAdaptor patches directory does not exist in v0.3.0
# If needed, patches should be added to docker/npu_patch/v0.3.0/megatron_adaptor/
echo "9. return to slime-ascend directory"
cd ../slime-ascend/

echo "9. build slime"
bash build.sh
# Create folder to store logs
mkdir -p logs