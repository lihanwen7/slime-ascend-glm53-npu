# Source this file to activate Conda and the required Ascend SDKs.
source /opt/miniconda3/etc/profile.d/conda.sh || return
conda activate /mnt/share/l00804615/conda/envs/lhw_glm_53_flash_native || return
source /mnt/share/l00804615/cann/9.2.0/cann-9.2.0/set_env.sh || return
source /mnt/share/l00804615/cann/9.2.0/nnal/atb/set_env.sh || return
source "$CONDA_PREFIX/lib/python3.12/site-packages/deep_ep/vendors/hwcomputing/bin/set_env.bash" || return
source "/mnt/share/l00804615/cann/9.2.0/cann-9.2.0/opp/vendors/customize/bin/set_env.bash" || return
source "/mnt/share/l00804615/cann/9.2.0/cann-9.2.0/opp/vendors/custom_transformer/bin/set_env.bash" || return
