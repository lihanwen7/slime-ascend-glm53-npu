#!/bin/bash
# slime-ascend 构建脚本
#
# 功能：
#   1. 重置 slime submodule 到指定 commit
#   2. 按顺序应用补丁 (npu-compat → upstream-fixes → features)
#   3. 生成可用的 slime-ascend

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SLIME_DIR="${REPO_ROOT}/slime"
PATCHES_DIR="${REPO_ROOT}/patches"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[!]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*"
}

# 显示帮助
show_help() {
    cat << EOF
slime-ascend 构建脚本

用法:
    ./build.sh [选项]

选项:
    --clean         清理 slime 目录并重新构建
    --dry-run       仅显示将要执行的操作，不实际构建
    --skip-submodule 跳过 submodule 更新（用于快速重建）
    --help          显示此帮助信息

构建流程:
    1. 初始化/更新 slime submodule
    2. 应用 patches/npu-compat/ 补丁
    3. 应用 patches/upstream-fixes/ 补丁
    4. 应用 patches/features/ 补丁
    5. 验证构建结果

构建产物:
    - slime/                打满补丁的 slime 代码（即 submodule 工作区）
EOF
}

# 解析命令行参数
CLEAN_BUILD=false
DRY_RUN=false
SKIP_SUBMODULE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-submodule)
            SKIP_SUBMODULE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "slime-ascend 构建系统"
echo "=========================================="
echo ""

# 步骤 1: 清理 slime 目录
if [ "$CLEAN_BUILD" = true ]; then
    log_info "清理 slime 目录..."
    rm -rf "${SLIME_DIR}"
    log_success "slime 目录已清理"
    echo ""
fi

# 步骤 2: 初始化 submodule
if [ "$SKIP_SUBMODULE" = false ]; then
    log_info "初始化/更新 slime submodule..."

    if [ ! -d "${SLIME_DIR}" ]; then
        log_info "首次初始化 submodule..."
        if [ "$DRY_RUN" = false ]; then
            git submodule update --init --recursive "${SLIME_DIR}"
        else
            echo "  [DRY-RUN] git submodule update --init --recursive"
        fi
    else
        log_info "更新已存在的 submodule..."
        if [ "$DRY_RUN" = false ]; then
            cd "${SLIME_DIR}"
            git clean -xfd
            git checkout .
            git fetch origin
            git checkout "$(git rev-parse HEAD)"  # 确保在正确的 commit
            cd "${REPO_ROOT}"
        else
            echo "  [DRY-RUN] git fetch && checkout pinned commit"
        fi
    fi

    log_success "Submodule 准备完成"
    echo ""
else
    log_warn "跳过 submodule 更新"
    echo ""
fi

# 应用补丁的辅助函数
apply_patch_category() {
    local category=$1
    local patch_dir="${PATCHES_DIR}/${category}"

    if [ ! -d "${patch_dir}" ] || [ -z "$(ls -A "${patch_dir}" 2>/dev/null)" ]; then
        log_warn "补丁目录为空: ${category}"
        return 0
    fi

    log_info "应用 ${category} 补丁..."
    local patch_count=0
    local failed_patches=()

    for patch in "${patch_dir}"/*.patch; do
        if [ -f "$patch" ]; then
            local patch_name=$(basename "$patch")

            if [ "$DRY_RUN" = false ]; then
                cd "${SLIME_DIR}"
                if patch -p1 --forward --no-backup-if-mismatch < "$patch" > /dev/null 2>&1; then
                    echo "  ✓ ${patch_name}"
                    ((patch_count++))
                else
                    # 尝试检查是否已应用
                    if patch -p1 --reverse --dry-run < "$patch" > /dev/null 2>&1; then
                        log_warn "  已应用: ${patch_name}"
                    else
                        log_error "  失败: ${patch_name}"
                        failed_patches+=("${patch_name}")
                    fi
                fi
                cd "${REPO_ROOT}"
            else
                echo "  [DRY-RUN] 应用 ${patch_name}"
                ((patch_count++))
            fi
        fi
    done

    if [ ${#failed_patches[@]} -gt 0 ]; then
        log_error "以下补丁应用失败:"
        for p in "${failed_patches[@]}"; do
            echo "    - $p"
        done
        return 1
    fi

    log_success "应用 ${patch_count} 个 ${category} 补丁"
    return 0
}

# 步骤 3: 应用补丁
echo "=========================================="
echo "应用补丁"
echo "=========================================="
echo ""

# 按顺序应用三类补丁
apply_patch_category "npu-compat" || exit 1
echo ""
apply_patch_category "upstream-fixes" || exit 1
echo ""
apply_patch_category "features" || exit 1
echo ""

# 步骤 4: 验证构建
log_info "验证构建结果..."
if [ "$DRY_RUN" = false ]; then
    # 检查关键文件是否存在
    critical_files=(
        "${SLIME_DIR}/slime/__init__.py"
        "${SLIME_DIR}/setup.py"
    )

    all_exist=true
    for file in "${critical_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "关键文件缺失: $file"
            all_exist=false
        fi
    done

    if [ "$all_exist" = false ]; then
        log_error "构建验证失败"
        exit 1
    fi
else
    echo "  [DRY-RUN] 验证关键文件存在"
fi
log_success "构建验证通过"
echo ""

# 步骤 5: 生成构建信息
if [ "$DRY_RUN" = false ]; then
    cat > "${SLIME_DIR}/.build_info" << EOF
# slime-ascend 构建信息
build_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
upstream_commit=$(cd "${SLIME_DIR}" && git rev-parse HEAD)
upstream_version=$(cd "${SLIME_DIR}" && git describe --tags --always)
patches_npu_compat=$(ls -1 "${PATCHES_DIR}/npu-compat"/*.patch 2>/dev/null | wc -l | tr -d ' ')
patches_upstream_fixes=$(ls -1 "${PATCHES_DIR}/upstream-fixes"/*.patch 2>/dev/null | wc -l | tr -d ' ')
patches_features=$(ls -1 "${PATCHES_DIR}/features"/*.patch 2>/dev/null | wc -l | tr -d ' ')
EOF
fi

echo "=========================================="
echo "构建完成！"
echo "=========================================="
echo ""
if [ "$DRY_RUN" = false ]; then
    echo "构建产物位置: ${SLIME_DIR}"
    echo ""
    echo "安装方法:"
    echo "  cd ${SLIME_DIR}"
    echo "  pip install -e ."
    echo ""
    echo "或者设置 PYTHONPATH:"
    echo "  export PYTHONPATH=\"${SLIME_DIR}:\$PYTHONPATH\""
else
    echo "[DRY-RUN] 模拟构建完成"
fi
echo ""
