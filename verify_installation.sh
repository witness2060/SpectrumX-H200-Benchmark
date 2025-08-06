#!/bin/bash
# プロジェクトの完全性を検証するスクリプト

echo "=== SpectrumX H200 Benchmark Suite - Installation Verification ==="
echo ""

# カラー定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# カウンター
PASSED=0
FAILED=0

# テスト関数
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $1 - NOT FOUND"
        ((FAILED++))
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $1/ - NOT FOUND"
        ((FAILED++))
        return 1
    fi
}

check_executable() {
    if [ -x "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 (executable)"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $1 - NOT EXECUTABLE"
        ((FAILED++))
        return 1
    fi
}

echo "1. Checking main scripts..."
check_executable "run_all.sh"
check_executable "test_quick.sh"

echo ""
echo "2. Checking setup scripts..."
check_executable "setup/cluster_config.sh"
check_executable "setup/install_dependencies.sh"
check_executable "setup/install_python_packages.sh"
check_executable "setup/configure_nccl.sh"
check_executable "setup/verify_cluster.sh"

echo ""
echo "3. Checking execution scripts..."
check_executable "scripts/run_benchmark.sh"
check_file "scripts/train_sft.py"
check_executable "scripts/collect_metrics.sh"
check_file "scripts/generate_report.py"
check_file "scripts/gpu_benchmark.py"

echo ""
echo "4. Checking configuration files..."
check_file "configs/ds_config_7b.json"
check_file "configs/ds_config_13b.json"
check_file "configs/ds_config_70b.json"
check_file "configs/nccl_config.conf"

echo ""
echo "5. Checking dataset scripts..."
check_file "datasets/prepare_dataset.py"
check_file "datasets/prepare_custom_dataset.py"

echo ""
echo "6. Checking directories..."
check_dir "results"
check_dir "docs"

echo ""
echo "7. Checking documentation..."
check_file "README.md"
check_file "CLAUDE.md"

echo ""
echo "8. Checking Python imports..."
echo -n "Testing Python environment... "
if python3 -c "import torch, transformers, deepspeed" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Core packages available"
    ((PASSED++))
else
    echo -e "${YELLOW}!${NC} Python packages not installed (run ./run_all.sh setup)"
    ((FAILED++))
fi

echo ""
echo "9. Checking system commands..."
for cmd in pdsh ssh nvidia-smi python3; do
    if command -v $cmd &> /dev/null; then
        echo -e "${GREEN}✓${NC} $cmd"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $cmd - NOT FOUND"
        ((FAILED++))
    fi
done

echo ""
echo "=================================="
echo "Verification Results:"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC} The installation is complete."
    echo ""
    echo "Next steps:"
    echo "1. Run quick test: ./test_quick.sh"
    echo "2. Full setup: ./run_all.sh setup"
    echo "3. Run benchmark: ./run_all.sh bench"
else
    echo -e "${RED}Some checks failed.${NC} Please review the errors above."
    echo ""
    echo "Common fixes:"
    echo "- Missing files: Check if you're in the correct directory"
    echo "- Not executable: Run 'chmod +x <filename>'"
    echo "- Python packages: Run './run_all.sh setup'"
fi

exit $FAILED