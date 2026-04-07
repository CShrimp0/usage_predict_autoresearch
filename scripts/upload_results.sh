#!/bin/bash
# ============================================================================
# 结果选择性上传脚本
# 功能：智能识别并上传重要的训练结果到 Git
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "结果选择性上传工具"
echo "=================================================="

cd "$PROJECT_ROOT"

# 检查是否在 Git 仓库中
if [ ! -d ".git" ]; then
    echo "❌ 错误: 当前目录不是 Git 仓库"
    exit 1
fi

echo -e "\n当前 Git 状态:"
git status --short

echo -e "\n=================================================="
echo "可上传的文件类型："
echo "=================================================="
echo "✅ 推荐上传:"
echo "   - 结果摘要 (results/summary/*.txt, *.md, *.json)"
echo "   - 关键图表 (results/figures/*.png - 精选)"
echo "   - 最佳模型配置 (results/best_results/*.json)"
echo ""
echo "⚠️ 不推荐上传:"
echo "   - 所有训练图表 (数量太多)"
echo "   - 完整 outputs/ 目录 (已在 .gitignore)"
echo "   - 权重文件 (已在 .gitignore)"
echo ""

# ============================================================================
# 智能识别重要文件
# ============================================================================
echo "=================================================="
echo "扫描重要文件..."
echo "=================================================="

IMPORTANT_FILES=()

# 1. results/summary/ 中的所有文件
if [ -d "results/summary" ]; then
    while IFS= read -r -d '' file; do
        IMPORTANT_FILES+=("$file")
    done < <(find results/summary -type f \( -name "*.txt" -o -name "*.md" -o -name "*.json" -o -name "*.csv" \) -print0)
fi

# 2. results/best_results/ 中的配置文件
if [ -d "results/best_results" ]; then
    while IFS= read -r -d '' file; do
        IMPORTANT_FILES+=("$file")
    done < <(find results/best_results -type f \( -name "*.json" -o -name "*.md" \) -print0)
fi

# 3. results/figures/ 中的关键图表（可以手动选择）
if [ -d "results/figures" ]; then
    echo -e "\nresults/figures/ 中的图表文件:"
    ls -lh results/figures/*.png 2>/dev/null || echo "  (无)"
fi

# 显示找到的文件
echo -e "\n找到以下重要文件:"
if [ ${#IMPORTANT_FILES[@]} -eq 0 ]; then
    echo "  (无)"
else
    for file in "${IMPORTANT_FILES[@]}"; do
        size=$(du -h "$file" | cut -f1)
        echo "  [$size] $file"
    done
fi

# ============================================================================
# 询问用户
# ============================================================================
echo -e "\n=================================================="
echo "上传选项"
echo "=================================================="
echo "1. 上传所有重要文件（推荐）"
echo "2. 选择性上传图表"
echo "3. 仅上传摘要和配置（不含图表）"
echo "4. 取消"
echo ""
read -p "请选择 [1-4]: " choice

case $choice in
    1)
        echo -e "\n上传所有重要文件..."
        
        # 添加所有重要文件
        for file in "${IMPORTANT_FILES[@]}"; do
            if [ -f "$file" ]; then
                git add "$file"
                echo "  ✓ 添加: $file"
            fi
        done
        
        # 添加所有图表
        if [ -d "results/figures" ]; then
            git add results/figures/*.png 2>/dev/null || true
            echo "  ✓ 添加: results/figures/*.png"
        fi
        ;;
        
    2)
        echo -e "\n选择性上传图表..."
        
        # 先添加摘要和配置
        for file in "${IMPORTANT_FILES[@]}"; do
            if [ -f "$file" ]; then
                git add "$file"
            fi
        done
        
        # 列出图表让用户选择
        if [ -d "results/figures" ]; then
            echo -e "\n可用图表："
            figures=(results/figures/*.png)
            for i in "${!figures[@]}"; do
                echo "  $((i+1)). ${figures[$i]}"
            done
            
            echo -e "\n输入要上传的图表编号（用空格分隔，如: 1 3 5）"
            echo "或输入 'all' 上传所有，'none' 不上传图表:"
            read -p "> " fig_choice
            
            if [ "$fig_choice" = "all" ]; then
                git add results/figures/*.png
                echo "  ✓ 添加所有图表"
            elif [ "$fig_choice" != "none" ]; then
                for num in $fig_choice; do
                    idx=$((num-1))
                    if [ $idx -ge 0 ] && [ $idx -lt ${#figures[@]} ]; then
                        git add "${figures[$idx]}"
                        echo "  ✓ 添加: ${figures[$idx]}"
                    fi
                done
            fi
        fi
        ;;
        
    3)
        echo -e "\n仅上传摘要和配置..."
        
        for file in "${IMPORTANT_FILES[@]}"; do
            if [ -f "$file" ]; then
                git add "$file"
                echo "  ✓ 添加: $file"
            fi
        done
        ;;
        
    4)
        echo "取消上传"
        exit 0
        ;;
        
    *)
        echo "无效选择"
        exit 1
        ;;
esac

# ============================================================================
# 显示暂存状态
# ============================================================================
echo -e "\n=================================================="
echo "当前暂存的文件:"
echo "=================================================="
git status --short

# 计算总大小
STAGED_SIZE=$(git diff --cached --stat | tail -n1 | grep -oP '\d+(?= file)' || echo "0")
echo -e "\n准备提交 $STAGED_SIZE 个文件"

# ============================================================================
# 确认并提交
# ============================================================================
echo -e "\n=================================================="
read -p "是否提交这些文件？[y/N]: " confirm

if [[ $confirm =~ ^[Yy]$ ]]; then
    read -p "输入提交信息: " commit_msg
    
    if [ -z "$commit_msg" ]; then
        commit_msg="Upload selected training results"
    fi
    
    git commit -m "$commit_msg"
    echo -e "\n✅ 提交成功！"
    echo ""
    echo "下一步："
    echo "  git push origin main    # 推送到远程仓库"
else
    echo -e "\n取消提交，文件仍在暂存区"
    echo "运行 'git reset' 取消暂存"
fi

echo -e "\n=================================================="
