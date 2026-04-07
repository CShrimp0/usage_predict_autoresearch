#!/bin/bash
# ============================================================================
# 目录迁移脚本 - 重组项目结构
# 功能：将现有文件迁移到新的优化目录结构
# ============================================================================

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "开始迁移目录结构"
echo "项目根目录: $PROJECT_ROOT"
echo "=================================================="

cd "$PROJECT_ROOT"

# ============================================================================
# 1. 迁移文档文件到 docs/
# ============================================================================
echo -e "\n[1/6] 迁移文档文件到 docs/..."

if [ -f "DATASET_OPTIMIZATION.md" ]; then
    mv -v DATASET_OPTIMIZATION.md docs/
fi

if [ -f "TRAINING_GUIDE.md" ]; then
    mv -v TRAINING_GUIDE.md docs/
fi

if [ -f "README.md" ]; then
    cp -v README.md docs/README_backup.md  # 保留一份在根目录
fi

echo "✓ 文档迁移完成"

# ============================================================================
# 2. 迁移工具脚本
# ============================================================================
echo -e "\n[2/6] 整理工具脚本..."

# utils/ 目录中的脚本移动到 scripts/
if [ -d "utils" ]; then
    echo "移动 utils/ 中的工具脚本到 scripts/..."
    
    if [ -f "utils/analyze_dataset.py" ]; then
        mv -v utils/analyze_dataset.py scripts/
    fi
    
    if [ -f "utils/verify_no_leakage.py" ]; then
        mv -v utils/verify_no_leakage.py scripts/
    fi
    
    if [ -f "utils/visualize_image_sizes.py" ]; then
        mv -v utils/visualize_image_sizes.py scripts/
    fi
    
    if [ -f "utils/plot_age_error.py" ]; then
        mv -v utils/plot_age_error.py scripts/
    fi
fi

echo "✓ 工具脚本整理完成"

# ============================================================================
# 3. 提取重要结果到 results/summary/
# ============================================================================
echo -e "\n[3/6] 提取重要结果到 results/summary/..."

# 从 analysis_results 提取重要文件
if [ -d "analysis_results" ]; then
    echo "从 analysis_results/ 提取关键结果..."
    
    if [ -f "analysis_results/analysis_summary.txt" ]; then
        cp -v analysis_results/analysis_summary.txt results/summary/
    fi
    
    if [ -f "analysis_results/DATA_LEAKAGE_FIX_SUMMARY.md" ]; then
        cp -v analysis_results/DATA_LEAKAGE_FIX_SUMMARY.md results/summary/
    fi
    
    if [ -f "analysis_results/CONFIG_IMPROVEMENTS.md" ]; then
        cp -v analysis_results/CONFIG_IMPROVEMENTS.md results/summary/
    fi
    
    # CSV 文件（小文件，可以保留）
    if [ -f "analysis_results/similarity_analysis.csv" ]; then
        cp -v analysis_results/similarity_analysis.csv results/summary/
    fi
    
    if [ -f "analysis_results/subject_summary.csv" ]; then
        cp -v analysis_results/subject_summary.csv results/summary/
    fi
fi

echo "✓ 结果提取完成"

# ============================================================================
# 4. 提取最佳模型的配置和历史（不包含权重）
# ============================================================================
echo -e "\n[4/6] 提取最佳模型的配置和历史..."

# 查找最佳训练运行（基于验证MAE）
BEST_RUN=""
BEST_MAE=999999.0

if [ -d "outputs" ]; then
    for run_dir in outputs/run_*/; do
        if [ -f "${run_dir}config.json" ]; then
            # 提取 val_mae（需要 jq 工具，如果没有则跳过）
            if command -v jq &> /dev/null; then
                # 从 history.json 获取最小 val_mae
                if [ -f "${run_dir}history.json" ]; then
                    val_mae=$(jq -r '.val_mae | min' "${run_dir}history.json" 2>/dev/null || echo "999999")
                    if (( $(echo "$val_mae < $BEST_MAE" | bc -l) )); then
                        BEST_MAE=$val_mae
                        BEST_RUN=$run_dir
                    fi
                fi
            fi
        fi
    done
    
    if [ -n "$BEST_RUN" ]; then
        echo "找到最佳训练运行: $BEST_RUN (MAE: $BEST_MAE)"
        
        # 复制配置和历史到 results/best_results/
        cp -v "${BEST_RUN}config.json" results/best_results/
        cp -v "${BEST_RUN}history.json" results/best_results/
        
        # 如果有训练曲线图，也复制过去
        if [ -f "${BEST_RUN}training_curves.png" ]; then
            cp -v "${BEST_RUN}training_curves.png" results/figures/best_training_curves.png
        fi
        
        # 如果有预测结果图，复制关键图片
        if [ -d "${BEST_RUN}predict_result" ]; then
            echo "复制预测结果可视化..."
            cp -v "${BEST_RUN}predict_result"/*.png results/figures/ 2>/dev/null || true
        fi
        
        echo "最佳模型配置和结果已保存到 results/best_results/"
    else
        echo "警告: 未找到有效的训练运行或缺少 jq 工具"
    fi
fi

echo "✓ 最佳结果提取完成"

# ============================================================================
# 5. 移动权重文件到专用目录（可选）
# ============================================================================
echo -e "\n[5/6] 整理权重文件..."

echo "注意: 权重文件 (.pth) 将保留在原位置"
echo "      它们已在 .gitignore 中配置为不上传"
echo "      如需备份，请手动使用外部存储"

# 创建权重目录的 README
cat > weights/README.md << 'EOF'
# 权重文件存储

此目录用于存储模型权重文件 (.pth, .pt)。

## 重要说明

⚠️ **权重文件不会上传到 GitHub**

- 所有 `.pth` 和 `.pt` 文件已在 `.gitignore` 中配置忽略
- 权重文件体积大，不适合 Git 版本控制
- 建议使用外部存储（如网盘、对象存储）备份重要权重

## 推荐做法

1. 将最佳模型权重复制到此目录并重命名：
   ```bash
   cp outputs/run_xxx/best_model.pth weights/model_v1_mae_6.89.pth
   ```

2. 创建权重清单记录重要模型：
   ```
   model_v1_mae_6.89.pth - ResNet50, 训练时间: 2025-12-27
   model_v2_ensemble.pth - 6模型集成
   ```

3. 使用云存储备份：
   - 百度网盘 / 阿里云盘
   - AWS S3 / 阿里云 OSS
   - 实验室服务器
EOF

echo "✓ 权重目录配置完成"

# ============================================================================
# 6. 创建目录结构说明
# ============================================================================
echo -e "\n[6/6] 生成目录结构说明..."

tree -L 2 -I 'outputs|analysis_results|__pycache__|*.pyc|*.pth' > docs/directory_structure.txt 2>/dev/null || echo "提示: 安装 tree 命令可生成目录树"

echo "✓ 说明生成完成"

# ============================================================================
# 完成
# ============================================================================
echo -e "\n=================================================="
echo "✅ 目录迁移完成！"
echo "=================================================="
echo ""
echo "新的目录结构："
echo "  docs/          - 文档文件"
echo "  scripts/       - 工具脚本"
echo "  results/       - 训练结果（可选择性上传）"
echo "    ├── figures/      - 可视化图表"
echo "    ├── summary/      - 结果摘要"
echo "    └── best_results/ - 最佳模型配置"
echo "  weights/       - 模型权重（不上传）"
echo "  outputs/       - 完整训练输出（不上传）"
echo ""
echo "下一步："
echo "  1. 检查迁移结果: ls -la docs/ scripts/ results/"
echo "  2. 提交代码: git add . && git commit -m 'Reorganize project structure'"
echo "  3. 使用 scripts/upload_results.sh 选择性上传结果"
echo "=================================================="
