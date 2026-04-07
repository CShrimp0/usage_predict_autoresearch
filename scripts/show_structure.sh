#!/bin/bash
# 快速查看项目目录结构

echo "========================================"
echo "项目目录结构"
echo "========================================"
echo ""

# 核心代码文件
echo "📄 核心代码文件:"
ls -lh *.py 2>/dev/null | awk '{printf "  %-30s %8s\n", $9, $5}'

echo ""
echo "📂 文档目录 (docs/):"
ls -lh docs/*.md 2>/dev/null | awk '{printf "  %-30s %8s\n", $9, $5}'

echo ""
echo "📂 工具脚本 (scripts/):"
ls -lh scripts/ 2>/dev/null | grep -E '\.(sh|py)$' | awk '{printf "  %-30s %8s\n", $9, $5}'

echo ""
echo "📂 训练结果 (results/):"
echo "  ├── summary/      $(find results/summary -type f 2>/dev/null | wc -l) 个文件"
echo "  ├── figures/      $(find results/figures -type f 2>/dev/null | wc -l) 个文件"
echo "  └── best_results/ $(find results/best_results -type f 2>/dev/null | wc -l) 个文件"

echo ""
echo "📂 模型权重 (weights/):"
if [ -d "weights" ]; then
    weight_count=$(find weights -name "*.pth" 2>/dev/null | wc -l)
    if [ $weight_count -gt 0 ]; then
        weight_size=$(du -sh weights/ 2>/dev/null | cut -f1)
        echo "  $weight_count 个权重文件 (总计: $weight_size)"
        find weights -name "*.pth" -exec basename {} \; | head -5
        if [ $weight_count -gt 5 ]; then
            echo "  ... 还有 $((weight_count - 5)) 个文件"
        fi
    else
        echo "  (暂无权重文件)"
    fi
fi

echo ""
echo "📂 训练输出 (outputs/):"
if [ -d "outputs" ]; then
    run_count=$(find outputs -maxdepth 1 -type d -name "run_*" 2>/dev/null | wc -l)
    outputs_size=$(du -sh outputs/ 2>/dev/null | cut -f1)
    echo "  $run_count 次训练运行 (总计: $outputs_size)"
    if [ $run_count -gt 0 ]; then
        echo "  最近的运行:"
        ls -td outputs/run_* 2>/dev/null | head -3 | while read dir; do
            run_name=$(basename "$dir")
            run_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "    $run_name ($run_size)"
        done
    fi
fi

echo ""
echo "========================================"
echo "磁盘占用统计"
echo "========================================"

# 计算各目录大小
for dir in outputs analysis_results weights results data; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        printf "%-20s %10s\n" "$dir/" "$size"
    fi
done

echo ""
echo "========================================"
echo "Git 状态"
echo "========================================"
git status --short 2>/dev/null || echo "  (不是Git仓库或Git未安装)"

echo ""
echo "========================================"
echo "快速操作"
echo "========================================"
echo "  bash scripts/migrate_structure.sh  - 迁移文件到新结构"
echo "  bash scripts/upload_results.sh     - 选择性上传结果"
echo "  cat OPTIMIZATION_SUMMARY.md        - 查看优化总结"
echo "========================================"
