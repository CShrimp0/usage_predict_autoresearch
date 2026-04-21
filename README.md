# White-Box Ultrasound Age Regression

`usage_predict_feature_engineering` 是一个从零实现的、面向超声图像年龄回归的白盒/可解释传统机器学习工程。  
核心范式是：

图像预处理 -> 显式特征提取 -> 泄漏安全的 sklearn Pipeline -> 组别感知验证 -> 评估与解释

## 设计目标

- 不使用端到端 CNN，不依赖深度特征。
- 支持 whole-image 和 ROI/mask 两种特征模式。
- 支持 `image_only`、`metadata_only`、`image_plus_metadata`。
- 所有关键步骤都可配置：列映射、特征组、split、特征筛选、模型、调参。
- 严格按 `subject_id` 做 group split，防止同一受试者跨训练/验证/测试。

## 目录结构

```text
usage_predict_feature_engineering/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── features/
│   ├── models/
│   └── experiments/
├── dataio/
├── features/
├── preprocessing/
├── models/
├── selection/
├── evaluation/
├── explain/
├── scripts/
├── utils/
└── outputs/
```

## 已实现能力

### 特征

- 基础灰度统计：`mean/std/min/max/median/percentiles/skewness/kurtosis/entropy/energy`
- 清晰度/对比度：`laplacian_variance/tenengrad/rms_contrast/dynamic_range`
- 纹理特征：
  - `GLCM`
  - `GLRLM`
  - `GLSZM`
  - `NGTDM`（可开关）
  - `LBP histogram`
  - `HOG`（可选）
- 形态学特征（ROI/mask 可用时）：
  - `area/perimeter/eccentricity/compactness/solidity/major_axis/minor_axis`
  - thickness-like 统计
- radiomics 风格：
  - `global_handcrafted` 模式
  - `pyradiomics_roi` 模式，可选依赖，缺失时优雅降级

### 模型

- 线性：`LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
- 核方法：`SVR`, `KernelRidge`
- 树模型：`RandomForestRegressor`, `ExtraTreesRegressor`
- Boosting：`GradientBoostingRegressor`, `HistGradientBoostingRegressor`
- 可选依赖：`XGBoost`, `LightGBM`, `CatBoost`

### 验证与防泄漏

- `predefined` subject-level split
- `holdout` subject-level split
- `group k-fold`
- `nested group CV`
- 缺失值填补、编码、标准化、低方差过滤、相关性过滤、特征筛选、PCA、调参都在训练折内 fit

### 输出

每次实验默认在 `outputs/run_YYYYMMDD_HHMMSS_experiment_name_model_name/` 下生成。

例如：

- `run_20260401_170000_ta_healthy_holdout_ridge`
- `run_20260401_170500_ta_healthy_nested_cv_random_forest`

- 顶层只保留：
  - `run_summary.md`
  - `predictions_readable.csv`
  - `results_overview.json`
  - `config_used.yaml`
  - `run.log`
  - `command.sh`
- `tables/`
  - `features_raw.csv`
  - `metrics.json`
  - `predictions.csv`
  - `feature_importance.csv`
  - `selected_features.csv`
  - `split_info.json`
  - 其他明细表
- `figures/`
  - `predicted_vs_true.png`
  - `bland_altman.png`
  - `residual_hist.png`
  - `residual_vs_age.png`
  - `age_bin_error.png`
  - `feature_importance_top20.png`
- `models/`
  - `model.joblib` 或 `model_fold_*.joblib`
- `shap/`（仅当启用且安装了 `shap`）
  - `shap_importance_all.csv`
  - `shap_top20.csv`
  - `shap_top20.png`
  - nested CV 额外包含 `shap_top20_by_fold.csv`

## 安装

```bash
cd /home/szdx/LNX/usage_predict_feature_engineering
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

可选依赖：

- `pip install shap`
- `pip install pyradiomics SimpleITK`
- `pip install xgboost lightgbm catboost`

说明：

- 默认并行会优先走 `joblib` 的线程后端，以避免 Python 3.13 下偶发的 `ResourceTracker` 退出噪声。

## 数据接口

最终建模表每行一个样本，至少包含：

- `sample_id`
- `subject_id`
- `image_path`
- `age`
- 可选 `mask_path` / `roi_path`
- 可选 `sex` / `bmi` / 其他 metadata

字段名不写死，通过 YAML 映射：

```yaml
data:
  columns:
    sample_id: SampleID
    subject_id: SubjectID
    age: Age
    image_path: ImagePath
    mask_path: MaskPath
    split: Split
    extra:
      sex: Sex
      bmi: BMI
```

改 Excel/CSV 列名时，优先改配置，不改源码。

默认会过滤到 `18-100` 岁；如需调整，修改 `data.age_filter`。

图像尺寸默认会按 `image.preprocessing.resize` 统一缩放到 `256x256`。如需直接使用原始尺寸，可设置：

```yaml
image:
  preprocessing:
    use_original_size: true
```

也可以在命令行追加：

```bash
--override image.preprocessing.use_original_size=true
```

## 最小可运行示例

### 1. 只做特征提取

```bash
python scripts/extract_features.py \
  --config configs/default.yaml \
  --override data.metadata_path='"/path/to/metadata.csv"' \
  --override data.feature_mode='"image_only"' \
  --override paths.outputs_root='"/home/szdx/LNX/usage_predict_feature_engineering/outputs"'
```

### 2. Hold-out 基线实验

```bash
python scripts/run_experiment.py \
  --config configs/default.yaml \
  --config configs/models/ridge.yaml \
  --config configs/experiments/image_only_holdout.yaml \
  --override data.metadata_path='"/path/to/metadata.csv"'
```

### 3. Nested CV

```bash
python scripts/run_nested_cv.py \
  --config configs/default.yaml \
  --config configs/models/elasticnet.yaml \
  --config configs/experiments/nested_cv.yaml \
  --override data.metadata_path='"/path/to/metadata.xlsx"'
```

### 4. 融合 metadata

```bash
python scripts/run_experiment.py \
  --config configs/default.yaml \
  --config configs/models/random_forest.yaml \
  --config configs/features/metadata_fusion.yaml \
  --override data.metadata_path='"/path/to/metadata.csv"'
```

## 默认 workflow

### `scripts/run_experiment.py`

- 适用于 `holdout` / `predefined`
- 训练集上做 inner-CV 调参
- 先评估 `val`
- 再用 `train + val` 重训 final model
- 在 `test` 上输出最终指标与解释结果

### `scripts/run_nested_cv.py`

- 外层：性能估计
- 内层：调参
- 输出 pooled outer predictions、fold metrics、稳定特征榜单

## 配置说明

关键配置入口在 [configs/default.yaml](/home/szdx/LNX/usage_predict_feature_engineering/configs/default.yaml)。

重点字段：

- `data.feature_mode`
  - `image_only`
  - `metadata_only`
  - `image_plus_metadata`
- `feature_extraction.scopes`
  - `whole_image`
  - `roi`
- `split.strategy`
  - `predefined`
  - `holdout`
  - `group_kfold`
- `selection.enabled`
- `model.name`
- `search.enabled`

## 说明

- ROI 模式下，`morphology` 依赖 `mask_path`。
- `pyradiomics_roi` 依赖 `pyradiomics + SimpleITK`，未安装时会自动降级为 NaN 输出并打印 warning。
- 当前工程默认是 2D 灰度超声图像 workflow。
- 如果你的数据一位受试者有多张图，split 仍然按 `subject_id` 做，不会 image-level 泄漏。
