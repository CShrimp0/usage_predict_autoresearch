# ==========================================
# 全测试集热力图重叠可视化 (Grad-CAM / Grad-CAM++ / SmoothGrad)
# ==========================================

import os
import sys
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
from torchvision import transforms
from PIL import Image

# 尝试切到弹窗后端（必须在 pyplot 导入前设置）
def _set_gui_backend():
    if os.environ.get("DISPLAY", "") == "":
        return False
    for backend in ("TkAgg", "Qt5Agg"):
        try:
            matplotlib.use(backend, force=True)
            return True
        except Exception:
            continue
    return False

GUI_BACKEND_READY = _set_gui_backend()

import matplotlib.pyplot as plt

# 添加项目路径以导入自定义模块
sys.path.insert(0, "/home/szdx/LNX/usage_predict")
from model import get_model

# 设置中文字体（优先使用系统已安装字体）
from matplotlib import font_manager
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
font_candidates = [
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "SimHei",
    "Arial Unicode MS",
]
chosen_font = next((f for f in font_candidates if f in available_fonts), None)
if chosen_font:
    plt.rcParams["font.sans-serif"] = [chosen_font, "DejaVu Sans"]
    print(f"✅ 使用中文字体: {chosen_font}")
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    print("⚠️ 未检测到中文字体，中文可能无法正常显示")
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------
# 1. 配置路径
# ------------------------------------------

# A. 模型权重文件的完整路径
MODEL_FILE_PATH = "/home/szdx/LNX/usage_predict/outputs/ablation/01_baseline/run_20260108_115437/best_model.pth"

# B. 评估结果目录 (包含 predictions.json 的文件夹)
EVAL_RESULT_DIR = "/home/szdx/LNX/usage_predict/evaluation_results/01_baseline_run_20260108_115437"
PREDICTIONS_PATH = os.path.join(EVAL_RESULT_DIR, "predictions.json")

# C. 数据集路径
IMG_ROOT = "/home/szdx/LNX/data/TA/Healthy/Images/"
MASK_ROOT = "/home/szdx/LNX/data/TA/Healthy/Masks/"

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")
print(f"Model Path: {MODEL_FILE_PATH}")
print(f"Eval Dir:   {EVAL_RESULT_DIR}")

# ------------------------------------------
# 2. 核心类与辅助函数
# ------------------------------------------


class GradCAM:
    """Grad-CAM 核心逻辑"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        # 前向传播
        output = self.model(x)

        # 反向传播
        self.model.zero_grad()
        output.backward()

        # 生成热力图
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()

        # 加权
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)  # ReLU
        heatmap = heatmap.detach().cpu().numpy()

        # 归一化 (防止除零)
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap


class GradCAMPlusPlus:
    """Grad-CAM++ 实现 - 改进版"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        output = self.model(x)
        self.model.zero_grad()
        output.backward()

        gradients = self.gradients
        activations = self.activations

        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)

        alpha_numer = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_numer / alpha_denom

        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)

        heatmap = heatmap.cpu().numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap


class SmoothGrad:
    """SmoothGrad 实现 - 通过添加噪声平滑梯度"""
    def __init__(self, model, n_samples=30, noise_level=0.15):
        self.model = model
        self.n_samples = n_samples
        self.noise_level = noise_level

    def __call__(self, x):
        stdev = self.noise_level * (x.max() - x.min())
        total_gradients = torch.zeros_like(x)

        for _ in range(self.n_samples):
            noise = torch.randn_like(x) * stdev
            noisy_input = (x + noise).requires_grad_(True)

            output = self.model(noisy_input)
            self.model.zero_grad()
            output.backward()

            if noisy_input.grad is not None:
                total_gradients += noisy_input.grad.detach()

        avg_gradients = total_gradients / self.n_samples
        heatmap = torch.mean(torch.abs(avg_gradients), dim=1).squeeze()
        heatmap = heatmap.cpu().numpy()

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap



def get_processed_data(img_name):
    """加载图像并预处理"""
    img_path = os.path.join(IMG_ROOT, img_name)
    if not os.path.exists(img_path):
        return None, None

    # 读取用于显示的原始图
    raw_image = Image.open(img_path).convert("RGB")
    raw_image = np.array(raw_image)

    # 预处理用于模型推理（严格按照训练时的预处理流程）
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(Image.fromarray(raw_image)).unsqueeze(0).to(device)

    return raw_image, input_tensor


def get_mask_overlay(img_name, raw_image):
    """获取 Mask 叠加图 (红色半透明)"""
    # 尝试多种后缀
    base_name = os.path.splitext(img_name)[0]
    potential_masks = [img_name, base_name + ".png", base_name + ".jpg"]

    mask = None
    for m_name in potential_masks:
        m_path = os.path.join(MASK_ROOT, m_name)
        if os.path.exists(m_path):
            mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            break

    if mask is None:
        return raw_image  # 无 mask 返回原图

    mask = cv2.resize(mask, (raw_image.shape[1], raw_image.shape[0]))

    # 红色覆盖
    overlay = raw_image.copy()
    overlay[mask > 0] = [255, 0, 0]  # 红色

    # 混合
    alpha = 0.3
    result = cv2.addWeighted(overlay, alpha, raw_image, 1 - alpha, 0)
    return result


def get_heatmap_only(heatmap, target_size):
    """生成纯热力图（不叠加原图）"""
    heatmap = cv2.resize(heatmap, target_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def get_heatmap_overlay(raw_image, heatmap):
    """获取热力图叠加图"""
    heatmap = cv2.resize(heatmap, (raw_image.shape[1], raw_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # BGR -> RGB

    overlay = cv2.addWeighted(heatmap, 0.5, raw_image, 0.5, 0)
    return overlay


def load_model(pth_file_path):
    """严格按照训练脚本加载模型（直接使用model.py中的get_model函数）"""
    if not os.path.exists(pth_file_path):
        print(f"Error: 模型文件不存在: {pth_file_path}")
        return None

    print(f"Loading model from: {pth_file_path}")

    # 加载checkpoint获取配置信息
    checkpoint = torch.load(pth_file_path, map_location=device, weights_only=False)

    # 读取训练时的配置
    model_name = "resnet50"
    dropout = 0.6
    if isinstance(checkpoint, dict):
        if "args" in checkpoint and isinstance(checkpoint["args"], dict):
            train_args = checkpoint["args"]
            model_name = train_args.get("model", model_name)
            dropout = train_args.get("dropout", dropout)
            print(f"检测到训练配置: model={model_name}, dropout={dropout}")
        elif "model_name" in checkpoint:
            model_name = checkpoint.get("model_name", model_name)
            print(f"检测到训练配置: model={model_name}, dropout={dropout}")
        else:
            print(f"使用默认配置: model={model_name}, dropout={dropout}")
    else:
        print(f"使用默认配置: model={model_name}, dropout={dropout}")

    # ✅ 使用get_model函数创建模型（与训练时完全一致）
    model = get_model(model_name, pretrained=False, dropout=dropout)
    print(f"模型结构已创建: {model_name}")

    try:
        # 提取state_dict
        state_dict = None
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                print("从checkpoint['model_state_dict']加载权重")
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                print("从checkpoint['state_dict']加载权重")
            elif "model" in checkpoint and hasattr(checkpoint["model"], "state_dict"):
                state_dict = checkpoint["model"].state_dict()
                print("从checkpoint['model']加载权重")
        elif hasattr(checkpoint, "keys"):
            # 直接保存的state_dict
            state_dict = checkpoint
            print("从checkpoint本体加载权重")

        if state_dict is None:
            print("Error: 未找到可用的state_dict")
            return None

        # 处理 DataParallel 的 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v

        # 加载权重（允许部分匹配，输出缺失/多余键方便排查）
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"⚠️ 缺失参数: {len(missing)}")
        if unexpected:
            print(f"⚠️ 多余参数: {len(unexpected)}")
        if not missing and not unexpected:
            print("✅ 模型权重加载成功！")
        else:
            print("✅ 模型权重已加载（存在部分不匹配参数）")

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    model.to(device)
    model.eval()

    # 验证模型输出
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224).to(device)
        test_output = model(test_input)
        try:
            test_val = test_output.item()
        except Exception:
            test_val = float(test_output.flatten()[0].item())
        print(f"模型输出测试: shape={tuple(test_output.shape)}, value={test_val:.2f}")

    return model


def find_target_layer(model):
    """查找目标卷积层"""
    target_layer = None
    for name, module in model.named_modules():
        if "layer4" in name and isinstance(module, nn.Conv2d):
            target_layer = module
            break
    if target_layer is None:
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break
    return target_layer


def load_test_filenames(pred_path):
    if not os.path.exists(pred_path):
        print(f"Error: 找不到 predictions.json: {pred_path}")
        return []
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    filenames = data.get("filenames", [])
    return filenames


def filter_valid_filenames(filenames):
    valid = []
    for name in filenames:
        raw_img, _ = get_processed_data(name)
        if raw_img is None:
            continue
        valid.append(name)
    return valid


def aggregate_heatmaps(method, filenames, label):
    heatmap_sum = None
    count = 0
    for idx, name in enumerate(filenames, 1):
        raw_img, input_tensor = get_processed_data(name)
        if input_tensor is None:
            continue
        with torch.enable_grad():
            heatmap = method(input_tensor)
        if heatmap_sum is None:
            heatmap_sum = np.zeros_like(heatmap, dtype=np.float64)
        heatmap_sum += heatmap
        count += 1
        if idx % 50 == 0:
            print(f"{label}: {idx}/{len(filenames)}")
    if count == 0:
        return None, 0
    avg_heatmap = heatmap_sum / count
    if avg_heatmap.max() > 0:
        avg_heatmap = avg_heatmap / avg_heatmap.max()
    return avg_heatmap, count


def main():
    # 加载模型（用于Grad-CAM）
    model = load_model(MODEL_FILE_PATH)

    if model:
        target_layer = find_target_layer(model)
        print(f"Target Layer: {target_layer}")
        grad_cam = GradCAM(model, target_layer)
    else:
        grad_cam = None

    # 读取测试集文件名
    all_test_filenames = load_test_filenames(PREDICTIONS_PATH)
    valid_filenames = filter_valid_filenames(all_test_filenames)
    print(f"\n测试集样本数: {len(valid_filenames)}")

    if not valid_filenames:
        print("Error: 无可用测试图像，无法生成重叠热力图")
        return

    print("\n初始化三种方法模型...")

    methods = {}

    if grad_cam is not None:
        methods["Grad-CAM"] = grad_cam
    else:
        print("⚠️ Grad-CAM 未初始化，跳过")

    # Grad-CAM++
    model_gcpp = load_model(MODEL_FILE_PATH)
    if model_gcpp:
        grad_cam_pp = GradCAMPlusPlus(model_gcpp, find_target_layer(model_gcpp))
        methods["Grad-CAM++"] = grad_cam_pp
    else:
        print("⚠️ Grad-CAM++ 模型加载失败，跳过")

    # SmoothGrad
    model_sg = load_model(MODEL_FILE_PATH)
    if model_sg:
        smooth_grad = SmoothGrad(model_sg, n_samples=30, noise_level=0.15)
        methods["SmoothGrad"] = smooth_grad
    else:
        print("⚠️ SmoothGrad 模型加载失败，跳过")

    if not methods:
        print("Error: 无可用方法，停止绘制")
        return

    print("\n生成全测试集热力图重叠...")

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6))
    if len(methods) == 1:
        axes = [axes]
    fig.suptitle("全测试集热力图统计结果", fontsize=16, fontweight="bold", y=0.95)

    for ax, (name, method) in zip(axes, methods.items()):
        avg_heatmap, count = aggregate_heatmaps(method, valid_filenames, name)
        if avg_heatmap is None:
            ax.set_title(f"{name}\n无可用样本", fontsize=12)
            ax.axis("off")
            continue
        heatmap_only = get_heatmap_only(avg_heatmap, (224, 224))
        ax.imshow(heatmap_only)
        ax.set_title(f"{name}\nN={count}", fontsize=12, fontweight="bold")
        ax.axis("off")
        print(
            f"{name}: N={count} | mean={avg_heatmap.mean():.4f} "
            f"| std={avg_heatmap.std():.4f} | max={avg_heatmap.max():.4f}"
        )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_dir = "/home/szdx/LNX/usage_predict/show_image/image"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "testset_heatmap_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    print(f"\n✅ 已保存: {output_path}")


if __name__ == "__main__":
    main()
