#!/usr/bin/env python3
"""Headless copy of the proposal demo (auto-generated)."""
import os
OUTDIR=os.environ.get("OUTDIR","outputs"); os.makedirs(OUTDIR,exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as _plt
_n=[0]
def _save(*a,**k):
    _n[0]+=1; _plt.savefig(os.path.join(OUTDIR,f"fig_{_n[0]}.png"),dpi=120,bbox_inches="tight"); _plt.close("all")
_plt.show=_save
import os, time
import numpy as np
import torch
import torch.nn.functional as F
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import DetrImageProcessor, DetrForObjectDetection

try:                                       # faster, more robust model download — no token needed
    import hf_transfer  # noqa: F401
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# All knobs are env-overridable so the same file runs on a laptop CPU or a Colab GPU.
N_STEPS    = int(os.environ.get("N_STEPS", "150"))     # naive-attack steps
A_STEPS    = int(os.environ.get("A_STEPS", "200"))     # adaptive-attack steps
LAMBDA     = float(os.environ.get("LAMBDA", "8.0"))    # weight on "evade the defense"
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "100"))
IMG_SIZE   = (int(os.environ.get("IMG_H", "384")), int(os.environ.get("IMG_W", "1280")))  # (H, W) — KITTI aspect ratio
WINDOW, STRIDE, THRESHOLD = 64, 32, 6.0                # saliency-defense settings

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device).eval()
for _p in model.parameters():
    _p.requires_grad_(False)          # freeze: we only ever optimise the patch

MEAN = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
STD  = torch.tensor(processor.image_std,  device=device).view(1, 3, 1, 1)
def normalize(img01):
    return (img01 - MEAN) / STD

def load_image_as_tensor(source, is_url=True):
    if is_url:
        resp = requests.get(source, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(source).convert("RGB")
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device), img

CANDIDATE_URLS = [
    # A real KITTI frame (the standard AV benchmark); the rest are fallbacks.
    "https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/demo/data/kitti/000008.png",
    "https://upload.wikimedia.org/wikipedia/commons/7/7a/20180914_Ann_Arbor_traffic.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
]
image_tensor = pil_image = None
for _u in CANDIDATE_URLS:
    try:
        image_tensor, pil_image = load_image_as_tensor(_u, is_url=True)
        print("Loaded image:", _u); break
    except Exception as _e:
        print("  skip", _u, "->", repr(_e))
if image_tensor is None:
    raise RuntimeError("Could not load any demo image; use a local path (is_url=False).")
# image_tensor, pil_image = load_image_as_tensor("/content/your_kitti_frame.png", is_url=False)

def detect(image01, threshold=0.7):
    with torch.no_grad():
        outputs = model(pixel_values=normalize(image01))
    target_sizes = torch.tensor([IMG_SIZE], device=device)
    return processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

def show_detections(image01, results, title, ax=None):
    img_np = image01.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(np.clip(img_np, 0, 1))
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x0, y0, x1, y1 = box.tolist()
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, lw=2, edgecolor="lime", facecolor="none"))
        ax.text(x0, y0 - 5, f'{model.config.id2label[label.item()]}:{score:.2f}', color="lime",
                fontsize=8, bbox=dict(facecolor="black", alpha=0.5, pad=1))
    ax.set_title(title); ax.axis("off")
    if own: plt.show()

baseline_results = detect(image_tensor)
n_baseline = len(baseline_results["scores"])
show_detections(image_tensor, baseline_results, f"Baseline: {n_baseline} objects")
print("Baseline objects:", n_baseline)

def place_patch(canvas_size, patch, angle_rad, scale, tx, ty):
    C, ph, pw = patch.shape
    H, W = canvas_size
    canvas = torch.zeros(1, C, H, W, device=patch.device)
    mask   = torch.zeros(1, 1, H, W, device=patch.device)
    top, left = (H - ph) // 2, (W - pw) // 2
    canvas[0, :, top:top + ph, left:left + pw] = patch
    mask[0, :, top:top + ph, left:left + pw] = 1.0
    cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
    theta = torch.stack([
        torch.stack([scale * cos_a, -scale * sin_a, tx]),
        torch.stack([scale * sin_a,  scale * cos_a, ty]),
    ]).unsqueeze(0).to(patch.device)
    grid = F.affine_grid(theta, canvas.size(), align_corners=False)
    return F.grid_sample(canvas, grid, align_corners=False), F.grid_sample(mask, grid, align_corners=False)

def apply_eot_patch(image01, patch):
    ang = torch.empty(1, device=patch.device).uniform_(-20, 20) * (np.pi / 180)
    sca = torch.empty(1, device=patch.device).uniform_(0.8, 1.2)
    tx  = torch.empty(1, device=patch.device).uniform_(-0.6, 0.6)
    ty  = torch.empty(1, device=patch.device).uniform_(-0.6, 0.6)
    bri = torch.empty(1, device=patch.device).uniform_(0.8, 1.2)
    canvas, mask = place_patch(IMG_SIZE, patch, ang[0], sca[0], tx[0], ty[0])
    return image01 * (1 - mask) + torch.clamp(canvas * bri, 0, 1) * mask

def apply_patch_fixed(image01, patch):
    z = torch.zeros(1, device=patch.device)
    canvas, mask = place_patch(IMG_SIZE, patch, z[0], z[0] + 1, z[0], z[0])
    return image01 * (1 - mask) + torch.clamp(canvas, 0, 1) * mask

def hide_loss(image01):
    outputs = model(pixel_values=normalize(image01))
    probs = outputs.logits.softmax(-1)
    no_obj = outputs.logits.shape[-1] - 1
    return (1 - probs[..., no_obj]).mean()          # mean "there is an object" confidence

def compute_saliency(image01):
    img = image01.clone().detach().requires_grad_(True)
    outputs = model(pixel_values=normalize(img))
    probs = outputs.logits.softmax(-1)
    no_obj = outputs.logits.shape[-1] - 1
    (1 - probs[..., no_obj]).mean().backward()
    return img.grad.abs().sum(dim=1, keepdim=True).detach()

def flag_region(saliency):
    pooled = F.avg_pool2d(saliency, kernel_size=WINDOW, stride=STRIDE)
    idx = torch.argmax(pooled).item()
    w_cells = pooled.shape[-1]
    row, col = divmod(idx, w_cells)
    ratio = pooled.max().item() / (saliency.mean().item() + 1e-8)
    return col * STRIDE, row * STRIDE, ratio

def defense_panel(image01, ax, title):
    sal = compute_saliency(image01)
    x, y, ratio = flag_region(sal)
    flagged = ratio > THRESHOLD
    img_np = image01.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    ax.imshow(np.clip(img_np, 0, 1))
    color = "red" if flagged else "yellow"
    ax.add_patch(mpatches.Rectangle((x, y), WINDOW, WINDOW, lw=2.5, edgecolor=color, facecolor="none", ls="--"))
    verdict = "FLAGGED (attack caught)" if flagged else "not flagged (defense blind)"
    ax.set_title(f"{title}\nanomaly ratio={ratio:.1f} (thr {THRESHOLD}) -> {verdict}")
    ax.axis("off")
    return ratio, flagged

naive_patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=device, requires_grad=True)
opt = torch.optim.Adam([naive_patch], lr=0.05)
print(f"Training NAIVE patch ({N_STEPS} steps)...")
for step in range(N_STEPS):
    opt.zero_grad()
    loss = hide_loss(apply_eot_patch(image_tensor, naive_patch))
    loss.backward(); opt.step()
    with torch.no_grad(): naive_patch.clamp_(0, 1)
    if step % max(1, N_STEPS // 6) == 0 or step == N_STEPS - 1:
        print(f"  step {step:4d} | foreground conf {loss.item():.4f}")

naive_img = apply_patch_fixed(image_tensor, naive_patch).detach()
naive_results = detect(naive_img)
n_naive = len(naive_results["scores"])
naive_sal = compute_saliency(naive_img); _, _, ratio_naive = flag_region(naive_sal)
print(f"objects {n_baseline} -> {n_naive} | anomaly ratio {ratio_naive:.1f} (flagged={ratio_naive>THRESHOLD})")

adv_patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=device, requires_grad=True)
opt2 = torch.optim.Adam([adv_patch], lr=0.05)
print(f"Training ADAPTIVE patch ({A_STEPS} steps, lambda={LAMBDA})...")
for step in range(A_STEPS):
    opt2.zero_grad()
    patched = apply_eot_patch(image_tensor, adv_patch)
    outputs = model(pixel_values=normalize(patched))
    probs = outputs.logits.softmax(-1)
    no_obj = outputs.logits.shape[-1] - 1
    hide = (1 - probs[..., no_obj]).mean()
    # differentiable proxy for the defense's anomaly ratio (keep saliency flat):
    sal_grad = torch.autograd.grad(hide, patched, create_graph=True)[0]
    sal = sal_grad.abs().sum(dim=1, keepdim=True)
    pooled = F.avg_pool2d(sal, kernel_size=WINDOW, stride=STRIDE)
    evade = pooled.max() / (sal.mean() + 1e-8)      # push this DOWN
    loss = hide + LAMBDA * evade
    loss.backward(); opt2.step()
    with torch.no_grad(): adv_patch.clamp_(0, 1)
    if step % max(1, A_STEPS // 6) == 0 or step == A_STEPS - 1:
        print(f"  step {step:4d} | hide {hide.item():.4f} | evade-ratio {evade.item():.2f}")

adv_img = apply_patch_fixed(image_tensor, adv_patch).detach()
adv_results = detect(adv_img)
n_adaptive = len(adv_results["scores"])
adv_sal = compute_saliency(adv_img); _, _, ratio_adaptive = flag_region(adv_sal)
print(f"objects {n_baseline} -> {n_adaptive} | anomaly ratio {ratio_adaptive:.1f} (flagged={ratio_adaptive>THRESHOLD})")

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
show_detections(image_tensor, baseline_results, f"1) Clean: {n_baseline} objects", ax=axes[0])
r_n, f_n = defense_panel(naive_img, axes[1], f"2) Naive attack: {n_naive} objects")
r_a, f_a = defense_panel(adv_img,   axes[2], f"3) Adaptive attack: {n_adaptive} objects")
plt.tight_layout(); plt.show()

print("="*68)
print(f"{'':22}{'objects':>10}{'anomaly':>10}{'flagged?':>12}")
print(f"{'clean':22}{n_baseline:>10}{'-':>10}{'-':>12}")
print(f"{'naive attack':22}{n_naive:>10}{r_n:>10.1f}{str(f_n):>12}")
print(f"{'adaptive attack':22}{n_adaptive:>10}{r_a:>10.1f}{str(f_a):>12}")
print("="*68)
print("Interpretation:")
print("  - Naive attacker: objects hidden AND detected by the defense.")
print("  - Adaptive attacker: objects hidden, but the same defense fails to detect it.")
print("  - This failure under an adaptive attacker is the problem the proposed research addresses.")

def time_it(fn, n=15):
    for _ in range(3): fn()
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): fn()
    if device.type == "cuda": torch.cuda.synchronize()
    return (time.time() - t0) / n

base_t = time_it(lambda: detect(image_tensor))
def_t  = time_it(lambda: flag_region(compute_saliency(image_tensor)))
print(f"detector alone : {base_t*1000:6.1f} ms/frame ({1/base_t:5.1f} FPS)")
print(f"+ defense      : {def_t*1000:6.1f} ms/frame ({1/def_t:5.1f} FPS)")

import json as _json
_s=dict(n_baseline=n_baseline,n_naive=n_naive,n_adaptive=n_adaptive,
        ratio_naive=round(float(r_n),2),ratio_adaptive=round(float(r_a),2),
        flagged_naive=bool(f_n),flagged_adaptive=bool(f_a),threshold=THRESHOLD,
        base_ms=round(base_t*1000,1),def_ms=round(def_t*1000,1),device=device.type,
        n_steps=N_STEPS,a_steps=A_STEPS,lam=LAMBDA)
_json.dump(_s,open(os.path.join(OUTDIR,"summary.json"),"w"),indent=2)
print("SUMMARY:",_s)
