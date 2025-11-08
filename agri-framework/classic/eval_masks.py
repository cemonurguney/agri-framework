import os, glob, csv, cv2, numpy as np

def binarize(x):
    return (x>127).astype(np.uint8)

def iou_f1(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / (union + 1e-6)
    prec = inter / (pred.sum() + 1e-6)
    rec  = inter / (gt.sum() + 1e-6)
    f1 = 2*prec*rec/(prec+rec+1e-6)
    return float(iou), float(f1), float(prec), float(rec)

IMG_DIR   = "data/images"
PRED_DIR  = "data/masks"      # bizim VARI+Otsu ile kaydettiğimiz maskeler
GT_DIR    = "data/gt_masks"   # varsa GT maskeler (yoksa bu kısmı atla)

os.makedirs("outputs", exist_ok=True)
rows = [["filename","IoU","F1","Precision","Recall"]]

pred_paths = sorted(glob.glob(os.path.join(PRED_DIR,"*.png")))
n=0; s_iou=s_f1=s_p=s_r=0.0

for pp in pred_paths:
    name = os.path.basename(pp)
    gp = os.path.join(GT_DIR, name)
    if not os.path.exists(gp):
        continue
    pred = binarize(cv2.imread(pp,0))
    gt   = binarize(cv2.imread(gp,0))
    i,f1,p,r = iou_f1(pred, gt)
    rows.append([name, f"{i:.3f}", f"{f1:.3f}", f"{p:.3f}", f"{r:.3f}"])
    s_iou+=i; s_f1+=f1; s_p+=p; s_r+=r; n+=1

if n>0:
    rows.append(["AVERAGE", f"{s_iou/n:.3f}", f"{s_f1/n:.3f}", f"{s_p/n:.3f}", f"{s_r/n:.3f}"])

with open("outputs/metrics_vari.csv","w",newline="") as f:
    csv.writer(f).writerows(rows)
print("[✓] outputs/metrics_vari.csv yazıldı.")
