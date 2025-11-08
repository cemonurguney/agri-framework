import os, glob, csv, cv2, numpy as np

def to_uint8(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-6) * 255
    return x.astype(np.uint8)

def vari(img):  # (G - R) / (G + R - B)
    b,g,r = cv2.split(img.astype(np.float32)+1e-6)
    return (g - r) / (g + r - b + 1e-6)

def exg(img):   # 2G - R - B
    b,g,r = cv2.split(img.astype(np.float32))
    return 2*g - r - b

def mgrvi(img): # (G^2 - R^2)/(G^2 + R^2)
    b,g,r = cv2.split(img.astype(np.float32)+1e-6)
    return (g*g - r*r) / (g*g + r*r)

def bin(x): return (x>127).astype(np.uint8)
def iou_f1(p, g):
    p, g = p.astype(bool), g.astype(bool)
    inter = np.logical_and(p,g).sum()
    union = np.logical_or(p,g).sum()
    iou = inter/(union+1e-6)
    prec = inter/(p.sum()+1e-6); rec = inter/(g.sum()+1e-6)
    f1 = 2*prec*rec/(prec+rec+1e-6)
    return iou, f1, prec, rec

IMG_DIR="data/images"; GT_DIR="data/gt_masks"  # GT yoksa metrik kısmını atlar
OUT_CSV="outputs/compare_indices.csv"
os.makedirs("outputs", exist_ok=True)

indices = {"VARI":vari, "ExG":exg, "MGRVI":mgrvi}
rows=[["filename","index","IoU","F1","Precision","Recall","thr"]]

for ip in sorted(glob.glob(os.path.join(IMG_DIR,"*.jpg"))+glob.glob(os.path.join(IMG_DIR,"*.png"))):
    name=os.path.basename(ip); img=cv2.imread(ip)
    gt_path=os.path.join(GT_DIR, name.rsplit(".",1)[0]+".png")
    gt = bin(cv2.imread(gt_path,0)) if os.path.exists(gt_path) else None
    for idx_name, fn in indices.items():
        m = to_uint8(fn(img))
        m = cv2.GaussianBlur(m,(5,5),0)
        thr, mask = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if gt is None:
            rows.append([name, idx_name, "", "", "", "", int(thr)])
        else:
            i,f1,p,r = iou_f1(bin(mask), gt)
            rows.append([name, idx_name, f"{i:.3f}", f"{f1:.3f}", f"{p:.3f}", f"{r:.3f}", int(thr)])

with open(OUT_CSV,"w",newline="") as f: csv.writer(f).writerows(rows)
print("[✓] Yazıldı:", OUT_CSV)
