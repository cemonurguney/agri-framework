import argparse, pandas as pd, matplotlib.pyplot as plt, os

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="Girdi CSV dosyası (tam/relatif yol)")
ap.add_argument("--out", required=True, help="Çıkış PNG dosyası (tam/relatif yol)")
ap.add_argument("--max_rows", type=int, default=30, help="Tabloda gösterilecek maksimum satır")
args = ap.parse_args()

if not os.path.exists(args.csv):
    raise FileNotFoundError(f"CSV bulunamadı: {args.csv}")

df = pd.read_csv(args.csv)
# Çok uzunsa kısalt
if len(df) > args.max_rows:
    df = df.head(args.max_rows)

df.columns = [c.strip() for c in df.columns]

fig = plt.figure(figsize=(8, 0.5 + 0.35*len(df)))
fig.patch.set_visible(False); plt.axis("off")

tbl = plt.table(cellText=df.values,
                colLabels=df.columns,
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
tbl.scale(1, 1.4)

for (r,c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#777")
    if r == 0:
        cell.set_facecolor("#eaeaea")
        cell.set_text_props(weight="bold")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
plt.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0.1)
print(f"Kaydedildi: {args.out}")
