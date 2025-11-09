# agri-framework

Akıllı tarım için makine öğrenimi/derin öğrenme çerçevesi. Veri hazırlama, eğitim, değerlendirme ve çıkarım boru hattını standartlaştırır.

## Özellikler
- Modüler veri hattı: `datasets/` ve `src/data/` ile kolay genişletme
- Eğitim/validasyon/test döngüsü
- CLI arayüzü: `train.py`, `infer.py`

---

## Hızlı Başlangıç

### Ortam
```bash
# Python venv (Windows PowerShell)
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

### Klasik Hat (VARI+Otsu)
```bash
python .\classic\vari_otsu.py --in_dir .\data\images --out_dir .\outputs\samples --csv .\outputs\area.csv --save_masks_dir .\data\masks
```

### Derin öğrenme (U-Net + EfficientNet-B3)
#### Eğitim
```bash
python .\dl\train_smp.py --img_dir .\data\images --mask_dir .\data\masks --out_dir .\outputs --size 384 --batch 4 --epochs 40 --lr 3e-4
```
#### Tahmin
```bash
python .\dl\infer_smp.py --img_dir .\data\test_images --out_dir .\outputs\pred_dl --size 384 --model .\outputs\model_smp.pt
```

