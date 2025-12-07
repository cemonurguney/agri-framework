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
python run.py train --model_name unet --run_name unet_exg_vari_v1 --size 512 --batch 4 --epochs 40 --lr 3e-4 --pos_weight 2.0
```
#### Tahmin
```bash
python run.py test --model_name unet --use_latest_run --size 512 --thr 0.65 --min_area 80 --test_tag exg_vari_v1
```
#### Pseudo Üretim
```bash
python classic/vari_otsu.py --in_dir data/images --out_dir outputs/pseudo --csv outputs/pseudo/area.csv --save_masks_dir data/masks_pseudo
```

