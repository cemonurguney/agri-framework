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
