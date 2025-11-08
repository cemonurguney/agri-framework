# agri-framework

Akıllı tarım için makine öğrenimi/derin öğrenme çerçevesi. Veri hazırlama, eğitim, değerlendirme ve çıkarım boru hattını standartlaştırır.

## Özellikler
- Modüler veri hattı: `datasets/` ve `src/data/` ile kolay genişletme
- Deney yönetimi: `configs/*.yaml` ile çoğaltılabilir koşular
- Eğitim/validasyon/test döngüsü
- Metrikler ve kayıt: `wandb` veya yerel `runs/` klasörü
- Model dışa aktarımı: `.pth / .onnx`
- CLI arayüzü: `train.py`, `eval.py`, `infer.py`

---

## Hızlı Başlangıç

### Ortam
```bash
# Python venv (Windows PowerShell)
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
