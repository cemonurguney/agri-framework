import subprocess
import os

# İndirilecek klasör adı
target_folder = "1355868"
dataset_slug = "noahbadoa/plantnet-300k-images"

# Komutu oluştur
command = [
    "kaggle", "datasets", "download",
    "-d", dataset_slug,
    "-f", f"{target_folder}/",  # Klasörün içindeki tüm dosyaları filtrele
    "--unzip"  # İndirdikten sonra otomatik olarak aç (opsiyonel)
]

print(f"'{target_folder}' klasörünün içindeki dosyalar indiriliyor...")

try:
    # Komutu çalıştır
    # Bu, komutun çalışmasını bekler ve çıktıyı gösterir
    subprocess.run(command, check=True)

    print("\nİndirme işlemi tamamlandı.")

    # İndirilen ve açılan dosyalar genellikle mevcut dizininize gelir.
    # Eğer `--unzip` kullandıysanız, '1355868' adlı bir klasör oluşacaktır.
    print(f"İndirilen dosyaları/klasörü mevcut çalışma dizininizde bulabilirsiniz.")

except subprocess.CalledProcessError as e:
    print(f"Hata oluştu: Kaggle komutu çalıştırılamadı. Hata Kodu: {e.returncode}")
    print("Kaggle API'sinin kurulu olduğundan ve kimlik bilgilerinizin doğru ayarlandığından emin olun.")