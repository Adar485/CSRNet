# 🎯 CSRNet — Kalabalık Analiz Sistemi

Gerçek zamanlı kalabalık sayma ve ısı haritası analizi. CSRNet derin öğrenme modeli ile video karelerini işleyerek anlık kişi sayısı tahmini ve yoğunluk haritası üretir.

---

## 🖥️ Arayüz

- Sol ekran: orijinal video
- Sağ ekran: model çıktısı — saf ısı haritası (kırmızı = yoğun, mavi = seyrek)
- Anlık kişi sayacı ve kalabalık seviye göstergesi
- Kare başına kişi grafiği

---

## 📁 Proje Yapısı

```
CSRNet/
├── model.py          # CSRNet mimarisi (VGG-16 + Dilated Conv)
├── app.py            # Flask backend + SSE stream
├── requirements.txt  # Bağımlılıklar
├── static/
│   ├── uploads/      # Yüklenen videolar
│   └── outputs/      # İşlenmiş videolar
└── templates/
    └── index.html    # Web arayüzü
```

---

## ⚙️ Kurulum

### 1. Gereksinimler

- Python 3.10+
- NVIDIA GPU (CUDA 12.4 önerilir)

### 2. Bağımlılıkları Kur

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python scipy matplotlib pillow flask
```

### 3. Model Ağırlıklarını İndir

Aşağıdaki linkten `weights.pth` dosyasını indirip proje klasörüne koy:

🔗 [weights.pth — Google Drive](https://drive.google.com/file/d/1Ti_DQ0lYXCLqhH9nGwtMMWH5Zt5xiJA2/view)

> Ağırlıklar ShanghaiTech Part A veri seti üzerinde eğitilmiştir.

### 4. Uygulamayı Başlat

```bash
python app.py
```

Tarayıcıda aç: **http://127.0.0.1:5000**

---

## 🚀 Kullanım

1. Video yükle (MP4, AVI, MOV)
2. **ANALİZ BAŞLAT** butonuna bas
3. Sol ekranda orijinal video oynar, sağ ekranda ısı haritası canlı güncellenir
4. İstersen **⏹ DURDUR** ile analizi durdurabilirsin
5. Analiz tamamlanınca istatistikler ve grafik görünür

---

## 🌡️ Kalabalık Seviyeleri

| Seviye | Kişi Sayısı | Renk |
|--------|-------------|------|
| Kalabalık Değil | < 50 | 🟢 |
| Orta Seviye Kalabalık | 50 – 250 | 🟡 |
| Kalabalık | 250 – 1000 | 🟠 |
| Aşırı Kalabalık | 1000+ | 🔴 |

---

## 🧠 Model Mimarisi

**CSRNet** (Congested Scene Recognition Network) iki bölümden oluşur:

- **Frontend:** VGG-16'nın ilk 10 konvolüsyon katmanı (ImageNet ağırlıkları)
- **Backend:** Dilated Convolution katmanları (dilation rate = 2)
- **Çıktı:** Yoğunluk haritası (density map) → toplam kişi sayısı

> Li et al., *"CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes"*, CVPR 2018

---

## 🛠️ Teknolojiler

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)

---

## 👤 Geliştirici

**Adar** — Fırat Üniversitesi, Bilgisayar Mühendisliği
