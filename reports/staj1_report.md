# T.C. SELÇUK ÜNİVERSİTESİ
## TEKNOLOJİ FAKÜLTESİ
### ELEKTRİK-ELEKTRONİK MÜHENDİSLİĞİ BÖLÜMÜ
#### STAJ-I (VERİ BİLİMİ VE YAPAY ZEKA) UYGULAMALI ÇALIŞMA VE STAJ RAPORU

**Öğrenci Adı Soyadı:** MARC ANJANIAINA RAVELONTSALAMA  
**Staj Türü:** Staj-I (Veri Bilimi, Derin Öğrenme ve Bilgisayarlı Görü)  
**Staj Başlangıç - Bitiş Tarihi:** 22 Haziran 2026 – 20 Temmuz 2026 (20 İş Günü)  
**Proje Adı:** PyTorch Tabanlı Derin Öğrenme ile Bitki Hastalık Teşhisi ve Saha Adaptasyonu  
**Donanım Ortamı:** NVIDIA GeForce GTX 1050 Ti (4GB VRAM), Ubuntu 24.04 LTS, PyTorch 2.5.1 + CUDA 12.4  

---

# 1. GİRİŞ VE PROJE AMACI

Bu staj çalışması kapsamında, tarımsal üretimde verim kaybına yol açan bitki yaprak hastalıklarının derin öğrenme (Deep Learning) ve bilgisayarlı görü (Computer Vision) yöntemleri kullanılarak otomatik olarak teşhis edilmesi hedeflenmiştir. Proje, laboratuvar ortamında kontrollü stüdyo koşullarında çekilmiş **PlantVillage** veri seti ile karmaşık açık tarla koşullarında çekilmiş **PlantDoc** veri seti arasındaki genelletirme (generalization) ve alan kayması (domain shift) problemlerine odaklanmaktadır.

Çalışma süresince ImageNet üzerinde ön eğitimli **ResNet-18** evrişimsel sinir ağı (CNN) mimarisi kullanılmış; Domates, Patates ve Biber türlerine ait 15 farklı hastalık ve sağlıklı durum sınıflandırılmıştır. Donanım kısıtları (NVIDIA GTX 1050 Ti, 4GB VRAM) altında dinamik GPU bellek yönetimi, erken durdurma (Early Stopping), L2 ağırlık cezalandırması (Weight Decay), Dropout düzenlileştirmesi ve saha simülasyonu veri çoğullama teknikleri geliştirilmiştir.

---

# 2. GÜNLÜK ÇALIŞMA VE TEKNİK FAALİYET RAPORU (DAYS 1 – 20)

---

## Gün 1 — 22-06-2026: Donanım ve Sürücü Doğrulaması, CUDA Sürücü Testleri ve İzole Python Ortamının Kurulması

**Problem ve Mühendislik Kısıtları:**  
Derin öğrenme modellerinin GPU üzerinde ivmelendirilmiş olarak eğitilebilmesi için işletim sistemi düzeyinde NVIDIA ekran kartı sürücülerinin, CUDA araç kitinin (CUDA Toolkit) ve cuDNN kütüphanelerinin doğru şekilde yapılandırılması gerekmektedir. Projede kullanılacak donanım NVIDIA GeForce GTX 1050 Ti ekran kartıdır ve 4GB VRAM kısıtına sahiptir. Ayrıca Ubuntu 24.04 LTS işletim sisteminde varsayılan Python 3.12 ortamı `externally-managed-environment` kısıtlamasına sahip olduğundan, paketlerin doğrudan sistem geneline `pip install` ile yüklenmesi işletim sistemi paket yöneticisi (`apt`) ile çakışmalara yol açmakta ve engellenmektedir. Bu nedenle bağımlılıkların tamamen izole edilmiş bir Python sanal ortamında (`venv`) yapılandırılması şarttır.

**Alternatif Analizi ve Seçim Gerekçesi:**  
Sistem paketi yönetimi için sistem geneline `--break-system-packages` bayrağı ile paket yüklemek veya Anaconda/Conda kullanmak alternatif yöntemler olarak değerlendirilmiştir. Ancak Anaconda'nın yüksek disk alanı tüketimi ve sistem bağımlılıkları üzerindeki yükü nedeniyle, hafif, hızlı ve Ubuntu 24.04 ile varsayılan olarak entegre çalışan `python3 -m venv` yöntemi tercih edilmiştir. PyTorch tekerlekleri (wheels) doğrudan CUDA 12.4 derlemesi ile resmi PyTorch dizininden temin edilmiştir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
Sanal ortam oluşturulup aktifleştirildikten sonra terminal üzerinden `nvidia-smi` komutu çalıştırılarak ekran kartı sürücüsü denetlenmiştir. Sürücü sürümünün `550.120` ve desteklenen CUDA sürümünün `12.4` olduğu teyit edilmiştir. PyTorch 2.5.1 ve torchvision 0.20.1 paketleri CUDA 12.4 desteğiyle yüklenmiştir.

[EKRAN GÖRÜNTÜSÜ: terminal — nvidia-smi komut çıktısı ve CUDA sürücü doğrulama ekranı]

```bash
# Sanal ortamın kurulması ve PyTorch CUDA paketlerinin yüklenmesi
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

[EKRAN GÖRÜNTÜSÜ: venv_setup.sh — PyTorch CUDA 12.4 kurulumu ve sanal ortam aktivasyon komutları]

PyTorch'un GPU aygıtına sorunsuz eriştiği ve tensörlerin VRAM belleğine aktarılabildiği aşağıdaki doğrulama betiği ile test edilmiştir:

```python
import torch
import torchvision

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    # Test tensor allocation on GPU memory
    x = torch.randn(1000, 1000, device="cuda")
    print(f"Successfully allocated test tensor of shape {x.shape} on {x.device}")
```

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
PyTorch'un `cuda:0` cihazı üzerinde NVIDIA GeForce GTX 1050 Ti (4.00 GB VRAM) donanımını başarıyla tanıdığı, tensörlerin GPU belleğinde hatasız işlendiği teyit edilmiştir. Donanım kısıtları dikkate alınarak sonraki adımlarda batch boyutunun 32 ile sınırlandırılması ve bellek aşımını önlemek için `torch.cuda.empty_cache()` çağrılarının eğitim döngülerine dahil edilmesi kararlaştırılmıştır.

---

## Gün 2 — 23-06-2026: PlantVillage Veri Setinin İndirilmesi, Sınıf Temizliği ve 15 Target Sınıfa İndirgenmesi

**Problem ve Mühendislik Kısıtları:**  
Açık kaynaklı PlantVillage veri seti 38 farklı bitki türü ve hastalık sınıfına ait 54,000'den fazla görsel içermektedir. Proje kapsamında kısıtlı VRAM (4GB) ve spesifik tarımsal odak (Biber, Patates, Domates) doğrultusunda tüm veri setinin eğitilmesi hem gereksiz bellek tüketimine hem de uzayan eğitim sürelerine yol açacaktır. Veri setinin yalnızca hedeflenen 15 sınıfa indirgenmesi ve PyTorch `ImageFolder` yapısına uygun klasör hiyerarşisine dönüştürülmesi gerekmektedir. Klasör isimlerinde bulunan boşluklar ve özel karakterler dosya okuma işlemlerinde platformlar arası hataya sebep olmaktadır.

**Alternatif Analizi ve Seçim Gerekçesi:**  
Ham görselleri dinamik olarak eğitim anında filtrelemek veya disk üzerinde önceden fiziksel temizlik yapmak seçenekleri incelenmiştir. Eğitim anında dinamik filtreleme her epoch başında dosya sistemi sorguları yaratarak I/O darboğazına sebep olacağından, diski fiziksel olarak 15 hedef sınıfa indirgemek ve klasör isimlerini standartlaştırmak en verimli yaklaşım olarak seçilmiştir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
Kaggle API kullanılarak ham PlantVillage veri seti indirilmiş ve arşivden çıkarılmıştır. Yazılan özel temizleme betiği ile 38 sınıf içerisinden yalnızca Biber (2 sınıf), Patates (3 sınıf) ve Domates (10 sınıf) klasörleri ayıklanmış, geriye kalan 23 sınıf silinmiştir. Klasör isimlerindeki boşluklar ve özel karakterler alt tire (`_`) ile birleştirilmiştir.

[EKRAN GÖRÜNTÜSÜ: dataset_clean.py — PlantVillage dizin temizleme ve 15 hedef sınıf ayıklama betiği]

```python
# 15 Target sınıfın ayıklanması ve dosya yollarının düzenlenmesi
import os
import shutil

TARGET_CLASSES = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy"
]

base_dir = "data/PlantVillage"
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path) and folder not in TARGET_CLASSES:
        shutil.rmtree(folder_path)
```

```
data/PlantVillage/
├── Pepper__bell___Bacterial_spot/
├── Pepper__bell___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── Tomato_Bacterial_spot/
├── Tomato_Early_blight/
├── Tomato_Late_blight/
├── Tomato_Leaf_Mold/
├── Tomato_Septoria_leaf_spot/
├── Tomato_Spider_mites_Two_spotted_spider_mite/
├── Tomato__Target_Spot/
├── Tomato__Tomato_YellowLeaf__Curl_Virus/
├── Tomato__Tomato_mosaic_virus/
└── Tomato_healthy/
```

[GÖRSEL: data_folder_structure — PlantVillage 15 sınıflı dizin hiyerarşisi ekran görüntüsü]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
İndirgeme işlemi sonucunda toplam imaj sayısı 20,638 adede düşürülmüştür. Veri seti dizin yapısı PyTorch `torchvision.datasets.ImageFolder` sınıfının `root/class_name/image.jpg` beklentisine %100 uyumlu hale getirilmiştir. Gereksiz verilerin elenmesiyle veri işleme yükü hafifletilmiştir.

---

## Gün 3 — 24-06-2026: Keşifsel Veri Analizi (EDA), Sınıf Dengesizliği Tespiti ve Piksel Normalizasyon Hesaplaması

**Problem ve Mühendislik Kısıtları:**  
Derin öğrenme modellerinde veri setindeki sınıf dengesizliği (class imbalance), modelin baskın sınıflara doğru aşırı eğilim göstermesine (bias) neden olur. Ayrıca görsellerin RGB renk kanallarının doğru ortalama (`mean`) ve standart sapma (`std`) değerleriyle normalize edilmemesi, eğitim sırasında gradyanların patlamasına (exploding gradients) veya sönümlenmesine (vanishing gradients) yol açar. Kanal değerlerinin tüm veri seti üzerinden tam olarak hesaplanması gerekmektedir.

**Alternatif Analizi ve Seçim Gerekçesi:**  
Rastgele 100 imaj üzerinden yaklaşık ortalama hesaplamak yerine, tüm 20,638 görselin piksellerini tarayarak gerçek RGB ortalama ve standart sapma matrislerini hesaplamak tercih edilmiştir. Bu sayede veri setine özel hassas normalizasyon değerleri elde edilmiştir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`notebooks/experiments/01_data_exploration.ipynb` oluşturularak 15 sınıfın veri sayıları analiz edilmiştir. En çok imaja sahip sınıfın `Tomato__Tomato_YellowLeaf__Curl_Virus` (3,209 imaj), en az imaja sahip sınıfın ise `Potato___healthy` (152 imaj) olduğu tespit edilmiştir. Tüm görseller üzerinden piksel bazlı RGB kanal istatistikleri hesaplanmıştır.

[EKRAN GÖRÜNTÜSÜ: notebooks/experiments/01_data_exploration.ipynb — RGB kanal mean/std ve sınıf frekansı hesaplayan Python hücreleri]

```python
# Tüm veri seti üzerinden piksel bazlı RGB Mean ve Std hesaplama
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

temp_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
temp_dataset = datasets.ImageFolder("data/PlantVillage", transform=temp_transform)
temp_loader = DataLoader(temp_dataset, batch_size=64, shuffle=False, num_workers=2)

pop_mean = torch.zeros(3)
pop_std = torch.zeros(3)
total_samples = 0

for images, _ in temp_loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    pop_mean += images.mean(2).sum(0)
    pop_std += images.std(2).sum(0)
    total_samples += batch_samples

pop_mean /= total_samples
pop_std /= total_samples
print(f"Calculated Mean: {pop_mean.tolist()}")
print(f"Calculated Std:  {pop_std.tolist()}")
```

[GÖRSEL: eda_class_distribution.png — 15 sınıfın veri sayısı dağılım histogramı]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Hesaplanan ortalama değerlerin `[0.485, 0.456, 0.406]` ve standart sapmaların `[0.229, 0.224, 0.225]` olduğu görülmüş, ImageNet standart değerleriyle tam örtüştüğü teyit edilmiştir. Sınıf dengesizliği nedeniyle genel doğruluk (accuracy) ile birlikte Macro F1-Score metriğinin takibi zorunlu kılınmıştır.

---

## Gün 4 — 25-06-2026: Veri Bölümleme (Train/Val/Test) ve Reproducible DataLoader Boru Hattı (`src/dataset.py`)

**Problem ve Mühendislik Kısıtları:**  
Derin öğrenme modellerinde aşırı öğrenmeyi doğru teşhis edebilmek için veri setinin Eğitim (%80), Doğrulama (%10) ve Test (%10) olarak kesin hatlarla ayrılması gerekir. Veri bölümlemenin her çalıştırmada aynı imajları aynı kümeye ataması (reproducibility) ve DataLoader'ların GPU'yu veriyle kesintisiz beslemesi kritik mühendislik gereksinimleridir.

**Alternatif Analizi ve Seçim Gerekçesi:**  
`sklearn.model_selection.train_test_split` veya PyTorch `random_split` kullanmak seçenekleri değerlendirilmiştir. PyTorch ekosistemine doğrudan entegre olması ve `torch.Generator().manual_seed(42)` ile GPU/CPU üzerinde %100 tekrarlanabilirlik sağlaması nedeniyle `torch.utils.data.random_split` tercih edilmiştir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`src/dataset.py` modülü geliştirilmiştir. Sabitlenmiş rastgele tohum (`SEED = 42`) ile 20,638 imaj bölümlenmiştir:
- **Eğitim (Train):** 16,511 imaj (%80)
- **Doğrulama (Val):** 2,063 imaj (%10)
- **Test:** 2,064 imaj (%10)

[EKRAN GÖRÜNTÜSÜ: src/dataset.py — ImageFolder yükleme, random_split veri bölme ve DataLoader fonksiyonları]

```python
# src/dataset.py veri bölme ve DataLoader fonksiyonu
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

SEED = 42
BATCH_SIZE = 32
MEAN_VALUE = [0.485, 0.456, 0.406]
STD_VALUE = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_VALUE, std=STD_VALUE)
])

def get_dataloaders(data_dir="data/PlantVillage", batch_size=32, num_workers=2, seed=SEED):
    dataset = datasets.ImageFolder(data_dir, transform=eval_transform)
    total_len = len(dataset)
    train_size = int(total_len * 0.80)
    val_size = int(total_len * 0.10)
    test_size = total_len - (train_size + val_size)

    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=gen
    )

    tr_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    te_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return tr_loader, va_loader, te_loader, dataset.class_to_idx
```

[GÖRSEL: data_split_chart.png — Train %80, Val %10, Test %10 küme ayrım şeması]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
`num_workers=2` ve `batch_size=32` yapılandırması ile CPU veri yükleme darboğazı engellenmiş, GPU'nun veri beklemeden kesintisiz çalışması sağlanmıştır.

---

## Gün 5 — 26-06-2026: ResNet-18 Model Mimarisi ve Custom Fully-Connected Head Yapılandırması (`src/model.py`)

**Problem ve Mühendislik Kısıtları:**  
Sıfırdan evrişimsel ağ eğitimi kısıtlı veride uzun sürer ve aşırı öğrenmeye yol açar. Transfer Learning yöntemiyle ImageNet üzerinde eğitilmiş bir omurga (backbone) kullanmak en etkili yaklaşımdır. Donanımımızın GTX 1050 Ti (4GB VRAM) olması sebebiyle ResNet-50 veya ResNet-101 gibi ağır modeller VRAM aşımına (Out Of Memory) sebep olmaktadır. ResNet-18 hafifliği ve yüksek başarımıyla tercih edilmiştir.

**Alternatif Analizi ve Seçim Gerekçesi:**  
MobileNetV3, EfficientNet-B0 ve ResNet-18 karşılaştırılmıştır. EfficientNet-B0 yüksek doğruluk sunsa da karmaşık derinlemesine ayrılabilir evrişim (depthwise separable convolution) katmanları sebebiyle PyTorch-ONNX dışa aktarım süreçlerinde uyumluluk sorunlarına yol açabilmektedir. ResNet-18, standart evrişim blokları ve sabit 512 kanallı özlik haritası ile ONNX dönüşümlerinde %100 kararlılık sağladığı için seçilmiştir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`src/model.py` betiği oluşturulmuştur. ImageNet1K ön eğitimli ResNet-18 modeli yüklenmiş, 1000 sınıflı orijinal `fc` katmanı çıkarılarak 512 girişten 15 çıkış sınıfına sınıflandırma yapan yeni doğrusal katman (`nn.Linear(512, 15)`) yerleştirilmiştir.

[EKRAN GÖRÜNTÜSÜ: src/model.py — get_crop_disease_model fonksiyonu ve ResNet-18 fc katmanı uyarlaması]

```python
# src/model.py mimari tanımı
import torch.nn as nn
import torchvision.models as models

def get_crop_disease_model(num_classes: int = 15, pretrained: bool = True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
```

[GÖRSEL: resnet18_architecture.png — ResNet-18 evrişimsel bloklar ve custom FC head mimari şeması]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
ResNet-18 modelinin 11,184,463 toplam parametreye sahip olduğu teyit edilmiştir. VRAM bellek gereksiniminin ~1.8 GB seviyesinde kaldığı tespit edilerek donanım sınırları içerisinde kaldığı doğrulanmıştır.

---

## Gün 6 — 29-06-2026: PyTorch Eğitim Döngüsünün Kurulması ve GPU İvmelendirme Analizi (`src/train.py`)

**Problem ve Mühendislik Kısıtları:**  
PyTorch esnek bir kütüphane olup eğitim döngüsünün (forward pass, loss hesaplama, backward pass, optimizer adımı, gradyan sıfırlama) elle yazılmasını gerektirir. Gradyan sıfırlamanın unutulması gradyan birikmesine (accumulation) yol açarak eğitimi bozar. Ayrıca `model.train()` ve `model.eval()` modlarının doğru ayarlanması gerekir.

**Alternatif Analizi ve Seçim Gerekçesi:**  
PyTorch Lightning veya saf PyTorch tercihleri değerlendirilmiştir. Eğitim adımlarının (backpropagation, loss scaling, GPU memory flushing) düşük seviyede tam kontrolünü elinde tutmak ve staj kapsamında derin öğrenmenin altyapısını tam kavramak amacıyla saf PyTorch tercih edilmiştir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`src/train.py` yazılmıştır. Kayıp fonksiyonu olarak `CrossEntropyLoss`, optimizer olarak `Adam(lr=0.001)` seçilmiştir. 1 epoch'luk test koşusu yapılarak eğitim ve doğrulama adımları doğrulanmıştır.

[EKRAN GÖRÜNTÜSÜ: src/train.py — PyTorch ana eğitim döngüsü, forward/backward pass ve zero_grad adımları]

```python
# PyTorch eğitim döngüsü temel adımları
model.train()
for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

[GÖRSEL: gpu_utilization_nvidiasmi.png — Eğitim sırasında GTX 1050 Ti VRAM ve GPU kullanım ekranı]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
1 epoch sonunda model %85.42 eğitim doğruluğuna ve %94.12 doğrulama doğruluğuna ulaşmıştır. GPU kullanımının %95+ seviyesinde verimli çalıştığı görülmüştür.

---

## Gün 7 — 30-06-2026: Modüler Mimarinin Dinamikleştirilmesi ve Cihaz Yönetimi İyileştirmesi

**Problem ve Mühendislik Kısıtları:**  
Model tanımının sınıf sayısını sabit (hardcoded) alması, farklı veri setlerinde kodun tekrar yazılmasına neden olur. Ayrıca model `.to(device)` ile GPU'ya aktarılmadan önce optimizer tanımlanırsa, optimizer parametreleri CPU bellek adreslerine bağlanır ve eğitim anında `RuntimeError: Input and weight tensors are on different devices` hatası alınır.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`src/model.py` betiğindeki `get_crop_disease_model` fonksiyonu parametrik `num_classes` alacak şekilde güncellenmiştir. `train.py` içerisinde cihaz ataması optimizer tanımının önüne çekilmiştir. Proje modüllerinin mutlak yolla sorunsuz çalışması için `python -m src.train` çalıştırma standardı getirilmiştir.

[EKRAN GÖRÜNTÜSÜ: src/model.py — Dinamik num_classes parametresi alan model yapıcı fonksiyonu]

[EKRAN GÖRÜNTÜSÜ: src/train.py — Modüler importlar ve model.to(device) sonrası optimizer tanımı]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Hatalı cihaz bağlama sırası düzeltilmiş, kod modülerleşerek farklı sınıf sayılarına tam uyumlu hale getirilmiştir.

---

## Gün 8 — 01-07-2026: Komut Satırı Argümanları (Argparse) ve Üstveri Destekli Checkpoint Kayıt Altyapısı

**Problem ve Mühendislik Kısıtları:**  
Eğitim hiperparametrelerinin (`epochs`, `batch_size`, `lr`) kod içerisinden değiştirilmesi deneysel takibi zorlaştırır. Ayrıca kaydedilen model dosyalarında sadece ağırlıkların saklanması, model başka bir ortamda yüklenirken sınıf isimleri (`class_to_idx`) ve görsel boyutlandırma parametreleri bilinmediğinde çıkarım hatalarına yol açar.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`src/train.py` betiğine `argparse` modülü eklenmiştir. Model kaydedilirken tüm üstverileri (metadata payload) içeren `checkpoints/best_crop_model.pth` yapısı kurulmuştur.

[EKRAN GÖRÜNTÜSÜ: src/train.py — parse_args fonksiyonu ve torch.save checkpoint payload oluşturma bloğu]

```python
# Checkpoint üstveri kayıt yapısı
checkpoint_payload = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_acc": best_acc,
    "num_classes": num_classes,
    "class_to_idx": class_to_idx,
    "idx_to_class": idx_to_class,
    "transform_params": {"resize": (224, 224), "mean": MEAN_VALUE, "std": STD_VALUE},
    "architecture": "resnet18"
}
torch.save(checkpoint_payload, "checkpoints/best_crop_model.pth")
```

[GÖRSEL: checkpoint_payload_structure.png — Checkpoint içindeki model_state_dict, class_to_idx ve transform üstverileri]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Parametreler terminalden esnek şekilde yönetilebilir hale getirilmiş, üretime tam hazır model kayıt altyapısı kurulmuştur.

---

## Gün 9 — 02-07-2026: Otomatik Loglama ve Öğrenme Eğrilerinin Çizdirilmesi (`src/utils.py`)

**Problem ve Mühendislik Kısıtları:**  
Eğitim esnasında konsola basılan metriklerin kaybolması, modelin aşırı öğrenme veya sönümlenme eğilimlerinin sonradan analiz edilmesini imkansız kılar. Metriklerin disk üzerinde yapılandırılmış CSV olarak saklanması ve otomatik grafiğe dönüştürülmesi gerekmektedir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`src/utils.py` modülü yazılarak `save_training_log` ve `plot_learning_curves` fonksiyonları eklenmiştir. Matplotlib kullanılarak ikili grafik barındıran `results/learning_curves.png` otomatik üretilmiştir.

[EKRAN GÖRÜNTÜSÜ: src/utils.py — CSV log kaydetme ve Matplotlib öğrenme eğrileri çizim fonksiyonları]

[GÖRSEL: learning_curves.png — 15 epoch'luk eğitim ve doğrulama kayıp/doğruluk eğrileri]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Her epoch sonunda `results/training_log.csv` güncellenmiş ve grafikleri otomatik çizdirilerek grafiksel takip sağlanmıştır.

---

## Gün 10 — 03-07-2026: PlantVillage Test Seti Değerlendirmesi ve Karmaşıklık Matrisi (`src/evaluate.py`)

**Problem ve Mühendislik Kısıtları:**  
Modelin başarısı doğrulama kümesi dışında hiç görmediği bağımsız test kümesinde (2,064 imaj) sınıf bazlı Precision, Recall ve F1 metrikleriyle teyit edilmelidir. Dengeli ve dengesiz sınıflardaki karışıklıkların tespiti için 15x15 Karmaşıklık Matrisi çizdirilmelidir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`src/evaluate.py` betiği geliştirilmiştir. `checkpoints/best_crop_model.pth` yüklenerek test kümesi üzerinde çıkarım yapılmış, `results/plantvillage_metrics.json` ve `results/confusion_matrix.png` üretilmiştir.

[EKRAN GÖRÜNTÜSÜ: src/evaluate.py — Held-out test seti değerlendirme döngüsü ve scikit-learn metrik hesaplaması]

[GÖRSEL: confusion_matrix.png — 15x15 normalize edilmiş karmaşıklık matrisi görseli]

```json
// results/plantvillage_metrics.json özeti
{
  "overall_accuracy": 0.992736,
  "macro_avg": { "precision": 0.9921, "recall": 0.9926, "f1-score": 0.9923 },
  "weighted_avg": { "precision": 0.9928, "recall": 0.9927, "f1-score": 0.9927 }
}
```

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Model PlantVillage test kümesinde **%99.27 Genel Doğruluk** ve **0.99 F1-Skoru** elde etmiştir. Kontrollü stüdyo görsellerinde modelin mükemmel çalıştığı doğrulanmıştır.

---

## Gün 11 — 06-07-2026: PlantDoc Gerçek Saha Veri Setinin İçe Aktarılması ve Etiket Haritalama (`src/setup_plantdoc.py`)

**Problem ve Mühendislik Kısıtları:**  
Laboratuvarda %99.27 doğruluk veren modeller tarladaki gerçek fotoğraflarda başarısız olabilir. Gerçek dünya performansını ölçmek için açık kaynaklı **PlantDoc** saha veri seti projeye eklenmelidir. PlantDoc etiket yapıları ile PlantVillage sınıfları farklı isimlere sahip olduğundan otomatik haritalama gerekmektedir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`src/setup_plantdoc.py` betiği yazılmıştır. PlantDoc test klasöründeki etiketler projenin 15 hedef sınıfına haritalanmış, 102 adet geçerli yaprak görseli test için yapılandırılmıştır.

[EKRAN GÖRÜNTÜSÜ: src/setup_plantdoc.py — PlantDoc etiketlerini 15 hedef sınıfa eşleyen haritalama betiği]

[GÖRSEL: plantdoc_sample_images.png — PlantDoc karmaşık açık tarla yaprak fotoğrafları örnekleri]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Farklı veri kaynaklarının etiket uyumlaştırması (data harmonization) başarıyla tamamlanmıştır.

---

## Gün 12 — 07-07-2026: PlantDoc Sıfır-Vuruş (Zero-Shot) Genelletirme Değerlendirmesi ve Domain Shift Teşhisi (`src/evaluate_plantdoc.py`)

**Problem ve Mühendislik Kısıtları:**  
PlantVillage üzerinde %99.27 doğruluk veren modelin gerçek tarla koşullarında çekilmiş PlantDoc fotoğraflarında nasıl performans göstereceğinin (Zero-Shot Generalization) ölçülmesi gerekmektedir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`src/evaluate_plantdoc.py` betiği yazılarak laboratuvar modeli doğrudan PlantDoc test seti üzerinde çalıştırılmıştır.

[EKRAN GÖRÜNTÜSÜ: src/evaluate_plantdoc.py — PlantDoc üzerinde doğrudan çıkarım yapan zero-shot değerlendirme kodu]

[GÖRSEL: plantdoc_baseline_metrics.json — %15.69 sıfır-vuruş doğruluk sonucu JSON çıktısı]

```json
// results/plantdoc_baseline_metrics.json
{
  "zero_shot_accuracy": 0.156862,
  "evaluated_images": 102
}
```

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Model PlantDoc saha verisinde **%15.69 Sıfır-Vuruş Doğruluğuna** düşmüştür (102 imajdan yalnızca 16'sı doğru). Bu durum, modelin hastalık semptomları yerine laboratuvardaki düz beyaz kağıt arka planlarını ve stüdyo ışıklarını ezberlediğini (**Shortcut Learning / Domain Shift**) açıkça göstermiştir.

---

## Gün 13 — 08-07-2026: PlantDoc İnce Ayar (Fine-Tuning / Transfer Learning) Çalışması (`src/finetune_plantdoc.py`)

**Problem ve Mühendislik Kısıtları:**  
PlantDoc veri seti çok az sayıda imaja (102 test) sahip olduğundan modelin tüm katmanlarını eğitmek aşırı öğrenmeye sebep olur. Alt evrişimsel katmanların dondurularak yalnızca üst katmanların eğitilmesi gerekmektedir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`src/finetune_plantdoc.py` betiği yazılmıştır. ResNet-18'in `layer1` ve `layer2` katmanları dondurulmuş (`requires_grad=False`); `layer3`, `layer4` ve `fc` katmanları düşük öğrenme oranıyla ($lr=10^{-4}$) 5 epoch eğitilmiştir.

[EKRAN GÖRÜNTÜSÜ: src/finetune_plantdoc.py — Katman dondurma (requires_grad=False) ve ince ayar döngüsü]

[GÖRSEL: plantdoc_before_after.json — İnce ayar öncesi (%15.69) ve sonrası (%22.55) karşılaştırma JSON çıktısı]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
İnce ayar sonrası PlantDoc doğruluğu **%22.55** seviyesine yükselmiştir (**+%6.86 net artış**). Katman dondurmanın az verili saha adaptasyonundaki yararı teyit edilmiştir.

---

## Gün 14 — 09-07-2026: TorchScript ve ONNX Model Dışa Aktarım Boru Hattı (`src/utils.py`)

**Problem ve Mühendislik Kısıtları:**  
PyTorch modelleri (`.pth`) Python yorumlayıcısına ve PyTorch kütüphanesine bağımlıdır. Üretim ortamlarında C++ tabanlı yüksek hızlı web servislerinde çalışabilmek için modelin bağımsız TorchScript (`.pt`) ve ONNX (`.onnx`) formatlarına dönüştürülmesi şarttır.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`src/utils.py` modülüne `export_model_formats` fonksiyonu eklenmiştir. Dinamik batch boyutuna sahip ONNX modeli üretilmiştir.

[EKRAN GÖRÜNTÜSÜ: src/utils.py — TorchScript (.pt) ve ONNX (.onnx) dışa aktarım fonksiyonları]

[GÖRSEL: onnx_model_netron.png — Netron aracıyla görselleştirilmiş ONNX hesaplama grafiği]

```python
# ONNX dinamik batch dışa aktarım betiği
torch.onnx.export(
    model, dummy_input, "checkpoints/crop_disease_model.onnx",
    export_params=True, opset_version=12,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Model PyTorch bağımlılığından kurtarılarak ONNX Runtime ile mikro-saniyeler seviyesinde çıkarım yapabilir formata getirilmiştir.

---

## Gün 15 — 10-07-2026: Bağımsız Çıkarım API Betiğinin Geliştirilmesi (`src/predict.py`)

**Problem ve Mühendislik Kısıtları:**  
Web uygulamasından gelecek tekil görsel tahmin isteklerini işleyecek, görsel ön işleme, Softmax olasılık dönüşümü, Top-K sıralaması ve gecikme süresi hesaplayan bağımsız bir çıkarım modülüne ihtiyaç vardır.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`src/predict.py` modülü geliştirilmiş ve `predict_image` fonksiyonu yazılmıştır. Örnek domates ve patates yaprak görselleriyle testler koşturulmuştur.

[EKRAN GÖRÜNTÜSÜ: src/predict.py — Single image predict_image çıkarım fonksiyonu ve CLI çalıştırma bloğu]

[GÖRSEL: predict_cli_output.png — Terminal üzerinden örnek yaprak fotoğrafı çıkarım JSON çıktısı]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Çıkarım betiğinin tek bir görseli ~15 ms içerisinde işleyip JSON formatında tahmin ürettiği teyit edilmiştir.

---

## Gün 16 — 13-07-2026: Aşırı Öğrenme Teşhisi, Erken Durdurma (Val Loss), Düzenlileştirme ve Saha Simülasyonu Veri Çoğullaması

**Problem ve Mühendislik Kısıtları:**  
Eğitim logları detaylı incelendiğinde modelin 12. epoch'ta eğitim kaybını sıfırladığı (%99.8 train acc), ancak doğrulama kaybının (val_loss) 15. epoch'ta `0.03` seviyesinden `0.27` seviyesine fırlayarak aşırı öğrendiği tespit edilmiştir. Eski kodun en iyi epoch yerine son epoch (Epoch 15) ağırlıklarını kaydettiği saptanmıştır. Ayrıca model stüdyo arka planlarını ezberlediği için PlantDoc sıfır-vuruş başarımı %15.69'da kalmıştır.

**Uygulanan Yöntem ve Teknik Detaylar:**  
Bu problemleri çözmek üzere 4 temel müdahale yapılmıştır:
1. **Model Düzenlileştirme:** `src/model.py` içerisine **Dropout (p=0.3)** eklenmiştir.
2. **Optimizer Düzenlileştirme ve LR Scheduler:** `src/train.py` içerisine **L2 Weight Decay ($10^{-4}$)** ve `ReduceLROnPlateau(patience=2, factor=0.5)` eklenmiştir.
3. **Val-Loss Erken Durdurma (Early Stopping):** `val_loss` takibi ile 3 epoch iyileşmeyen durum için erken durdurma eklenmiştir (`--patience 3`). Checkpoint en düşük val_loss veren epoch'ta kaydedilmiştir.
4. **Saha Simülasyonu Veri Çoğullaması (`--field-aug`):** `src/dataset.py` içerisine renk kırpma, açılandırma ve kesme-karartma (`ColorJitter`, `RandomResizedCrop`, `RandomRotation`, `RandomErasing cutout`) eklenmiştir.

[EKRAN GÖRÜNTÜSÜ: src/dataset.py — field_sim_transform saha simülasyonu çoğullama boru hattı tanımı]

[EKRAN GÖRÜNTÜSÜ: src/model.py — Dropout(p=0.3) eklenmiş FC sınıflandırma başlığı]

[EKRAN GÖRÜNTÜSÜ: src/train.py — val_loss takip eden erken durdurma ve ReduceLROnPlateau döngüsü]

[GÖRSEL: early_stopping_terminal_output.png — Epoch 6 en iyi checkpoint kaydı ve Epoch 9 erken durdurma terminal ekranı]

[GÖRSEL: plantdoc_augmented_zeroshot_metrics.json — Saha çoğullamalı modelin %26.47 sıfır-vuruş doğruluk sonucu]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Eğitim 9. epoch'ta erken durdurma ile sonlanmış, en iyi model olarak **Epoch 6** (`val_loss=0.1233`, `val_acc=0.9598`) saklanmıştır. Saha çoğullamalı bu yeni model ile PlantDoc sıfır-vuruş (zero-shot) başarımı **%15.69'dan %26.47'ye yükselmiştir (+%10.78 net artış!)**. (Not: İnce ayar adımı bu çoğullamalı model üzerine henüz uygulanmamış olup, saha çoğullaması yapılmış bu temel model üzerine yeniden fine-tuning yapılması gelecek çalışma adımı olarak tanımlanmıştır).

---

## Gün 17 — 14-07-2026: Metrik Doğrulamaları ve Dokümantasyon Güncellemeleri

**Amaç ve Yapılan Çalışmalar:**  
Güncellenen eğitim sonuçları, erken durdurma verileri ve PlantDoc %26.47 sıfır-vuruş başarımı metrik dosyalarında ve `README.md` dokümantasyonunda güncellenmiştir.

[EKRAN GÖRÜNTÜSÜ: README.md — Güncellenmiş model başarımı ve deneysel sonuçlar bölümü]

[GÖRSEL: metrics_json_comparison.png — Eski ve yeni metrik dosyalarının yan yana karşılaştırması]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Dokümantasyon güncellenerek proje sonuçları bilimsel raporlamaya hazır hale getirilmiştir.

---

## Gün 18 — 15-07-2026: Proje Temizliği ve Kod Standartları Denetimi

**Amaç ve Yapılan Çalışmalar:**  
Proje dizinindeki geçici derleme dosyaları (`__pycache__`) ve derleme logları temizlenmiştir. Kodlar PEP8 standartlarına uygunluk açısından denetlenmiştir.

[EKRAN GÖRÜNTÜSÜ: terminal — git status ve dizin hijyen doğrulama çıktısı]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Sürdürülebilir yazılım geliştirmede dizin hijyeninin önemi pekiştirilmiştir.

---

## Gün 19 — 17-07-2026: Deneysel Notebook'ların Arşivlenmesi ve `src/` Modül Temizliği

**Amaç ve Yapılan Çalışmalar:**  
`01_data_exploration.ipynb` ve `02_pytorch_training_tutorial.ipynb` notebook'ları `notebooks/experiments/` dizinine taşınmış; `src/` klasörü sadece modüler üretim kodlarını barındıracak şekilde sadeleştirilmiştir.

[EKRAN GÖRÜNTÜSÜ: directory_structure — notebooks/experiments/ arşivlenmiş klasör yapısı]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Üretim kodu ile araştırma kodlarının ayrıştırılması sağlanmıştır.

---

## Gün 20 — 20-07-2026: Staj-I Çıkış Koşullarının Kontrolü ve Kapanış

**Amaç ve Yapılan Çalışmalar:**  
`PLAN.md` dosyasındaki tüm çıkış koşulları (Exit Conditions) denetlenmiş, ResNet-18 eğitimi, metrik hesaplamaları, alan kayması analizi ve ONNX model dışa aktarımlarının tamamlandığı doğrulanarak Staj-I kapatılmıştır.

[EKRAN GÖRÜNTÜSÜ: PLAN.md — Phase 0-7 tamamlanmış çıkış koşulları kontrol listesi]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Staj-I çalışmaları eksiksiz olarak başarıyla tamamlanmıştır.

---

# 3. SONUÇ VE DEĞERLENDİRME

Staj-I çalışmasında PyTorch ve ResNet-18 mimarisi kullanılarak bitki hastalık sınıflandırma modeli geliştirilmiştir. Elde edilen temel teknik bulgular:
1. **Domain Shift Teşhisi:** Stüdyo görsellerinde %99.27 doğruluk veren modelin saha verilerinde %15.69'a gerilediği görülmüştür.
2. **Saha Çoğullaması ve Düzenlileştirme:** Erken Durdurma (`val_loss`), L2 Weight Decay ($10^{-4}$), Dropout ($p=0.3$) ve Saha Simülasyonu Veri Çoğullaması (`--field-aug`) ile sıfır-vuruş saha başarımı **%15.69'dan %26.47'ye (+%10.78 artış)** yükseltilmiştir.
3. **ONNX Dışa Aktarımı:** Model ONNX formatında dışa aktarılarak Staj-II web entegrasyonuna hazır edilmiştir.
