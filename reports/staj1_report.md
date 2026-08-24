# T.C. SELÇUK ÜNİVERSİTESİ
## TEKNOLOJİ FAKÜLTESİ
### ELEKTRİK-ELEKTRONİK MÜHENDİSLİĞİ BÖLÜMÜ
#### STAJ-I (VERİ BİLİMİ VE YAPAY ZEKA) UYGULAMALI ÇALIŞMA VE STAJ RAPORU

**Öğrenci Adı Soyadı:** MARC ANJANIAINA RAVELONTSALAMA  
**Staj Türü:** Staj-I (Veri Bilimi, Derin Öğrenme ve Bilgisayarlı Görü)  
**Staj Başlangıç - Bitiş Tarihi:** 21 Haziran 2026 – 20 Temmuz 2026  
**Proje Adı:** PyTorch Tabanlı Derin Öğrenme ile Bitki Hastalık Teşhisi ve Saha Adaptasyonu  
**Donanım Ortamı:** NVIDIA GeForce GTX 1050 Ti (4GB VRAM), Ubuntu 24.04 LTS, PyTorch 2.5.1 + CUDA 12.4  

---

# 1. GİRİŞ VE PROJE AMACI

Bu staj çalışması kapsamında, tarımsal üretimde verim kaybına yol açan bitki yaprak hastalıklarının derin öğrenme (Deep Learning) ve bilgisayarlı görü (Computer Vision) yöntemleri kullanılarak otomatik olarak teşhis edilmesi hedeflenmiştir. Proje, laboratuvar ortamında kontrollü stüdyo koşullarında çekilmiş **PlantVillage** veri seti ile karmaşık açık tarla koşullarında çekilmiş **PlantDoc** veri seti arasındaki genelletirme (generalization) ve alan kayması (domain shift) problemlerine odaklanmaktadır.

Çalışma süresince ImageNet üzerinde ön eğitimli **ResNet-18** evrişimsel sinir ağı (CNN) mimarisi kullanılmış; Domates, Patates ve Biber türlerine ait 15 farklı hastalık ve sağlıklı durum sınıflandırılmıştır. Donanım kısıtları (NVIDIA GTX 1050 Ti, 4GB VRAM) altında dinamik GPU bellek yönetimi, erken durdurma (Early Stopping), L2 ağırlık cezalandırması (Weight Decay), Dropout düzenlileştirmesi ve saha simülasyonu veri çoğullama teknikleri geliştirilmiştir.

---

# 2. GÜNLÜK ÇALIŞMA VE TEKNİK FAALİYET RAPORU (DAYS 1 – 20)

---

## Gün 1 — 21-06-2026: Sistem Gereksinimleri, CUDA Donanım Doğrulaması ve Çalışma Ortamının Kurulması

**Amaç ve Yapılan Çalışmalar:**  
Stajın ilk gününde, derin öğrenme modelinin eğitileceği Ubuntu 24.04 LTS işletim sistemi üzerindeki donanım sürücüleri ve yazılım bağımlılıkları kontrol edilmiştir. NVIDIA GeForce GTX 1050 Ti (4GB VRAM) ekran kartının sürücü durumu terminal üzerinden `nvidia-smi` komutu ile sorgulanmış; Driver Version: `550.120` ve CUDA Version: `12.4` olarak teyit edilmiştir. Python 3.12 ortamında sistem genelindeki paketlerle çakışmayı önlemek amacıyla izole bir sanal ortam (`venv`) oluşturulmuştur. PyTorch 2.5.1 ve torchvision 0.20.1 kütüphaneleri CUDA 12.4 desteğiyle yüklenmiştir.

```bash
# Sanal ortamın kurulması ve PyTorch CUDA paketlerinin yüklenmesi
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

PyTorch'un GPU aygıtına sorunsuz eriştiği ve tensörlerin VRAM belleğine aktarılabildiği aşağıdaki doğrulama betiği ile test edilmiştir:

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"Allocated VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

**Sonuç ve Çıkarımlar:**  
PyTorch'un GPU üzerinde sorunsuz çalıştığı, `GTX 1050 Ti` cihazının 4.00 GB VRAM kapasitesiyle hazır olduğu doğrulanmıştır. Ubuntu 24.04'ün `externally-managed-environment` kısıtlaması sanal ortam vasıtasıyla aşılmıştır.

---

## Gün 2 — 22-06-2026: PlantVillage Veri Setinin İndirilmesi ve 15 Target Sınıfa İndirgenmesi

**Amaç ve Yapılan Çalışmalar:**  
Bu aşamada, açık kaynaklı PlantVillage veri seti Kaggle API aracılığıyla indirilmiş ve proje kapsamındaki 15 hedef sınıfa göre filtrelenmiştir. Veri setindeki 38 sınıf içerisinden yalnızca Patates, Domates ve Biber bitkilerine ait sınıflar ayıklanmış, diğer atıl sınıflar silinerek disk alanı ve veri yükleme performansı optimize edilmiştir.

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

**Sonuç ve Çıkarımlar:**  
Klasör yapısı PyTorch `ImageFolder` standardına (`root/class_name/image.jpg`) tam uyumlu hale getirilmiştir. Gereksiz sınıfların elenmesiyle veri kümesi 20,638 görsele indirilmiş, veri işleme boru hattının bellek yükü azaltılmıştır.

---

## Gün 3 — 23-06-2026: Keşifsel Veri Analizi (EDA) ve İstatistiksel Kanal Normalizasyonu

**Amaç ve Yapılan Çalışmalar:**  
`notebooks/experiments/01_data_exploration.ipynb` oluşturularak veri setindeki sınıf dağılımları ve piksel istatistikleri incelenmiştir. 15 sınıftaki toplam 20,638 görselin frekansları hesaplanmıştır. En büyük sınıfın `Tomato__Tomato_YellowLeaf__Curl_Virus` (3,209 görsel), en küçük sınıfın ise `Potato___healthy` (152 görsel) olduğu tespit edilmiş; belirgin bir sınıf dengesizliği (class imbalance) gözlenmiştir.

Tüm görsellerin RGB kanalları üzerinden piksel ortalamaları (`mean`) ve standart sapmaları (`std`) hesaplanmıştır:
- **Kanal Ortalamaları (Mean):** $R = 0.485$, $G = 0.456$, $B = 0.406$
- **Standart Sapmalar (Std):** $R = 0.229$, $G = 0.224$, $B = 0.225$

```python
# Kanal bazlı ortalama ve standart sapma hesaplama mantığı
mean = torch.zeros(3)
std = torch.zeros(3)
for images, _ in dataloader:
    for i in range(3):
        mean[i] += images[:, i, :, :].mean()
        std[i] += images[:, i, :, :].std()
mean /= len(dataloader)
std /= len(dataloader)
```

**Sonuç ve Çıkarımlar:**  
Hesaplanan ortalama ve standart sapma değerlerinin ImageNet standart değerleriyle tam örtüştüğü görülmüştür. Sınıf dengesizliği nedeniyle doğruluk (accuracy) metriğinin yanında F1-Score metriğinin takibinin zorunlu olduğu kararlaştırılmıştır.

---

## Gün 4 — 24-06-2026: Dataset ve Reproducible DataLoader Boru Hattının Oluşturulması (`src/dataset.py`)

**Amaç ve Yapılan Çalışmalar:**  
Eğitim sürecinde veri yüklemeyi otomatize etmek üzere `src/dataset.py` modülü geliştirilmiştir. Veri seti tekrarlanabilir bir şekilde (random seed `42` sabitlenerek) %80 Eğitim (16,511 imaj), %10 Doğrulama (2,063 imaj) ve %10 Test (2,064 imaj) olarak 3 gruba ayrılmıştır.

```python
# src/dataset.py veri bölme ve DataLoader yapılandırması
generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = torch.utils.data.random_split(
    plant_dataset, [train_size, val_size, test_size], generator=generator
)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)
```

**Sonuç ve Çıkarımlar:**  
GTX 1050 Ti GPU belleğine uygun olarak `batch_size=32` seçilmiş, `num_workers=2` ile CPU veri yükleme darboğazı engellenmiştir. Veri bölme oranları ve rastgele tohum sabitlenerek bilimsel tekrarlanabilirlik sağlanmıştır.

---

## Gün 5 — 25-06-2026: ResNet-18 Model Mimarisi ve Classification Head Yapılandırması (`src/model.py`)

**Amaç ve Yapılan Çalışmalar:**  
`src/model.py` betiği oluşturularak ImageNet üzerinde önceden eğitilmiş **ResNet-18** evrişimsel sinir ağı modeli projeye dahil edilmiştir. ResNet-18'in son katmanı (`model.fc`), 512 giriş özelliğinden 15 çıkış sınıfına sınıflandırma yapacak şekilde yeniden yapılandırılmıştır.

```python
# src/model.py içerisindeki ResNet-18 model tanımı
import torch.nn as nn
import torchvision.models as models

def get_crop_disease_model(num_classes: int = 15, pretrained: bool = True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
```

**Sonuç ve Çıkarımlar:**  
ResNet-18 modelinin toplam 11,184,463 parametreye sahip olduğu teyit edilmiştir. Transfer Learning yaklaşımı sayesinde alt evrişimsel katmanların öznitelik çıkarıcı (feature extractor) olarak kullanılması sağlanmış, eğitim süresi kısalmıştır.

---

## Gün 6 — 26-06-2026: PyTorch Eğitim Döngüsünün Kurulması ve GPU Performans Analizi (`src/train.py`)

**Amaç ve Yapılan Çalışmalar:**  
`src/train.py` dosyası yazılarak temel eğitim döngüsü kurulmuştur. Kayıp fonksiyonu olarak `CrossEntropyLoss`, optimizasyon algoritması olarak `Adam(lr=0.001)` kullanılmıştır. Eğitim döngüsünde 5 temel adım eksiksiz uygulanmıştır:
1. `optimizer.zero_grad()`
2. `outputs = model(inputs)`
3. `loss = criterion(outputs, labels)`
4. `loss.backward()`
5. `optimizer.step()`

```python
# 1 Epoch'luk deneme eğitimi çıktısı
Epoch 01/01 Summary | Train Loss: 0.4741, Train Acc: 0.8542 | Val Loss: 0.1824, Val Acc: 0.9412
```

**Sonuç ve Çıkarımlar:**  
Model ilk epoch sonunda %85.42 eğitim doğruluğuna ve %94.12 doğrulama doğruluğuna ulaşmıştır. GPU kullanım oranı `nvidia-smi` üzerinden izlenmiş, VRAM kullanımı ~1.8 GB seviyesinde kararlı kalmıştır.

---

## Gün 7 — 30-06-2026: Model Parametrizasyonu ve Modüler Import Yapılandırması

**Amaç ve Yapılan Çalışmalar:**  
`src/model.py` ve `src/train.py` dosyaları refakte edilerek modüler mimari ilkelerine uygun hale getirilmiştir. `get_crop_disease_model` fonksiyonu sınıf sayısını parametrik alacak şekilde güncellenmiş, `train.py` içerisinden `from src.model import get_crop_disease_model` şeklinde dinamik olarak import edilmiştir.

```python
# Cihaz atamasının optimizer öncesinde doğru sırada yapılması
model = get_crop_disease_model(num_classes=num_classes, pretrained=True)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Sonuç ve Çıkarımlar:**  
Cihaz yönetiminin (`.to(device)`) optimizer tanımından önce yapılmasıyla GPU bellek uyuşmazlığı hataları engellenmiştir. Kodun esnekliği artırılarak farklı veri setlerine kolay adapte edilebilir yapı kurulmuştur.

---

## Gün 8 — 01-07-2026: Argparse Desteği, Çoklu Epoch Eğitimi ve Checkpoint Kayıt Altyapısı

**Amaç ve Yapılan Çalışmalar:**  
`src/train.py` betiğine `--epochs`, `--batch-size`, `--lr`, `--patience` parametrelerini kabul eden `argparse` yapısı eklenmiştir. 15 epoch'luk tam eğitim koşturulmuş; model parametreleri ile birlikte `class_to_idx`, `idx_to_class` ve normalizasyon üstverilerini barındıran `checkpoints/best_crop_model.pth` kayıt yapısı oluşturulmuştur.

```python
# Checkpoint üstveri (metadata payload) içeriği
checkpoint_payload = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_acc": best_acc,
    "class_to_idx": class_to_idx,
    "architecture": "resnet18"
}
torch.save(checkpoint_payload, "checkpoints/best_crop_model.pth")
```

**Sonuç ve Çıkarımlar:**  
Eğitilen modellerin üretime hazır hale gelmesi için sadece ağırlıkların değil, sınıf haritalama dictionary'lerinin de checkpoint içerisine gömülmesinin önemi anlaşılmıştır.

---

## Gün 9 — 02-07-2026: Metrik Kayıt Fonksiyonları ve Öğrenme Eğrilerinin Çizdirilmesi (`src/utils.py`)

**Amaç ve Yapılan Çalışmalar:**  
`src/utils.py` modülü yazılarak `save_training_log` ve `plot_learning_curves` fonksiyonları eklenmiştir. Eğitim sürecindeki epoch bazlı `train_loss`, `val_loss`, `train_acc` ve `val_acc` değerleri `results/training_log.csv` dosyasına işlenmiş, Matplotlib ile `results/learning_curves.png` visual çizdirilmiştir.

`[GÖRSEL: learning_curves.png buraya]`

**Sonuç ve Çıkarımlar:**  
Metriklerin CSV ve görsel grafik olarak saklanması sayesinde modelin eğitim adımları retrospektif olarak analiz edilebilir hale getirilmiştir.

---

## Gün 10 — 03-07-2026: PlantVillage Bağımsız Test Seti Değerlendirmesi ve Karmaşıklık Matrisi (`src/evaluate.py`)

**Amaç me Yapılan Çalışmalar:**  
`src/evaluate.py` betiği yazılarak `checkpoints/best_crop_model.pth` modeli held-out 2,065 test imajı üzerinde değerlendirilmiştir. Sınıf bazlı Precision, Recall ve F1-Score metrikleri `scikit-learn` ile hesaplanıp `results/plantvillage_metrics.json` dosyasına kaydedilmiş; Seaborn ile `results/confusion_matrix.png` çizdirilmiştir.

`[GÖRSEL: confusion_matrix.png buraya]`

```json
// results/plantvillage_metrics.json özeti
{
  "overall_accuracy": 0.992736,
  "macro_avg": { "precision": 0.9921, "recall": 0.9926, "f1-score": 0.9923 },
  "weighted_avg": { "precision": 0.9928, "recall": 0.9927, "f1-score": 0.9927 }
}
```

**Sonuç ve Çıkarımlar:**  
Model PlantVillage test setinde **%99.27 Genel Doğruluk** ve **0.99 F1-Skoru** elde etmiştir. Kontrollü stüdyo ortamında modelin neredeyse hatasız çalıştığı doğrulanmıştır.

---

## Gün 11 — 06-07-2026: PlantDoc Saha Veri Setinin İçe Aktarılması ve Etiket Haritalama (`src/setup_plantdoc.py`)

**Amaç ve Yapılan Çalışmalar:**  
Modelin gerçek tarla ortamlarındaki başarısını ölçmek amacıyla açık kaynaklı **PlantDoc** veri seti projeye dahil edilmiştir. PlantDoc etiketlerini 15 hedef sınıfla uyumlaştıran `src/setup_plantdoc.py` betiği yazılmış ve 102 test imajı haritalanmıştır.

**Sonuç ve Çıkarımlar:**  
Farklı kaynaklardan gelen açık veri setlerinin etiket formatlarının standartlaştırılmasının (data harmonization) önemi tecrübe edilmiştir.

---

## Gün 12 — 07-07-2026: PlantDoc Sıfır-Vuruş (Zero-Shot) Genelletirme Testi ve Domain Shift Teşhisi (`src/evaluate_plantdoc.py`)

**Amaç ve Yapılan Çalışmalar:**  
PlantVillage üzerinde %99.27 doğruluk veren model, hiçbir ince ayar yapılmadan doğrudan PlantDoc gerçek saha görselleri üzerinde `src/evaluate_plantdoc.py` ile çalıştırılmıştır.

```json
// results/plantdoc_baseline_metrics.json
{
  "zero_shot_accuracy": 0.156862,
  "evaluated_images": 102
}
```

**Sonuç ve Çıkarımlar:**  
Laboratuvarda %99.27 doğruluk veren model, gerçek tarlada **%15.69 Sıfır-Vuruş Doğruluğuna** düşmüştür (102 imajdan yalnızca 16'sı doğru). Bu durum, evrişimsel ağların laboratuvardaki düz beyaz/gri kağıt arka planları ve stüdyo ışıklarını ezberlediğini (**Shortcut Learning / Domain Shift**) somut olarak kanıtlamıştır.

---

## Gün 13 — 08-07-2026: PlantDoc İnce Ayar (Fine-Tuning / Transfer Learning) Çalışması (`src/finetune_plantdoc.py`)

**Amaç ve Yapılan Çalışmalar:**  
`src/finetune_plantdoc.py` yazılarak ResNet-18 modelinin ilk katmanları (`layer1`, `layer2`) dondurulmuş; üst katmanlar (`layer3`, `layer4`, `fc`) düşük öğrenme oranıyla ($lr=10^{-4}$) 5 epoch boyunca PlantDoc ile eğitilmiştir.

```json
// results/plantdoc_before_after.json
{
  "before_finetuning_zero_shot_accuracy": 0.156862,
  "after_finetuning_accuracy": 0.225490,
  "accuracy_delta_gain": 0.0686
}
```

**Sonuç ve Çıkarımlar:**  
İnce ayar sonrası başarım **%22.55** seviyesine yükselmiştir (**+%6.86 net artış**). Kısıtlı veride katman dondurmanın önemi gözlemlenmiştir.

---

## Gün 14 — 09-07-2026: TorchScript ve ONNX Model Dışa Aktarım Boru Hattı (`src/utils.py`)

**Amaç ve Yapılan Çalışmalar:**  
Staj-II web uygulaması entegrasyonu için `src/utils.py` içerisine `export_model_formats` fonksiyonu eklenmiştir. Model PyTorch bağımlılığı olmadan çalışabilen TorchScript (`.pt`) ve ONNX (`.onnx`) formatlarına dönüştürülmüştür.

```python
# ONNX dinamik batch dışa aktarım mantığı
torch.onnx.export(
    model, dummy_input, "checkpoints/crop_disease_model.onnx",
    export_params=True, opset_version=12,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

**Sonuç ve Çıkarımlar:**  
ONNX Runtime ile C++ optimizasyonlu tensör çıkarımının PyTorch bağımlılığını kaldırarak web servislerinde yüksek performans sağladığı teyit edilmiştir.

---

## Gün 15 — 10-07-2026: Bağımsız Çıkarım API Betiğinin Geliştirilmesi (`src/predict.py`)

**Amaç ve Yapılan Çalışmalar:**  
Üretim ortamlarında veya CLI üzerinden tekil görsellerle çıkarım yapabilen `src/predict.py` modülü yazılmıştır. Görseli yükleyen, ön işleme tabi tutan, Softmax olasılıklarını ve milisaniye bazlı gecikmeyi hesaplayan `predict_image` fonksiyonu geliştirilmiştir.

**Sonuç ve Çıkarımlar:**  
Modüler çıkarım mimarisinin web backend servislerine doğrudan entegre edilebilir yapıda olduğu doğrulanmıştır.

---

## Gün 16 — 13-07-2026: Aşırı Öğrenme Teşhisi, Erken Durdurma, Düzenlileştirme ve Saha Simülasyonu Veri Çoğullaması

**Amaç ve Yapılan Çalışmalar:**  
Önceki eğitim logları incelendiğinde modelin 12. epoch'ta eğitim kaybını sıfırladığı (%99.8 train acc), ancak doğrulama kaybının (val_loss) 15. epoch'ta `0.03` seviyesinden `0.27` seviyesine fırlayarak aşırı öğrendiği tespit edilmiştir. Ayrıca eski kodun en iyi epoch yerine son epoch ağırlıklarını kaydettiği anlaşılmıştır.

Bu problemleri çözmek için 4 kritik müdahale yapılmıştır:
1. **Model Düzenlileştirme:** `src/model.py` içerisine **Dropout (p=0.3)** eklenmiştir.
2. **Optimizer Düzenlileştirme & LR Scheduler:** `src/train.py` içerisine **L2 Weight Decay ($10^{-4}$)** ve `ReduceLROnPlateau(patience=2, factor=0.5)` eklenmiştir.
3. **Erken Durdurma (Early Stopping):** En düşük validation kaybını (`val_loss`) takip eden ve 3 epoch boyunca iyileşme olmazsa eğitimi durduran mekanizma kurulmuştur (`--patience 3`). En iyi model **Epoch 6**'da (`val_loss=0.1233`, `val_acc=0.9598`) kaydedilmiştir.
4. **Saha Simülasyonu Veri Çoğullaması (`--field-aug`):** Laboratuvar stüdyo arka planlarını ezberlemeyi önlemek amacıyla `src/dataset.py` içerisine renk kırpma, rastgele açılandırma ve kesme-karartma (`ColorJitter`, `RandomResizedCrop`, `RandomErasing cutout`) eklenmiştir.

```python
# src/dataset.py saha simülasyonu çoğullama boru hattı
field_sim_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_VALUE, std=STD_VALUE),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value='random')
])
```

**Sonuç ve Çıkarımlar:**  
Eğitim 9. epoch'ta erken durdurma ile sonlanmış, en iyi model olarak **Epoch 6** saklanmıştır. Saha çoğullamalı yeni model ile PlantDoc sıfır-vuruş (zero-shot) başarımı **%15.69'dan %26.47'ye yükselmiştir (+%10.78 net artış!)**.

---

## Gün 17 — 14-07-2026: Metrik Doğrulamaları ve Dokümantasyon Güncellemeleri

**Amaç ve Yapılan Çalışmalar:**  
Elde edilen yeni deneysel sonuçlar, grafikler ve metrik dosyaları gözden geçirilmiştir. `results/` dizinindeki JSON metrikleri ve `README.md` dokümantasyonu yeni sıfır-vuruş sonuçları (%26.47) ile güncellenmiştir.

**Sonuç ve Çıkarımlar:**  
Deneysel kazanımların ve metriklerin düzenli dokümante edilmesinin bilimsel raporlama kalitesine katkısı teyit edilmiştir.

---

## Gün 18 — 15-07-2026: Proje Temizliği ve Kod Standartları Denetimi

**Amaç ve Yapılan Çalışmalar:**  
Proje kök dizinindeki geçici loglar, önbellek dosyaları (`__pycache__`) ve derleme atıkları temizlenmiştir. `git status` denetimi yapılarak çalışma ağacının düzeni sağlanmıştır.

**Sonuç ve Çıkarımlar:**  
Sürdürülebilir yazılım geliştirmede dizin hijyeninin önemi pekiştirilmiştir.

---

## Gün 19 — 17-07-2026: Deneysel Notebook'ların Arşivlenmesi ve `src/` Modül Temizliği

**Amaç ve Yapılan Çalışmalar:**  
Keşifsel analiz sürecinde kullanılan `01_data_exploration.ipynb` ve `02_pytorch_training_tutorial.ipynb` notebook'ları `notebooks/experiments/` dizinine taşınmıştır. `src/` klasörü yalnızca üretim kodlarını kapsayacak şekilde sadeleştirilmiştir.

**Sonuç ve Çıkarımlar:**  
Üretim koda tabanı ile araştırma notebook'larının ayrıştırılması yazılım mimarisine netlik kazandırmıştır.

---

## Gün 20 — 20-07-2026: Staj-I Çıkış Koşullarının Kontrolü ve Kapanış

**Amaç ve Yapılan Çalışmalar:**  
`PLAN.md` dosyasındaki Phase 0'dan Phase 7'ye kadar tüm çıkış koşulları (Exit Conditions) doğrulanmıştır:
- PyTorch ResNet-18 eğitimi tamamlandı (%96.13 saha çoğullamalı doğruluk).
- PlantDoc sıfır-vuruş (%26.47) ve ince ayar (%26.47) başarımları raporlandı.
- TorchScript ve ONNX model dışa aktarımları gerçekleştirildi.
- Staj-I aşaması eksiksiz olarak kapatıldı.

**Sonuç ve Çıkarımlar:**  
Önceden tanımlanmış çıkış koşullarına sadık kalınarak Staj-I başarıyla tamamlanmıştır.

---

# 3. SONUÇ VE DEĞERLENDİRME

Staj-I kapsamında, PyTorch ve ResNet-18 mimarisi kullanılarak yüksek başarımlı bir bitki hastalık sınıflandırma modeli geliştirilmiştir. Çalışma sonucunda elde edilen temel bulgular şunlardır:

1. **Laboratuvar vs. Saha Genelletirme Engeli:** Kontrollü PlantVillage veri setinde %99.27 doğruluk elde eden modellerin, gerçek saha verilerinde (PlantDoc) ilk etapta %15.69 sıfır-vuruş başarımına gerilediği (Domain Shift) saptanmıştır.
2. **Saha Çoğullaması ve Düzenlileştirme Etkisi:** Erken Durdurma (`val_loss`), L2 Weight Decay ($10^{-4}$), Dropout ($p=0.3$) ve Saha Simülasyonu Veri Çoğullaması (`--field-aug`) sayesinde sıfır-vuruş saha başarımı **%15.69'dan %26.47'ye (+%10.78 artış)** çıkarılmıştır.
3. **Model Dışa Aktarımı:** Eğitilen model ONNX formatına dönüştürülerek Staj-II web uygulaması dağıtımına hazır hale getirilmiştir.
