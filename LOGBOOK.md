# Staj-I Günlük Çalışma Günlüğü (LOGBOOK.md)

## Gün 1 - 21-06

**Görev:**
Sistem gereksinimlerinin belirlenmesi, NVIDIA sürücüleri ve CUDA/cuDNN kurulumlarının kontrolü, Python sanal ortamının (`venv`) oluşturulması ve temel bağımlılıkların (`torch`, `torchvision`, `torchaudio`) yüklenmesi.

**Yapılan:**
- Ubuntu 24.04 LTS işletim sistemi üzerinde NVIDIA GTX 1050 Ti (4GB VRAM) ekran kartının sürücü durumu `nvidia-smi` komutu ile kontrol edildi. Driver Version: 550.120, CUDA Version: 12.4 olarak teyit edildi.
- Proje dizininde isolated bir çalışma ortamı sağlamak amacıyla Python 3.12 ile sanal ortam (`venv`) oluşturuldu ve aktifleştirildi.
- PyTorch 2.5.1 ve torchvision 0.20.1 kütüphaneleri CUDA 12.4 desteğiyle yüklendi (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`).
- PyTorch GPU erişilebilirliği ve VRAM miktarı `torch.cuda.is_available()` ve `torch.cuda.get_device_name(0)` kod betikleri ile doğrulandı.

**Öğrenilenler:**
- Sanal ortam (`venv`) kullanımının sistem genelindeki Python paketleri ile proje bağımlılıklarının çakışmasını engellemedeki önemi pekiştirildi.
- CUDA sürücüsü ile PyTorch tekerlekleri (wheels) arasındaki uyumluluk matrisi incelendi; PyTorch'un kendi içinde getirdiği CUDA runtime sürümü ile sistem sürücüsünün ilişkisi anlaşıldı.

**Engeller:**
- Ubuntu 24.04 üzerinde Python 3.12'nin varsayılan olarak `externally-managed-environment` uyarısı vermesi nedeniyle paketlerin doğrudan `pip` ile sistem geneline yüklenmesi engellendi. Bu durum proje dizininde `python3 -m venv venv` komutu ile sanal ortam oluşturularak çözüldü.

**Sonraki Adım:**
- Git versiyon kontrol sisteminin yapılandırılması, `.gitignore` dosyasının hazırlanması ve PlantVillage veri setinin indirilmesi.

---

## Gün 2 - 22-06

**Görev:**
PlantVillage veri setinin indirilmesi, patates, domates ve biber sınıflarının ayıklanması ve hedef veri seti dizin yapısının (`data/PlantVillage/`) oluşturulması.

**Yapılan:**
- PlantVillage veri seti Kaggle API kullanılarak indirildi ve arşivden çıkarıldı.
- Veri setindeki 38 sınıf arasından yalnızca projede hedeflenen 3 bitki türüne (Patates, Domates, Biber) ait 15 hastalık ve sağlıklı durum sınıfı seçilerek `data/PlantVillage/` dizinine taşındı.
- Temizlenen 15 sınıf:
  - Biber (2 sınıf): `Pepper__bell___Bacterial_spot`, `Pepper__bell___healthy`
  - Patates (3 sınıf): `Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`
  - Domates (10 sınıf): `Tomato_Bacterial_spot`, `Tomato_Early_blight`, `Tomato_Late_blight`, `Tomato_Leaf_Mold`, `Tomato_Septoria_leaf_spot`, `Tomato_Spider_mites_Two_spotted_spider_mite`, `Tomato__Target_Spot`, `Tomato__Tomato_YellowLeaf__Curl_Virus`, `Tomato__Tomato_mosaic_virus`, `Tomato_healthy`
- Atıl sınıflar silinerek disk alanı optimize edildi. Toplam imaj sayısı ve sınıf bazlı dağılımlar kontrol edildi.

**Öğrenilenler:**
- Veri seti ön işleme adımında hedef problem uzayına odaklanmanın (15 sınıf sınırlaması) bellek ve GPU kaynak optimizasyonundaki kritik rolü kavrandı.
- Dizin yapısının PyTorch `ImageFolder` sınıfına uygun formatta (`root/class_name/image.jpg`) düzenlenmesinin veri yükleme boru hatlarını kolaylaştırdığı teyit edildi.

**Engeller:**
- PlantVillage veri setinde bazı klasör isimlerinde yer alan özel karakterler ve boşluklar dosya yollarında tutarsızlığa yol açtı. Klasör isimleri standartlaştırıldı (`_` ile birleştirildi).

**Sonraki Adım:**
- Veri seti istatistiklerinin analiz edilmesi (sınıf bazlı imaj sayıları, sınıf dengesizliği kontrolü) ve veri keşif betiğinin yazılması.

---

## Gün 3 - 23-06

**Görev:**
Keşifsel Veri Analizi (EDA) betiğinin yazılması, sınıf dağılımlarının incelenmesi ve görsellerin kanal (RGB) istatistiklerinin hesaplanması.

**Yapılan:**
- `notebooks/experiments/01_data_exploration.ipynb` dosyası oluşturuldu.
- 15 sınıftaki toplam 20,638 görselin sınıf bazlı dağılımı hesaplandı. En fazla görsele sahip sınıfın `Tomato__Tomato_YellowLeaf__Curl_Virus` (3,209 imaj), en az görsele sahip sınıfın `Potato___healthy` (152 imaj) olduğu ve belirgin bir sınıf dengesizliği (class imbalance) bulunduğu tespit edildi.
- Rastgele seçilen yaprak örnekleri görselleştirildi.
- Piksel değerlerinin kanal bazlı ortalamaları (`mean`) ve standart sapmaları (`std`) hesaplandı:
  - Calculated Mean: `[0.485, 0.456, 0.406]`
  - Calculated Std: `[0.229, 0.224, 0.225]`
  (ImageNet standart değerleri ile örtüştüğü teyit edildi).

**Öğrenilenler:**
- Sınıf dengesizliğinin model değerlendirme aşamasında doğruluk (accuracy) metriğini yanıltıcı kılabileceği, bu nedenle Macro F1-Score metriğinin takibinin zorunlu olduğu öğrenildi.
- Giriş görsellerinin normalizasyonunun gradyan akışını kararlı kıldığı pekiştirildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- PyTorch `Dataset` ve `DataLoader` boru hattının kurulması, veri seti bölünmesi (Train/Val/Test).

---

## Gün 4 - 24-06

**Görev:**
`src/dataset.py` modülünün yazılması; veri setinin Train (%80), Val (%10) ve Test (%10) olarak bölünmesi ve PyTorch DataDataLoader'ların oluşturulması.

**Yapılan:**
- `src/dataset.py` betiği oluşturuldu.
- `torchvision.transforms` ile ön işleme boru hattı tanımlandı: `Resize((224, 224))`, `ToTensor()`, `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
- `torchvision.datasets.ImageFolder` kullanılarak tüm veri seti yüklendi (20,638 imaj, 15 sınıf).
- Tekrarlanabilirlik (reproducibility) amacıyla rastgele tohum (`SEED = 42`) sabitlendi.
- `torch.utils.data.random_split` ile veri seti bölündü:
  - Eğitim (Train): 16,511 imaj (%80)
  - Doğrulama (Val): 2,063 imaj (%10)
  - Test: 2,064 imaj (%10)
- GTX 1050 Ti VRAM sınırlarına (4GB) uygun olarak `BATCH_SIZE = 32` belirlendi ve DataDataLoader'lar hazırlandı (`num_workers=2`).

**Öğrenilenler:**
- Rastgele bölme (random split) işlemlerinde tohum (seed) sabitlemenin deneysel tekrarlanabilirlik açısından hayati olduğu pekiştirildi.
- PyTorch `DataLoader` yapısındaki `pin_memory` ve `num_workers` parametrelerinin CPU-GPU veri transfer hızına etkisi kavrandı.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- ResNet-18 model mimarisinin `src/model.py` içerisinde tanımlanması ve son katmanın 15 sınıfa göre uyarlanması.

---

## Gün 5 - 25-06

**Görev:**
`src/model.py` modülünün oluşturulması, ImageNet ön eğitimli ResNet-18 modelinin yüklenmesi ve classification head katmanının 15 sınıfa güncellenmesi.

**Yapılan:**
- `src/model.py` dosyası yazıldı.
- `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` ile ImageNet üzerinde önceden eğitilmiş ağırlıklar yüklendi.
- ResNet-18'in son tam bağlantılı katmanı (`model.fc`), 512 giriş özelliğinden 15 çıkış sınıfına eşleme yapacak şekilde yeniden tanımlandı (`nn.Linear(512, 15)`).
- Toplam parametre sayısı (11,184,463) ve eğitilebilir parametre sayısı doğrulandı.

**Öğrenilenler:**
- Transfer Learning (Aktarımlı Öğrenme) kavramının temelleri pekiştirildi; ön eğitimli evrişimsel katmanların (feature extractor) genel görsel özellikleri (kenarlar, dokular) hazır olarak sunduğu anlaşıldı.
- GTX 1050 Ti (4GB VRAM) gibi kısıtlı donanımlarda ResNet-18'in düşük parametre sayısı ve hızlı çıkarım süresi nedeniyle en ideal mimari olduğu teyit edildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Eğitim betiğinin (`src/train.py`) yazılması, kayıp fonksiyonu (CrossEntropyLoss) ve optimizer (Adam) ayarlarının yapılması.

---

## Gün 6 - 26-06

**Görev:**
`src/train.py` eğitim döngüsünün yazılması, kayıp ve doğruluk metriklerinin takibi, ilk deneme koşusunun yapılması.

**Yapılan:**
- `src/train.py` betiği oluşturuldu.
- `nn.CrossEntropyLoss()` kayıp fonksiyonu ve `optim.Adam(model.parameters(), lr=0.001)` optimizer'ı tanımlandı.
- Eğitim ve doğrulama aşamalarını içeren ana döngü kuruldu. Her epoch sonunda `train_loss`, `train_acc`, `val_loss`, `val_acc` değerlerinin ekrana basılması sağlandı.
- Model `cuda` cihazına transfer edilerek 1 epoch'luk deneme eğitimi koşturuldu. Eğitim adımları başarıyla tamamlandı.

**Öğrenilenler:**
- PyTorch eğitim döngüsünün 5 temel adımı uygulandı: `optimizer.zero_grad()`, `forward pass`, `loss.backward()`, `optimizer.step()`, ve metrik biriktirme.
- `model.train()` ve `model.eval()` modlarının Dropout ve BatchNorm katmanları üzerindeki davranış farklılıkları pekiştirildi.

**Engeller:**
- `DataLoader` varsayılan olarak tek çekirdekte yükleme yaptığı için CPU tarafında veri yükleme darboğazı oluştu. `num_workers=2` düzenlemesi yapılarak GPU kullanımı optimize edildi.

**Sonraki Adım:**
- Çoklu epoch eğitimi, öğrenme oranı zamanlayıcısı (ReduceLROnPlateau) ve erken durdurma (early stopping) mekanizmasının eklenmesi.

---

## Gün 7 - 30-06

**Görev:**
`src/model.py` modülünün parametrik hale getirilmesi, dinamik sınıf sayısı desteği ve `train.py` entegrasyonu.

**Yapılan:**
- `src/model.py` içerisindeki `get_crop_disease_model` fonksiyonu `num_classes` parametresi alacak şekilde dinamikleştirildi.
- Sınıf sayısının sert kodlanması (hardcode) engellenerek `src.dataset.NUM_CLASSES` üzerinden beslenmesi sağlandı.
- Model tanımının cihaz atamasını (`.to(device)`) çağıran tarafa bırakarak esneklik kazandırıldı.
- `train.py` içerisinde modüler import gerçekleştirildi (`from src.model import get_crop_disease_model`).

**Öğrenilenler:**
- Modüler yazılım mimarisi ilkeleri çerçevesinde veri yükleme, model tanımı ve eğitim döngüsü sorumluluklarının kesin hatlarla ayrılmasının koda sağladığı esneklik kavrandı.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Komut satırı argümanlarının (`argparse`) eklenmesi, çoklu epoch eğitimi ve model kayıt (`torch.save`) altyapısının kurulması.

---

## Gün 8 - 01-07

**Görev:**
`src/train.py` dosyasına `argparse` desteği, validation döngüsü, checkpoint kaydetme ve tam eğitim koşusunun başlatılması.

**Yapılan:**
- `--epochs`, `--batch-size`, `--lr`, `--patience` parametrelerini alan `argparse` altyapısı eklendi.
- Eğitim sonunda en iyi validation başarımına sahip modelin `checkpoints/best_crop_model.pth` dosyasına kaydedilmesi sağlandı.
- Checkpoint içerisine yalnızca ağırlıklar değil, `epoch`, `optimizer_state_dict`, `class_to_idx`, `idx_to_class` ve `transform_params` bilgileri de üstveri (metadata payload) olarak eklendi.
- 15 epoch'luk ilk tam eğitim koşusu çalıştırıldı.

**Öğrenilenler:**
- Üretime hazır (production-ready) checkpoint dosyalarında model üstverilerinin saklanmasının, modellerin daha sonra başka sistemlerde bağımsız olarak yüklenmesini ne kadar kolaylaştırdığı öğrenildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Eğitim sonuçlarının analizi, metrik kayıt betiği (`src/utils.py`) ve öğrenme eğrilerinin görselleştirilmesi.

---

## Gün 9 - 02-07

**Görev:**
`src/utils.py` modülünün yazılması, eğitim loglarının CSV olarak saklanması ve öğrenme eğrilerinin (`learning_curves.png`) çizdirilmesi.

**Yapılan:**
- `src/utils.py` modülü yazılarak `save_training_log` ve `plot_learning_curves` fonksiyonları eklendi.
- Eğitim sırasında her epoch'un `train_loss`, `train_acc`, `val_loss`, `val_acc` ve `lr` değerleri `results/training_log.csv` dosyasına kaydedildi.
- Matplotlib kullanılarak ikili grafik içeren `results/learning_curves.png` oluşturuldu.

**Öğrenilenler:**
- Eğitim eğrilerinin görselleştirilmesinin aşırı öğrenme (overfitting) ve eksik öğrenme (underfitting) durumlarını teşhis etmedeki rolü pekiştirildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Bağımsız test seti değerlendirme betiğinin (`src/evaluate.py`) yazılması ve karmaşıklık matrisinin (Confusion Matrix) çizdirilmesi.

---

## Gün 10 - 03-07

**Görev:**
`src/evaluate.py` betiğinin geliştirilmesi, held-out test seti üzerinde model değerlendirmesi ve karmaşıklık matrisinin oluşturulması.

**Yapılan:**
- `src/evaluate.py` yazıldı; `checkpoints/best_crop_model.pth` yüklenerek 2,065 imajlık test seti üzerinde çıkarım yapıldı.
- Genel test doğruluğu (Overall Accuracy): **%99.27** olarak hesaplandı.
- Sınıf bazlı Precision, Recall ve F1-Score metrikleri `scikit-learn` ile hesaplanarak `results/plantvillage_metrics.json` dosyasına kaydedildi.
- Seaborn kütüphanesi ile 15x15 normalized `results/confusion_matrix.png` görseli üretildi.

**Öğrenilenler:**
- Laboratuvar veri setlerinde (PlantVillage) homojen arka planlar nedeniyle evrişimsel ağların yüksek başarım gösterdiği teyit edildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- PlantDoc saha veri setinin projeye dahil edilmesi ve sıfır-vuruş (zero-shot) genelletirme testlerinin planlanması.

---

## Gün 11 - 06-07

**Görev:**
PlantDoc gerçek saha veri setinin `data/plantdoc/` dizinine aktarılması ve klasör haritalama betiğinin (`src/setup_plantdoc.py`) yazılması.

**Yapılan:**
- Gerçek tarla koşullarında çekilen görsellerden oluşan PlantDoc veri seti projeye eklendi.
- PlantDoc klasör isimlerini projedeki 15 standart sınıfla eşleştiren `src/setup_plantdoc.py` betiği yazıldı.
- Test klasöründe 102 adet geçerli yaprak görseli haritalandı.

**Öğrenilenler:**
- Farklı kaynaklardan gelen açık veri setlerinin etiket standartlarının uyumlaştırılmasının (data harmonization) önemi kavrandı.

**Engeller:**
- PlantDoc etiketlerinin bir kısmının PlantVillage sınıf isimlerinden farklı olması nedeniyle otomatik eşleme fonksiyonu yazıldı.

**Sonraki Adım:**
- PlantDoc sıfır-vuruş (zero-shot) baseline değerlendirme betiğinin (`src/evaluate_plantdoc.py`) yazılması.

---

## Gün 12 - 07-07

**Görev:**
`src/evaluate_plantdoc.py` betiği ile laboratuvar modelinin gerçek saha fotoğrafları üzerindeki sıfır-vuruş (zero-shot) genelletirme performansının ölçülmesi.

**Yapılan:**
- Yalnızca PlantVillage stüdyo görselleriyle eğitilmiş model, hiçbir uyarlama yapılmadan PlantDoc test seti üzerinde çalıştırıldı.
- **Sıfır-Vuruş (Zero-Shot) Doğruluğu:** **%15.69** olarak ölçüldü (102 imajın yalnızca 16'sı doğru tahmin edilebildi).
- Sonuçlar `results/plantdoc_baseline_metrics.json` dosyasına kaydedildi.

**Öğrenilenler:**
- Laboratuvar ortamında %99.27 doğruluk veren modelin sahada %15.69'a düşmesi, yapay zekanın stüdyo arka planlarını ve ışıklandırma düzenlerini ezberlediğini (Shortcut Learning / Domain Shift) somut olarak kanıtladı.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- PlantDoc veri seti üzerinde ince ayar (Fine-Tuning / Domain Adaptation) betiğinin (`src/finetune_plantdoc.py`) yazılması.

---

## Gün 13 - 08-07

**Görev:**
`src/finetune_plantdoc.py` betiğinin yazılması, katman dondurma (layer freezing) ve Transfer Learning ile saha adaptasyonunun gerçekleştirilmesi.

**Yapılan:**
- ResNet-18 modelinin ilk evrişimsel katmanları (`layer1`, `layer2`) donduruldu; üst katmanlar (`layer3`, `layer4`, `fc`) düşük öğrenme oranıyla ($lr=10^{-4}$) 5 epoch boyunca PlantDoc üzerinde eğitildi.
- İnce ayar sonrası PlantDoc test doğruluğu **%22.55** seviyesine yükseldi (+%6.86 net artış).
- İnce ayarlı model `checkpoints/best_plantdoc_model.pth` olarak kaydedildi ve karşılaştırma metrikleri `results/plantdoc_before_after.json` dosyasına yazıldı.

**Öğrenilenler:**
- Kısıtlı verili saha verilerinde katman dondurmanın ezberlemeyi önlemedeki rolü öğrenildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Modellerin üretim dağıtımı için ONNX ve TorchScript formatlarına dönüştürülmesi.

---

## Gün 14 - 09-07

**Görev:**
`src/utils.py` içerisine model dışa aktarım (`export_model_formats`) fonksiyonunun eklenmesi, TorchScript (`.pt`) ve ONNX (`.onnx`) modellerinin üretilmesi.

**Yapılan:**
- `torch.jit.trace` ile TorchScript modeli (`checkpoints/crop_disease_model.pt`) üretildi.
- `torch.onnx.export` kullanılarak dinamik batch boyutlu ONNX modeli (`checkpoints/crop_disease_model.onnx`) dışa aktarıldı.
- ONNX Runtime ile modelin PyTorch bağımlılığı olmadan çıkarım yaptığı doğrulandı.

**Öğrenilenler:**
- ONNX formatının farklı platformlar ve diller arası model taşınabilirliğindeki önemi pekiştirildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Tekli görsel çıkarımı için üretim ortamı çıkarım betiğinin (`src/predict.py`) yazılması.

---

## Gün 15 - 10-07

**Görev:**
`src/predict.py` bağımsız çıkarım modülünün geliştirilmesi ve örnek yaprak fotoğrafları ile test edilmesi.

**Yapılan:**
- Herhangi bir dış yaprak fotoğrafını alıp ön işleme tabi tutan, ONNX veya PyTorch checkpoint'i üzerinden çıkarım yapıp top-k olasılıkları ve milisaniye bazlı gecikmeyi döndüren `predict_image` fonksiyonu yazıldı.
- Örnek domates ve patates yaprakları ile çıkarım doğrulandı.

**Öğrenilenler:**
- Üretim ortamlarında (production) model çıkarım boru hatlarının esnek ve bağımsız tasarlanmasının önemi kavrandı.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Aşırı öğrenme (overfitting) analizi, Erken Durdurma (Early Stopping), L2 Weight Decay, Dropout ve Saha Simülasyon Veri Çoğullaması ile model boru hattının iyileştirilmesi.

---

## Gün 16 - 13-07

**Görev:**
Eğitim boru hattında aşırı öğrenme (overfitting) teşhisi, Erken Durdurma (val_loss takibi), L2 Weight Decay (1e-4), Dropout (p=0.3) ve Saha Simülasyonu Veri Çoğullaması (`--field-aug`) entegrasyonu.

**Yapılan:**
- **Teşhis ve Nedenler:**
  - Önceki eğitim runs incelemesinde modelin 12. epoch'ta eğitim kaybını neredeyse sıfırladığı (%99.8 train acc), ancak doğrulama kaybının (val_loss) 15. epoch'ta `0.03` seviyesinden `0.27` seviyesine fırlayarak aşırı öğrendiği tespit edildi.
  - Eski kodun en düşük val_loss veren epoch yerine son epoch'taki (epoch 15) en kötü ağırlıkları sakladığı belirlendi.
  - Ayrıca modelin laboratuvar stüdyo arka planlarını ezberlemesi nedeniyle PlantDoc sıfır-vuruş başarımı %15.69 seviyesinde kalmıştı.
- **İyileştirmeler:**
  1. `src/model.py` içerisindeki classification head katmanına **Dropout (p=0.3)** eklendi.
  2. `src/dataset.py` içerisine saha şartlarını simüle eden **Saha Veri Çoğullama Boru Hattı** (`field_sim_transform`: `RandomResizedCrop(0.7-1.0)`, `ColorJitter`, `RandomRotation(15°)`, `RandomErasing cutout`) ve `--field-aug` bayrağı eklendi.
  3. `src/train.py` içerisine L2 Weight Decay (`weight_decay=1e-4`), `ReduceLROnPlateau(patience=2, factor=0.5)` ve en düşük validation kaybını takip eden **Erken Durdurma (Early Stopping - patience=3)** mekanizması entegre edildi. En iyi checkpoint (`checkpoints/best_crop_model.pth`) artık en düşük val_loss veren epoch'ta kaydedilmektedir.
- **Deneysel Sonuçlar:**
  - Eğitim 9. epoch'ta Erken Durdurma ile otomatik olarak sonlandırıldı.
  - **En İyi Checkpoint:** **Epoch 6** (`val_loss=0.1233`, `val_acc=0.9598`). Son epoch (Epoch 9) yerine Epoch 6 ağırlıkları başarıyla kaydedildi.
  - **PlantVillage Test Doğruluğu:** **%96.13** (Saha çoğullamalı model).
  - **PlantDoc Sıfır-Vuruş (Zero-Shot) Doğruluğu:** Eski **%15.69** seviyesinden **%26.47** seviyesine yükseldi (**+%10.78 net artış**)!
  - Güncellenmiş ağırlıklar ONNX formatında (`checkpoints/crop_disease_model.onnx`) dışa aktarıldı.

**Öğrenilenler:**
- Eğitim aşamasında uygulanan saha simülasyonu veri çoğullamalarının (ColorJitter, Cutout, RandomCrop) evrişimsel ağların arka plan stüdyo kestirmelerini (shortcut learning) ezberlemesini engelleyerek gerçek dünya sıfır-vuruş başarımını %15.69'dan %26.47'ye çıkardığı deneysel olarak kanıtlandı.
- Validation kaybına dayalı Erken Durdurma mekanizmasının aşırı öğrenmiş son epoch ağırlıkları yerine en genelleyici epoch ağırlıklarını korumadaki kritik rolü kavrandı.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Proje dokümantasyonu, README.md ve Staj-I rapor çıktılarının güncellenmesi.

---

## Gün 17 - 14-07

**Görev:**
Kod tabanı temizliği, performans metriklerinin raporlanması ve sistem genel testi.

**Yapılan:**
- Tüm deneysel eğitim çıktıları ve güncellenen metrikler `results/` dizininde doğrulandı.
- `README.md` dosyası güncellenerek yeni sıfır-vuruş metrikleri (%26.47) ve erken durdurma bilgileri eklendi.

**Öğrenilenler:**
- Metrik sonuçlarının ve deneysel kazanımların düzenli dokümante edilmesinin raporlama süreçlerine katkısı pekiştirildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Proje dizin temizliği ve exploratif betiklerin arşivlenmesi.

---

## Gün 18 - 15-07

**Görev:**
Proje temizliği, geçici dosyaların silinmesi ve arşivleme.

**Yapılan:**
- Gereksiz geçici dosyalar ve loglar temizlendi.
- Projenin `git status` temizliği sağlandı.

**Öğrenilenler:**
- Proje sürümlerinin temiz tutulmasının ekip çalışmasındaki önemi kavrandı.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- `src/predict.py` modülünün son testi ve Staj-I kapanış hazırlığı.

---

## Gün 19 - 17-07

**Görev:**
Ölü kod ve keşif notebook'larını temizleme; `src/predict.py` üretim çıkarım betiğini tamamlama.

**Yapılan:**
- `notebooks/01_data_exploration.ipynb` ve `notebooks/02_pytorch_training_tutorial.ipynb` dosyaları `notebooks/experiments/` dizinine arşivlendi.
- `src/` dizini temizlenerek yalnızca modüler, üretime hazır üretim betikleri (`dataset.py`, `model.py`, `train.py`, `evaluate.py`, `evaluate_plantdoc.py`, `finetune_plantdoc.py`, `predict.py`, `utils.py`) bırakıldı.
- Staj-II web uygulaması backend'i tarafından doğrudan import edilebilen veya CLI üzerinden çalıştırılabilen `src/predict.py` tekli görüntü çıkarım betiği tamamlandı ve test edildi.

**Öğrenilenler:**
- Temiz kod ilkeleri (Clean Code) ve proje dizin düzeninin yazılım sürdürülebilirliğine katkısı deneyimlendi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Staj-I sunum ve rapor hazırlığı; PLAN.md çıkış koşullarının gözden geçirilmesi.

--- Hafta Sonu: 18, 19-07

## Gün 20 - 20-07

**Görev:**
Staj-I sunum/rapor hazırlığı; `PLAN.md` çıkış koşullarını gözden geçirme ve Staj-I aşamasını tamamlama.

**Yapılan:**
- `PLAN.md` dosyasındaki Phase 0'dan Phase 7'ye kadar tüm çıkış koşulları kontrol edildi ve tüm aşamaların eksiksiz tamamlandığı doğrulandı.
- Tüm nicel sonuçlar (JSON metrikleri, karmaşıklık matrisi, öğrenme eğrileri) resmi Staj Raporu (*Staj-I Raporu*) hazırlığı için düzenlendi.
- Yeni teknik iş planlanmayarak Staj-I kapsamı başarıyla kapatıldı.

**Öğrenilenler:**
- Bir mühendislik projesinde önceden tanımlanmış çıkış koşullarına (exit conditions) sadık kalmanın kapsam kaymasını (scope creep) önlemedeki kritik rolü anlaşıldı.

**Engeller:**
- Yaşanmadı.