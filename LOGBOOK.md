## Gün 1 — 22 Haziran 2026

**Görev:** 
Ortam kurulumu, proje dizin mimarisinin inşası ve donanım kısıtlamalarına uygun veri setinin edinimi.

**Yapılan:** 
- `data/`, `src/`, `notebooks/`, `checkpoints/`, `results/` klasör yapısı oluşturuldu ve `PLAN.md` eklendi.
- `modprobe` komutu kullanılarak inaktif olan donanım sürücüsü aktifleştirildi ve GTX 1050 Ti (4GB VRAM, CUDA 11.8) ortamı doğrulandı.
- 38 sınıflı orijinal PlantVillage veri seti yerine, 20 günlük süre ve VRAM limiti gözetilerek `emmarex/plantdisease` 15 sınıflı alt kümesi (Tomato, Potato, Pepper) seçildi ve indirildi.
- Veri seti dizin yapısının `ImageFolder` ile uyumluluğu kontrol edildi.

**Öğrenilenler:** 
- Linux tabanlı sistemlerde donanım sürücülerinin yüklü olmasının yeterli olmadığı, çekirdek (kernel) modüllerinin manuel olarak tetiklenmesi gerekebileceği tecrübe edildi.
- Veri bilimi projelerinde veri seti ölçeğinin, mevcut donanım ve proje takvimi ile ters orantılı olarak optimize edilmesinin önemi anlaşıldı.

**Engeller:** 
- CUDA sürücüsünün başlangıçta yüklenmemesi (aynı gün çözüldü).
- Orijinal veri setinin aşırı büyük olması (15 sınıflı alt kümeye geçilerek çözüldü).

**Sonraki Adım:** 
- `dataset.py` dosyasının yazılması, veri ön işleme boru hattının kurulması ve ilk batch çekilerek tensor boyutlarının doğrulanması.

--- 

## Gün 2 — 23 Haziran 2026

**Görev:** 
`dataset.py` üzerinden veri ön işleme (transform) boru hattının kurulması ve veri kümesinin RAM'e yüklenerek doğrulanması.

**Yapılan:** 
- `transforms.Compose` içerisine Resize (224x224), ToTensor ve ImageNet normalizasyon değerleri tanımlandı.
- `datasets.ImageFolder` kullanılarak veri seti hiyerarşisi başarıyla tarandı ve `class_to_idx` eşleştirmeleri terminale yazdırıldı.
- `batch_size=16` ile `DataLoader` başlatıldı.
- `next(iter(data_loader))` ile ilk veri grubu çekildi. Görüntü tensorlerinin $[16, 3, 224, 224]$ ve etiketlerin $[16]$ boyutlarında olduğu matematiksel olarak kanıtlandı.

**Öğrenilenler:** 
- PyTorch kütüphanesinin iç yapısını, obje özelliklerini ve metodlarını keşfetmek için Python'ın yerleşik `dir()` ve `help()` fonksiyonlarının hata ayıklama (debugging) sürecinde ne kadar kritik olduğu deneyimlendi.

**Engeller:** 
- Gemini API günlük kota (20 RPD) sınırı aşılarak HTTP 429 hatası alındı.
- Groq API üzerinden inline autocomplete (kod tamamlama) kullanılırken yüksek frekanslı istekler sebebiyle TPM sınırı aşıldı; otonom kod tamamlama özelliği tamamen kapatılarak ve prompt dosyası küçültülerek ağ darboğazı çözüldü.

**Sonraki Adım:** 
- `full_dataset` nesnesinin tekrarlanabilirlik (manual_seed) gözetilerek `random_split` mekanizması ile %80 Eğitim, %10 Doğrulama, %10 Test olarak ayrıştırılması.

---

## Gün 3

**Yapılan:**
- `src/dataset.py` içerisinde `transforms.Compose` ile Resize (224x224), ToTensor ve ImageNet normalizasyon (`MEAN_VALUE`, `STD_VALUE`) tanımlandı; `datasets.ImageFolder` ile veri seti hiyerarşisi taranarak `class_to_idx` eşleştirmesi oluşturuldu.
- `torch.utils.data.random_split` ile veri seti %80 eğitim, %10 doğrulama, %10 test olacak şekilde sabit uzunluklarda bölündü; `torch.Generator().manual_seed(SEED)` kullanılarak bölünmenin her çalıştırmada aynı indeksleri üretmesi sağlandı.
- `train_dataloader`, `val_dataloader`, `test_dataloader` nesneleri `batch_size=16` ile oluşturuldu; yalnızca eğitim loader'ı `shuffle=True` olarak ayarlandı, doğrulama ve test loader'ları sabit sıralamada bırakıldı.
- `src/train.py` dosyası oluşturuldu; `from src.dataset import ...` yapısı ile dataloader'lar içe aktarıldı (proje kökünden `python -m src.train` komutu ile çalıştırılarak modül yolu sorunu çözüldü).
- Pretrained ResNet-18 (`weights="IMAGENET1K_V1"`) yüklendi, son `fc` katmanı PlantVillage'daki 15 sınıfa uyacak şekilde değiştirildi.
- `torch.nn.CrossEntropyLoss` ve Adam optimizer tanımlandı; eğitim döngüsü bir epoch boyunca `train_dataloader` üzerinde forward, loss hesaplama, backward ve optimizer step adımlarını çalıştıracak şekilde yazıldı ve ortalama kayıp (loss) terminale yazdırıldı.
- `notebooks/01_data_exploration.ipynb` dosyası oluşturuldu; `sys.path.append("..")` ile proje köküne erişim sağlanarak `src.dataset` içe aktarıldı.
- `next(iter(train_dataloader))` ile bir batch çekildi; görüntü tensörünün `[16, 3, 224, 224]`, etiket tensörünün `[16]` boyutunda olduğu doğrulandı.
- `class_to_idx` sözlüğü ters çevrilerek (`idx_to_class`) etiket indekslerinden sınıf isimlerine dönüşüm sağlandı.
- Normalizasyonun tersini alan (`img * std + mean`) bir fonksiyon yazıldı; `mean`/`std` `(3,1,1)` boyutuna `reshape` edilerek broadcasting uyumu sağlandı, `permute(1,2,0)` ile kanal sırası `(H,W,C)`'ye çevrildi, `np.clip(0,1)` ile değer aralığı sınırlandırıldı.
- `show_image()` fonksiyonu ile `plt.subplots(3,3)` üzerinden 9 görüntülük bir grid oluşturuldu; tüm görüntülerin doğru renkte ve doğru etiketle göründüğü görsel olarak teyit edildi — pipeline'ın transform, normalizasyon ve sınıf eşleştirme aşamalarının uçtan uca doğru çalıştığı kanıtlandı.

**Öğrenilenler:**
- Python'da farklı dosyalar arasında kod ve nesneleri paylaşmak için `from <modül_adı> import <nesne_adı>` yapısı kullanılır; bu, C'deki `#include` gibi metin yapıştırma değildir — modül bir defa çalıştırılıp `sys.modules` içinde önbelleğe alınır.
- `random_split`'e sağlanan `torch.Generator`'ın `manual_seed` ile ayarlanması, rastgele indeks seçim sürecini deterministik bir kaynağa bağlar; aynı seed tekrar kullanıldığında veri setinin aynı eğitim/validasyon/test bölmeleri yeniden üretilir. Seed yalnızca deterministik bir başlangıç noktasıdır.
- `Subset` nesneleri orijinal `ImageFolder` veri setine referans tutar; dolayısıyla aynı transformasyonlar ve sınıf-etiket haritalaması paylaşılır, ancak her `DataLoader` yalnızca kendisine atanmış indeksleri kullanır.
- `DataLoader` uzunluğu, ilgili `Subset`'in örnek sayısının `batch_size`'a bölünmesiyle (yukarı yuvarlanarak) elde edilir.
- Eğitim döngüsünün yalnızca bir epoch çalıştırılarak kaybın azaldığının doğrulanması, sonraki aşamalara geçiş için temel mekanizmayı kurar.
- Notebook'un çalışma dizini (`cwd`) ile script'in çalıştırıldığı dizin arasındaki farkın `import` hatalarına yol açabildiği; `os.chdir()`'in durum bağımlı (stateful) ve riskli olduğu, `sys.path.append()`'in ise yan etkisiz olduğu için tercih edilmesi gerektiği deneyimlendi.
- Modül diskte değişse bile aynı kernel oturumunda eski halinin önbellekte tutulduğu, bu yüzden `importlib.reload()` gerekliliği gözlemlendi.
- Tensör broadcasting kurallarının boyutları sağdan sola hizaladığı; `(3,)` boyutundaki bir vektörün `(3,224,224)` ile kanal bazında çarpılabilmesi için `(3,1,1)`'e `reshape` edilmesi gerektiği matematiksel olarak kanıtlandı.
- `.reshape()`, `.permute()` gibi tensör metodlarının yerinde (in-place) değişiklik yapmadığı; dönüş değerinin değişkene atanması gerektiği, atanmadığında işlemin sessizce kaybolduğu gözlemlendi.
- Bir metodun referansını yazmak (`.item`) ile çağırmak (`.item()`) arasındaki farkın kritik bir hata kaynağı olduğu deneyimlendi.
- `plt.imshow()` (global/örtük) ile `ax.imshow()` (belirli bir subplot'a açık referans) arasındaki farkın, çoklu subplot grid'lerinde doğru görselleştirme için zorunlu olduğu öğrenildi.
- Google Cloud API geliştirici kotalarının tüketici aboneliklerinden (Google One) tamamen bağımsız çalıştığı, faturalandırma katmanına (Tier 1) geçişin ücretsiz kullanım hakkını iptal ettiği ve kota aşımlarının HTTP 429/503 erişim engelleriyle sonuçlandığı deneyimlendi. Bu ağ darboğazlarını aşmak için Groq/Llama3 gibi sağlayıcıdan bağımsız, yedekli LLM altyapılarının (fallback) Continue gibi araçlara entegre edilmesinin önemi kavrandı.

**Engeller:**
- Notebook'un çalışma dizini ile proje kökü arasındaki farktan kaynaklanan `ModuleNotFoundError: No module named 'src'` hatası alındı; `sys.path` manipülasyonu ile çözüldü.
- Ardışık `os.chdir("..")` çağrıları sonucu çalışma dizini proje kökünün dışına (Ubuntu kök dizinine kadar) çıkıldı; kernel yeniden başlatılarak ve mutlak yol ile düzeltildi.
- `dataset.py` dosyasına eklenen yeni değişkenlere (`MEAN_VALUE`, `STD_VALUE`) notebook'ta erişilememesi, modül önbelleğinden (`sys.modules`) kaynaklandı; `importlib.reload()` ile çözüldü.
- Gemini API günlük kota (20 RPD) sınırı aşılarak HTTP 429 hatası alındı.
- Groq API üzerinden inline autocomplete kullanılırken yüksek frekanslı istekler sebebiyle TPM sınırı aşıldı; otonom kod tamamlama özelliği kapatılarak ve prompt dosyası küçültülerek çözüldü.

---

## Gün 4

**Yapılan:**
- Notebook: notebooks/02_pytorch_training_tutorial.ipynb

**Öğrenilenler:**
- Öğrenilenler: Tek bir eğitim epoch'u boyunca kaybın azaldığını doğrulamak, tüm sistem entegrasyonunun temelini oluşturur. 
- in_features değerini mevcut model.fc katmanından almak ve torch.manual_seed() ile tüm rastgele işlemleri sabitlemek, yeniden üretilebilir eğitim deneyleri sağlar.

---

## Gün 5 - 26-06

**Öğrenilenler:**
- Bugün PyTorch resmi dokümantasyonundan tensor oluşturmayı inceledim: Python listelerinden, NumPy dizilerinden ve mevcut bir tensörden (ones_like, rand_like) yeni tensörler üretildi.
- Tensor oluşturma, şekil/dtype/device sorgulama, GPU’ye taşıma, dilimleme, eleman‑bazlı çarpma, in‑place işlemler ve NumPy‑PyTorch ortak belleği nasıl yönetileceği öğrenildi.


--- Hafta Sonu: 27, 28-06

## Gün 6 - 29-06

**Görev:**
Prob eğitim döngüsünü tamamlama: 3–5 epoch çalıştırıp kaybın düştüğünü doğrulama; prob checkpoint'i arşivleme veya silme.

**Yapılan:**
- `torch.device("cuda" if torch.cuda.is_available() else "cpu")` kullanılarak GPU/CPU cihaz tespiti gerçekleştirildi ve model `cuda` cihazına taşındı.
- `tqdm` kütüphanesi yardımıyla tek epoch'luk eğitim döngüsü yazıldı.
- Eğitim döngüsünde her batch için; görüntü tensörünün $[N, C, H, W]$ formatında `inputs.to(device)` ve etiket tensörünün `labels.to(device)` ile GPU'ya aktarılması, `optimizer.zero_grad()`, `forward pass` gerçekleştirilmesi, `criterion(output, labels)` ile kayıp (loss) hesaplanması, `loss.backward()` ile gradyanların geriye yayılması ve `optimizer.step()` ile ağırlıkların güncellenmesi adımlarını içeren 5 adımlı eğitim boru hattı uygulandı.
- `loop.set_postfix(loss=...)` ile her batch sonrasında anlık kayıp değeri `tqdm` arayüzü üzerinde canlı olarak gösterildi.
- Tek epoch sonundaki ortalama kayıp hesaplanarak `0.4739` olarak yazdırıldı.
- `nvidia-smi` komutu ile `GTX 1050 Ti` üzerinde %97 GPU kullanımı ve `7.67 batch/s` işlem hızı elde edildiği doğrulandı.
- `src/dataset.py` dosyasındaki dataloader nesnelerinde `num_workers=4` ayarlanarak CPU veri yükleme (`data loading`) darboğazı giderildi.

**Öğrenilenler:**
- PyTorch eğitim döngüsündeki 5 temel adımın ve işlevlerinin detayları öğrenildi:
  - `zero_grad()`: Gradyanların her adımda birikmesini (accumulation) engelleyerek her adımda temiz gradyanlarla başlanmasını sağlar.
  - `forward pass`: Girdileri model katmanlarından geçirerek çıktıları üretir ve hesap grafiğini (computation graph) oluşturur.
  - `loss.backward()`: Kayıp değerinden geriye doğru gradyanları hesaplayarak model parametrelerinin `.grad` niteliğini doldurur; ağırlıkları güncellemez.
  - `optimizer.step()`: Biriken `.grad` değerlerini kullanarak optimizer algoritmasına göre model ağırlıklarını günceller.
- Transfer learning yaklaşımının etkisi tecrübe edildi; ImageNet ağırlıklarıyla başlatılan model sayesinde ilk epoch sonunda ortalama kaybın `0.474` seviyesine kadar gerilediği gözlemlendi.
- `num_workers` parametresinin değeri ile GPU'nun veriyle beslenme hızı arasındaki doğrudan ilişki ve `nvidia-smi` aracıyla GPU kullanım oranının izlenmesi yöntemi öğrenildi.

**Engeller:**
- `DataLoader` varsayılan olarak verileri tek çekirdekte yüklediği için CPU tarafında bir veri yükleme (`data loading`) darboğazı oluştu ve GPU (`GTX 1050 Ti`) boşta bekleyerek tam verimle çalışamadı. `num_workers=4` düzenlemesi yapılarak bu veri darboğazı giderildi ve GPU kullanımı %97 seviyesine çıkarıldı.
- Görev kapsamında 3–5 epoch çalıştırılması hedeflenmiş olsa da, boru hattının ilk doğrulamasında tek epoch üzerinden ilerlendi. Çoklu epoch ve doğrulama süreçleri modelin modüler yapıya kavuşturulmasından sonraya bırakıldı.

**Sonraki Adım:**
- Model mimarisini modüler hale getirmek üzere `src/model.py` dosyasının oluşturulması, model tanımının oraya taşınması ve sınıf sayısının dinamik olarak ayarlanması.

## Gün 7 - 30-06

**Görev:**
`src/model.py` oluşturma: ResNet18 yükleme, 15 sınıflı `fc` değişimi ve `train.py`'den modüler import.

**Yapılan:**
- Modüler mimari tasarımı doğrultusunda `src/model.py` dosyası oluşturuldu ve ResNet18 model tanımı ile `fc` (fully connected) katman değişimi `train.py` dosyasından bu yeni modüle taşındı.
- Model kurucu fonksiyon olan `get_crop_disease_model`, sınıf sayısını (`num_classes`) parametrik olarak alacak şekilde dinamikleştirildi; böylece sınıf sayısı model dosyası içerisinde hardcoded olarak tanımlanmak yerine, `src.dataset.NUM_CLASSES` üzerinden dinamik olarak beslendi.
- Model tanımının kendi içinde cihaz ataması (device binding) yapmadığı, yalnızca inşa edilen modeli döndürdüğü teyit edilerek cihaz yönetiminin (`.to(device)`) çağıran taraf olan `train.py` sorumluluğunda kalması sağlandı.
- `train.py` içerisinde `from src.model import get_crop_disease_model` ifadesi ile modüler import gerçekleştirildi.
- PyTorch en iyi uygulamaları (best practices) uyarınca, optimizer tanımının doğru cihaz parametreleri üzerinden yapılması için `model.to(device)` işlemi optimizer başlangıç atamasının öncesine çekildi.
- Yapılan değişikliklerin doğruluğu kod incelemesi ile kontrol edildi ve `python -m src.train` komutu proje kökünden çalıştırılarak eğitim döngüsünün Gün 6 baseline'ı ile birebir aynı şekilde hatasız çalıştığı doğrulandı.

**Öğrenilenler:**
- `python -m` parametresi kullanılarak bir modülün çalıştırılmasının, Python yorumlayıcısının proje kök dizinini `sys.path` listesine eklemesini sağladığı öğrenildi. Bu sayede modüller içindeki scriptlerin birbirini `src.model` veya `src.dataset` gibi mutlak yollarla sorunsuz import edebildiği kavrandı.
- Paket içi bağımlılıkların çözümlenmesi esnasında `ModuleNotFoundError` hatalarının önüne geçmek için modülleri script şeklinde doğrudan çalıştırmak yerine, `-m` bayrağı ile çalıştırmanın en güvenli yöntem olduğu deneyimlendi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- train.py'ye argparse, çok-epoch döngüsü ve torch.save checkpoint mantığı ekleme; ilk tam eğitim koşusunu başlatma.

## Gün 8 - 01-07

**Görev:**
`train.py`'ye `argparse`, çok-epoch döngüsü ve `torch.save` checkpoint mantığı ekleme; ilk tam eğitim koşusunu başlatma.

## Gün 9 - 02-07

**Görev:**
Tam eğitimi sürdürme; batch size'ı 4GB VRAM sınırına göre ayarlama (`nvidia-smi` ile OOM kontrolü).

## Gün 10 - 03-07

**Görev:**
Train/val loss ve accuracy loglama; matplotlib veya CSV ile eğri çizimi, aşırı öğrenme belirtilerini izleme.

--- Hafta Sonu: 04, 05-07

## Gün 11 - 06-07

**Görev:**
En iyi validation accuracy checkpoint'ini kaydetme; plato noktasında eğitimi durdurma.

## Gün 12 - 07-07

**Görev:**
`src/evaluate.py`: test set accuracy, sınıf bazlı precision/recall/F1 hesaplama.

## Gün 13 - 08-07

**Görev:**
Confusion matrix üretme; karışan sınıfları ve olası nedenlerini kısa paragraf olarak yazma.

## Gün 14 - 09-07

**Görev:**
PlantDoc test split'inde modeli fine-tune etmeden değerlendirme; baseline accuracy kaydı.

## Gün 15 - 10-07

**Görev:**
PlantDoc veri setini indirme/hazırlama; `src/finetune_plantdoc.py` iskeletini oluşturma.

--- Hafta Sonu: 11, 12-07

## Gün 16 - 13-07

**Görev:**
PlantDoc üzerinde 5–10 epoch fine-tune; validation loss izleyerek erken durdurma.

## Gün 17 - 14-07

**Görev:**
Fine-tune sonrası PlantDoc test setini yeniden değerlendirme; önce/sonra metriklerini `results/plantdoc_before_after.json`'a yazma.

--- Atlanan Gün: 15-07

## Gün 18 - 16-07

**Görev:**
README yeniden yazımı; Phase 4–5 sonuçlarını tek `results/` bölümünde birleştirme.

## Gün 19 - 17-07

**Görev:**
Ölü kod ve prob notebook'larını temizleme veya `notebooks/experiments/` altına arşivleme.

--- Hafta Sonu: 18, 19-07

## Gün 20 - 20-07

**Görev:**
Staj-I sunum/rapor hazırlığı; PLAN.md çıkış koşullarını gözden geçirme — yeni teknik iş planlanmaz.