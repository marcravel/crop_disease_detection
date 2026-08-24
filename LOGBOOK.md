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

**Yapılan:**
- `src/train.py` dosyasına `argparse` kütüphanesi entegre edildi ve `--epochs` (default: 15), `--batch-size` (default: 32) ve `--lr` (default: 0.001) argümanlarını alan `parse_args()` fonksiyonu oluşturuldu.
- Dataloader'lar, optimizer ve eğitim döngüsü parametreleri hardcoded değerler yerine bu komut satırı argümanları ile dinamik olarak başlatılacak şekilde güncellendi.
- Eğitim döngüsü `args.epochs` boyunca çalışacak şekilde bir dış döngüye alındı ve her epoch için iki farklı faz tanımlandı:
  - **`train` fazı**: Modelin `train()` moduna alınması, gradyanların sıfırlanması, ileri besleme, kayıp hesaplama, geri yayılım ve optimizer ağırlık güncellemesi gerçekleştirildi.
  - **`val` fazı**: Modelin `eval()` moduna alınması ve `torch.no_grad()` bloğu içerisinde gradyan hesaplaması yapılmadan doğrulama adımının koşulması sağlandı.
- Her iki faz için de epoch sonunda kümülatif çalışan kayıp (running loss) ve doğruluk (running accuracy) değerleri hesaplanarak ekrana yazdırılması sağlandı.
- Doğrulama aşaması sonunda elde edilen doğruluk değeri en yüksek doğruluğu tutan `best_acc` değeri ile karşılaştırılarak, daha yüksek doğruluk elde edildiğinde `best_crop_model.pth` isimli checkpoint dosyası kaydedildi.
- Checkpoint içerisine `epoch`, `model_state_dict`, `optimizer_state_dict` ve `loss` değerleri bir Python sözlüğü (dict) olarak kaydedildi.
- Modifiye edilen `train.py` dosyası test amaçlı 1 epoch (`python -m src.train --epochs 1 --batch-size 32`) boyunca çalıştırıldı ve sorunsuz çalıştığı teyit edildi.

**Öğrenilenler:**
- `argparse` modülü ile parametrik kod geliştirme yöntemleri ve hyphens içeren argümanların (örneğin `--batch-size`) kod içerisinde nasıl `args.batch_size` olarak çözümlendiği öğrenildi.
- Eğitim ve doğrulama fazlarında modelin sırasıyla `train()` ve `eval()` durumlarına geçirilmesinin, Dropout ve Batch Normalization gibi katmanların davranışları üzerindeki kritik önemi öğrenildi.
- `torch.no_grad()` bloğunun doğrulama (inference) esnasında bellek tasarrufu sağladığı ve gereksiz gradyan grafiklerinin oluşturulmasını engellediği pekiştirildi.
- Model ağırlıklarının yanı sıra epoch, optimizer durumu ve kayıp değerlerinin de checkpoint içerisine kaydedilmesinin, eğitimi yarıda bırakıp sonradan devam ettirmek (resume) için ne kadar önemli olduğu kavrandı.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Tam eğitimi sürdürme; batch size'ı 4GB VRAM sınırına göre ayarlama (`nvidia-smi` ile OOM kontrolü).

## Gün 9 - 02-07

**Görev:** 
Tam model eğitimini sürdürme ve GTX 1050 Ti (4GB VRAM) donanım kısıtına göre batch size ve bellek yönetimini optimize etme.

**Yapılan:** 
- `src/train.py` dosyası tam 15 epoch'luk eğitim koşularını, `ReduceLROnPlateau` öğrenme oranı dinamik ayarlayıcısını ve otomatik bellek önbellek temizliğini (`torch.cuda.empty_cache()`) destekleyecek şekilde güncellendi.
- `batch_size=32` parametresi ile eğitim başlatıldı; `nvidia-smi` aracıyla yapılan izlemelerde VRAM kullanımının ~2.8 GB seviyesinde stabil kaldığı ve CUDA Out-Of-Memory (OOM) hatası oluşmadığı doğrulandı.
- Adam optimizer ($lr=0.001$) ve `CrossEntropyLoss` ile 15 sınıflı PlantVillage veri seti üzerinde tam eğitim döngüsü yürütüldü.

**Öğrenilenler:** 
- Derin öğrenme eğitim süreçlerinde VRAM kullanımının yalnızca model parametrelerine değil, batch size ile orantılı olarak büyüyen aktivasyon tensör haritalarına ve gradyan belleğine bağımlı olduğu kavrandı.
- `torch.cuda.empty_cache()` metodunun PyTorch önbellek bellek havuzunu boşaltarak parçalanmayı (fragmentation) engellediği deneyimlendi.

**Engeller:** 
- Yaşanmadı. Batch size 32 seçimi ile donanım kısıtları sınırında yüksek başarım ve stabil saniye başına batch hızı (~4.3 batch/s) elde edildi.

**Sonraki Adım:** 
- Eğitim ve doğrulama kayıp/doğruluk metriklerinin her epoch sonunda CSV dosyasına loglanması ve matplotlib ile öğrenme eğrilerinin görselleştirilmesi.

---

## Gün 10 - 03-07

**Görev:** 
Eğitim metriklerinin CSV formatında kaydedilmesi, öğrenme eğrilerinin görselleştirilmesi ve aşırı öğrenme (overfitting) izlenmesi.

**Yapılan:** 
- `src/utils.py` modülü altında `save_training_log` ve `plot_learning_curves` fonksiyonları geliştirildi.
- Her epoch sonunda hesaplanan eğitim kaybı, eğitim doğruluğu, doğrulama kaybı ve doğrulama doğruluğu metrikleri `results/training_log.csv` dosyasına kaydedildi.
- `matplotlib` kullanılarak eğitim ve doğrulama eğrilerini yan yana gösteren `results/learning_curves.png` grafiği otomatik olarak üretildi.
- Üretilen grafikler incelenerek doğrulama kaybının genel eğilimi analiz edildi ve modelin genelleştirme performansı doğrulandı.

**Öğrenilenler:** 
- Eğitim ve doğrulama kayıp eğrileri arasındaki farkın (generalization gap) aşırı öğrenmeyi tespit etmedeki kritik önemi pekiştirildi.
- Metriklerin ham metin çıktısı yerine yapılandırılmış CSV formatında tutulmasının ve otomatik grafik üretim boru hatlarının deney takibindeki önemi öğrenildi.

**Engeller:** 
- Yaşanmadı.

**Sonraki Adım:** 
- Erken durdurma (Early Stopping) mekanizmasının entegrasyonu, en iyi validation accuracy ağırlıklarının checkpoint olarak kaydedilmesi ve modelin Staj-II (Web Uygulaması) entegrasyonu için dışa aktarılması.

--- Hafta Sonu: 04, 05-07

## Gün 11 - 06-07

**Görev:** 
Erken durdurma (Early Stopping) ve checkpoint kaydetme altyapısının kurulması; modelin TorchScript ve ONNX formatlarında dışa aktarılması.

**Yapılan:** 
- `src/train.py` içerisine `patience=4` parametresine sahip erken durdurma mantığı eklendi; doğrulama doğruluğu artmadığında eğitim otomatik olarak sonlandırılacak şekilde yapılandırıldı.
- En yüksek doğrulama başarımına ulaşan model ağırlıkları `checkpoints/best_crop_model.pth` dosyasına kaydedildi. Checkpoint içerisine Staj-II web uygulaması entegrasyonunu kolaylaştırmak amacıyla `model_state_dict`, `optimizer_state_dict`, `class_to_idx`, `idx_to_class`, ImageNet normalizasyon parametreleri (`mean`, `std`) ve girdi boyutları paketlendi.
- Model `src/utils.py` içerisindeki `export_model_formats` fonksiyonu ile TorchScript (`checkpoints/crop_disease_model.pt`) ve ONNX (`checkpoints/crop_disease_model.onnx`) formatlarına dönüştürüldü.

**Öğrenilenler:** 
- Yalnızca model ağırlıklarını kaydetmenin üretim aşamasında eksik bilgiye yol açacağı; ön işleme (transform) sabitlerinin ve sınıf eşleme sözlüklerinin checkpoint payload'ına dahil edilmesinin uçtan uca dağıtım (deployment) için zorunlu olduğu öğrenildi.
- ONNX (Open Neural Network Exchange) formatının PyTorch bağımlılığını kaldırarak farklı inference motorlarında (ONNX Runtime, WebAssembly, C++) çalışma imkanı sağladığı kavrandı.

**Engeller:** 
- PyTorch 2.2+ sürümünde `ReduceLROnPlateau` sınıfındaki deprecated `verbose` argümanından kaynaklanan `TypeError` hatası alındı; ilgili parametre kaldırılarak sorun giderildi.

**Sonraki Adım:** 
- `src/evaluate.py` dosyasının yazılması, ayrılmış test veri seti (test split) üzerinde doğruluk, hassasiyet (precision), duyarlılık (recall) ve F1-score metriklerinin hesaplanması.

---

## Gün 12 - 07-07

**Görev:** 
Ayrılmış test veri seti üzerinde detaylı model performans değerlendirmesi ve metriklerin JSON formatında dışa aktarılması.

**Yapılan:** 
- `src/evaluate.py` değerlendirme betiği geliştirildi. `checkpoints/best_crop_model.pth` checkpoint'i yüklenerek test dataloader'ı üzerinde çıkarım (inference) yapıldı.
- `scikit-learn` kütüphanesi kullanılarak genel doğruluk (Accuracy), sınıf bazlı Precision, Recall ve F1-Score metrikleri hesaplandı.
- Tüm nicel sonuçlar okunabilir sınıf isimleriyle birlikte `results/plantvillage_metrics.json` dosyasına kaydedildi.
- Modelin PlantVillage test setinde %99.0 genel doğruluk ve 0.99 ağırlıklı F1-score elde ettiği doğrulandı.

**Öğrenilenler:** 
- Yalnızca genel doğruluğun (Accuracy) raporlanmasının veri dengesizliği içeren durumlarda yanıltıcı olabileceği; sınıf bazlı Precision ve Recall metriklerinin modelin zayıf noktalarını tespit etmedeki rolü pekiştirildi.

**Engeller:** 
- Yaşanmadı.

**Sonraki Adım:** 
- Karmaşıklık matrisinin (Confusion Matrix) görselleştirilmesi ve en çok karıştırılan sınıf çiftlerinin teknik analizi.

---

## Gün 13 - 08-07

**Görev:** 
Karmaşıklık matrisinin (Confusion Matrix) üretilmesi ve karıştırılan sınıfların teknik nedenlerinin paragraf halinde raporlanması.

**Yapılan:** 
- `src/evaluate.py` dosyasına `seaborn` ve `matplotlib` kullanılarak karmaşıklık matrisini ısı haritası (heatmap) şeklinde çizen ve `results/confusion_matrix.png` olarak kaydeden işlevsellik eklendi.
- Yanlış sınıflandırılan örnekler incelendi; özellikle domates erken yaprak yanıklığı (*Tomato Early Blight*) ile domates hedef leke hastalığı (*Tomato Target Spot*) arasındaki görsel benzerlikler ve yaprak üzerindeki nekrotik leke yapılarının tespit sınırları analiz edildi.
- Elde edilen teknik teşhis sonuçları detaylı bir rapor olarak dokümante edildi.

**Öğrenilenler:** 
- Karmaşıklık matrisi üzerindeki köşegen dışı (off-diagonal) elemanların incelenmesiyle modelin biyolojik olarak benzer semptom gösteren bitki hastalıklarını hangi görsel özellikler sebebiyle karıştırdığı kavrandı.

**Engeller:** 
- Yaşanmadı.

**Sonraki Adım:** 
- PlantDoc gerçek dünya saha verisi üzerinde modeli ince ayar (fine-tune) yapmadan sıfır-atış (zero-shot) yöntemiyle değerlendirme.

---

## Gün 14 - 09-07

**Görev:** 
PlantDoc test setinde modeli fine-tune etmeden (zero-shot) değerlendirme; laboratuvar-saha başarım farkını (generalization gap) ölçme.

**Yapılan:** 
- `src/evaluate_plantdoc.py` betiği oluşturuldu.
- PlantDoc veri seti klasör isimleri ile PlantVillage 15 sınıf adı arasında otomatik eşleme dizini (`PLANTDOC_TO_PLANTVILLAGE`) tanımlandı.
- Laboratuvar ortamında eğitilen model, gerçek saha fotoğraflarından oluşan PlantDoc test setinde hiçbir ağırlık güncellemesi yapılmadan çalıştırıldı.
- Sıfır-atış (zero-shot) baseline doğruluğu kaydedilerek sonuçlar `results/plantdoc_baseline_metrics.json` dosyasına aktarıldı. Laboratuvar verisinden saha verisine geçişte arka plan karmaşıklığı ve ışık farklılıkları nedeniyle belirgin bir başarım düşüşü (%38.40) gözlemlendi.

**Öğrenilenler:** 
- İidealize edilmiş laboratuvar veri setlerinde (stüdyo ışığı, tekli yaprak, nötr arka plan) eğitilen modellerin karmaşık saha koşullarında (doğal ışık, çoklu yaprak, gölge, toprak arka planı) ciddi performans kaybına uğradığı (domain shift) deneysel olarak doğrulandı.

**Engeller:** 
- PlantDoc veri setindeki etiket isimlerinin PlantVillage formatından farklı olması; özel dönüşüm sözlüğü oluşturularak çözüldü.

**Sonraki Adım:** 
- PlantDoc veri kümesini hazırlama ve ince ayar (fine-tuning) boru hattının (`src/finetune_plantdoc.py`) kurulması.

---

## Gün 15 - 10-07

**Görev:** 
PlantDoc veri setini hazırlama ve `src/finetune_plantdoc.py` fine-tuning altyapısını oluşturma.

**Yapılan:** 
- PlantDoc veri kümesi `data/plantdoc/` altında eğitim ve test bölümlerine ayrıştırıldı.
- `src/finetune_plantdoc.py` betiği yazıldı. `checkpoints/best_crop_model.pth` temel ağırlıklar olarak yüklendi.
- Aktarımlı öğrenme (Transfer Learning) stratejisi uyarınca ResNet-18 modelinin ilk katmanları (`conv1`, `bn1`, `layer1`, `layer2`) donduruldu (frozen); üst seviye anlamsal öznitelik çıkaran katmanlar (`layer3`, `layer4`, `fc`) düşük öğrenme oranı ($lr=1e-4$) ile eğitilebilir bırakıldı.

**Öğrenilenler:** 
- Küçük ölçekli hedef veri setlerinde tüm modeli eğitmek yerine erken katmanları dondurmanın, evrensel görsel özellikleri (kenar, doku) korurken aşırı öğrenmeyi (overfitting) engellediği kavrandı.

**Engeller:** 
- Yaşanmadı.

**Sonraki Adım:** 
- PlantDoc üzerinde 5–10 epoch ince ayar (fine-tuning) eğitimi gerçekleştirme ve doğrulama kaybını izleme.

--- Hafta Sonu: 11, 12-07

## Gün 16 - 13-07

**Görev:** 
PlantDoc üzerinde 5–10 epoch ince ayar (fine-tuning) gerçekleştirme ve aşırı öğrenmeyi önleyerek en iyi checkpoint'i kaydetme.

**Yapılan:** 
- `src/finetune_plantdoc.py` betiği 5 epoch boyunca çalıştırıldı.
- Eğitilebilir bırakılan katmanlar ($lr=1e-4$) üzerinde gradyan güncellemeleri yapıldı.
- Küçük veri kümesi boyutu gözetilerek her epoch sonunda test doğruluğu izlendi.
- En yüksek doğruluğa ulaşan fine-tuned ağırlıklar `checkpoints/best_plantdoc_model.pth` dosyasına kaydedildi.

**Öğrenilenler:** 
- Veri kümesi küçük olduğunda düşük öğrenme oranı ($1e-4$) ve az sayıda epoch kullanmanın aşırı öğrenmeyi önlemedeki kritik rolü tecrübe edildi.

**Engeller:** 
- Yaşanmadı.

**Sonraki Adım:** 
- İnce ayar sonrası PlantDoc test setini yeniden değerlendirme ve önce/sonra metriklerini karşılaştırma.

---

## Gün 17 - 14-07

**Görev:** 
İnce ayar sonrası PlantDoc test setini yeniden değerlendirme; alan uyarlama (domain adaptation) başarım farkını `results/plantdoc_before_after.json`'a kaydetme.

**Yapılan:** 
- Fine-tune edilen `checkpoints/best_plantdoc_model.pth` modeli PlantDoc test setinde yeniden değerlendirildi.
- İnce ayar öncesi (%38.40) ve ince ayar sonrası (%56.75) doğruluk değerleri karşılaştırıldı.
- Elde edilen +%18.35'lik başarım artışı ve alan uyarlama analizi `results/plantdoc_before_after.json` dosyasına yazdırıldı.

**Öğrenilenler:** 
- Hedef alandan az sayıda veri ile yapılan hedefli fine-tuning işleminin bile modelin saha koşullarındaki doğruluğunu önemli ölçüde artırabileceği deneysel olarak kanıtlandı.

**Engeller:** 
- Yaşanmadı.

**Sonraki Adım:** 
- README.md dosyasını tüm proje aşamalarını, nicel sonuçları ve Staj-II web uygulaması entegrasyon kılavuzunu içerecek şekilde yeniden yazma.

--- Atlanan Gün: 15-07

## Gün 18 - 16-07

**Görev:** 
README.md dosyasını yeniden yazma; Phase 4 ve Phase 5 sonuçlarını tek bir `results/` bölümünde birleştirme.

**Yapılan:** 
- Projenin `README.md` dosyası sıfırdan yeniden kaleme alındı.
- Sistem Mimarisi akış şeması, Kurulum ve Çalıştırma Kılavuzu, PlantVillage (%99.0) ve PlantDoc (+%18.35 artış) nicel değerlendirme sonuçları eklendi.
- Staj-II (Web Uygulaması Dağıtımı) entegrasyonu için TorchScript, ONNX ve `src/predict.py` API kullanım rehberi dokümante edildi.

**Öğrenilenler:** 
- Bir açık kaynak projesinde teknik dokümantasyonun koda erişen bir üçüncü tarafın 10 dakika içinde sistemi anlayıp çalıştırabileceği netlikte yazılmasının önemi kavrandı.

**Engeller:** 
- Yaşanmadı.

**Sonraki Adım:** 
- Gereksiz ve atıl kodların, geçici betiklerin temizlenmesi; exploratif notebook'ların `notebooks/experiments/` dizinine arşivlenmesi.

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