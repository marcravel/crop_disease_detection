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

## Gün 6