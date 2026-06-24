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

**Öğrenilen:** 
- Google Cloud API geliştirici kotalarının tüketici aboneliklerinden (Google One) tamamen bağımsız çalıştığı, faturalandırma katmanına (Tier 1) geçişin ücretsiz kullanım hakkını iptal ettiği ve kota aşımlarının HTTP 429/503 erişim engelleriyle sonuçlandığı deneyimlendi. Geliştirme süreçlerindeki bu ağ darboğazlarını aşmak için Groq/Llama3 gibi sağlayıcıdan bağımsız, yedekli LLM altyapılarının (fallback) Continue gibi araçlara entegre edilmesinin önemi kavrandı. Yazılım mimarisi tarafında ise, veri seti bölütleme (random_split) esnasında veri sızıntısını (data leakage) engellemek ve eğitim süreçlerinde matematiksel tekrarlanabilirliği (reproducibility) garanti altına almak için torch.Generator().manual_seed() mekanizması ile bellek durumunun (state) nasıl sabitlendiği öğrenildi.
- torch.utils.data.random_split ile veri seti %80 eğitim, %10 doğrulama, %10 test olarak sabit uzunluklarda bölünür; torch.Generator().manual_seed() kullanarak bu bölünmenin her çalıştırmada aynı rastgele indeksleri üretmesi sağlanır ve böylece reproducible (tekrarlanabilir) veri seti oluşturulur.
- DataLoader uzunluğu, ilgili Subset’in örnek sayısının batch_size’a bölünmesiyle elde edilen (yukarı yuvarlanmış) tam sayıdır; bu yüzden yalnızca eğitim DataLoader’ı shuffle=True ile karıştırılır, doğrulama ve test loader’ları sabit sıralamada kalır.
- Subset nesneleri, orijinal ImageFolder veri setine bir referans tutar; dolayısıyla aynı transformasyonlar ve sınıf‑etiket haritalaması paylaşılır, ancak her DataLoader yalnızca kendisine atanmış indeksleri kullanır.
- random_split’e sağlanan torch.Generator’ın manual_seed ile ayarlanması, rastgele indeks seçim sürecini deterministik bir kaynağa bağlar; aynı seed tekrar kullanıldığında veri setinin aynı eğitim/validasyon/test bölmeleri yeniden üretilir.