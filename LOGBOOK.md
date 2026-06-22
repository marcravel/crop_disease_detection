# Project Logbook | Proje Günlüğü
/// <summary>
/// Handles daily logging activities that are compiled and finalized at the end of the period.
/// Rather than creating polished reports each day (which wastes time and requires rewriting),
/// this system collects raw daily logs and compiles them into a final report once the complete
/// story is known.
/// </summary>
/// <remarks>
/// Günlük güncellemeleri toplar ve dönemin sonunda derlenir.
/// Her gün cilalı raporlar oluşturmak yerine (bu zaman kaybı ve yeniden yazma gerektirir),
/// bu sistem ham günlük kayıtları toplar ve tam hikaye bilindiğinde bunları son raporta derler.
/// </remarks>

## Day 1 — June 22, 2026

**Planned:** Set up project environment, define structure, and acquire dataset
**Done:** Initialized project repository and created a structured project layout (`data/`, `src/`, `notebooks/`, `configs/`). Created `plan.md` outlining the 20-day internship roadmap. Downloaded and extracted the PlantVillage/PlantDisease dataset (emmarex version). Inspected dataset structure and confirmed it contains 15 classes across 3 crops (Tomato, Potato, Pepper), organized in a format compatible with PyTorch `ImageFolder`. Reviewed multiple Kaggle PlantVillage variants and validated current dataset suitability for a controlled classification experiment under compute constraints.
**Blockers:** Initial ambiguity regarding dataset selection (full PlantVillage vs reduced subset). Resolved by proceeding with the current dataset for a focused 15-class experimental setup aligned with hardware and timeline constraints.
**Tomorrow:** Clean and validate dataset paths, check image integrity, implement PyTorch `ImageFolder` loader, and run a first batch iteration test to confirm pipeline stability.

---

## Gün 1 — 22 Haziran 2026

**Planlanan:** Proje ortamını kurmak, klasör yapısını oluşturmak ve veri setini edinmek
**Yapılan:** Proje repository’si oluşturuldu ve düzenli bir proje klasör yapısı kuruldu (`data/`, `src/`, `notebooks/`, `configs/`). 20 günlük staj planını içeren `plan.md` dosyası hazırlandı. PlantVillage/PlantDisease veri seti (emmarex versiyonu) indirildi ve çıkarıldı. Veri seti yapısı incelendi ve 3 bitki türü (Tomato, Potato, Pepper) altında 15 sınıftan oluştuğu doğrulandı. Yapı PyTorch `ImageFolder` kullanımı için uygun bulundu. Birden fazla Kaggle PlantVillage versiyonu analiz edildi ve mevcut veri seti, donanım ve zaman kısıtları altında kontrollü bir deney için uygun olarak değerlendirildi.
**Engeller:** Veri seti seçimi konusunda başlangıçta belirsizlik (tam PlantVillage vs indirgenmiş sürüm). Bu durum mevcut 15 sınıflı veri seti ile devam edilerek çözüldü.
**Yarın:** Veri seti yollarının temizlenmesi ve doğrulanması, görüntü bütünlüğünün kontrol edilmesi, PyTorch `ImageFolder` veri yükleyicisinin kurulması ve ilk batch testinin çalıştırılması.

