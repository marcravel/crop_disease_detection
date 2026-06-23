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

**Planned:** Set up environment, project structure, acquire dataset
**Done:** Repo initialized, structure created (data/, src/, notebooks/, checkpoints/, results/). PLAN.md written. Fixed CUDA driver issue (driver installed but not loaded — modprobe fix). Confirmed GPU working (GTX 1050 Ti, 4GB VRAM, CUDA 11.8). Evaluated multiple PlantVillage Kaggle versions, chose emmarex/plantdisease (15 classes: Tomato, Potato, Pepper) over full 38-class version — faster iteration given 20-day timeline. Downloaded, extracted, verified folder structure matches ImageFolder requirements.
**Blockers:** CUDA driver not loading after install — resolved same day. Dataset version ambiguity — resolved by choosing 15-class subset, documented reasoning.
**Tomorrow:** Write dataset.py (ImageFolder loader, transforms, train/val/test split), validate by loading one batch.

---

## Gün 1 — 22 Haziran 2026

**Planlanan:** Ortam kurulumu, proje yapısı, veri seti edinimi
**Yapılan:** Repo oluşturuldu, klasör yapısı kuruldu (data/, src/, notebooks/, checkpoints/, results/). PLAN.md yazıldı. CUDA driver sorunu çözüldü (driver yüklüydü ama aktif değildi — modprobe ile düzeltildi). GPU doğrulandı (GTX 1050 Ti, 4GB VRAM, CUDA 11.8). Birden fazla PlantVillage versiyonu değerlendirildi, emmarex/plantdisease seçildi (15 sınıf: Tomato, Potato, Pepper) — 20 günlük süre için tam 38 sınıflı versiyondan daha hızlı. Veri seti indirildi, klasör yapısı ImageFolder ile uyumlu doğrulandı.
**Engeller:** CUDA driver yüklenmiyordu — aynı gün çözüldü. Veri seti versiyon belirsizliği — 15 sınıflı alt küme seçilerek çözüldü.
**Yarın:** dataset.py yazılacak (ImageFolder loader, transformlar, train/val/test split), bir batch yükleyerek doğrulama yapılacak.
