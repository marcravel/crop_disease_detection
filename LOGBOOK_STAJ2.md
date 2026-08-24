# Staj-II Günlük Çalışma Günlüğü (LOGBOOK_STAJ2.md)
**Proje Adı:** Crop Disease Detector — Web Uygulaması Geliştirme ve Derin Öğrenme Model Dağıtımı (Staj-II)
**Başlangıç Tarihi:** 21 Temmuz 2026
**Bitiş Tarihi:** 17 Ağustos 2026

---

## Gün 1 — 21-07-2026

**Görev:**
Staj-II proje planlaması, monorepo dizin mimarisinin (`backend/` ve `frontend/`) oluşturulması ve Staj-I çıktılarının entegrasyon stratejisinin belirlenmesi.

**Yapılan:**
- Staj-I kapsamında eğitilen ve dışa aktarılan ONNX modelinin (`checkpoints/crop_disease_model.onnx`) web ortamında dağıtımı için monorepo mimarisi tasarlandı.
- Proje kök dizininde FastAPI servislerini barındıracak `backend/` ve Next.js kullanıcı arayüzünü barındıracak `frontend/` klasör yapıları inşa edildi.
- Staj-II çalışma takvimi 20 iş günü (21 Temmuz – 17 Ağustos 2026) olacak şekilde hafta sonları hariç tutularak planlandı.

**Öğrenilenler:**
- Derin öğrenme modellerinin üretim ortamlarında (production) modüler mimari ile sunulmasının, ön yüz ve arka yüz bağımsızlığını sağladığı pekiştirildi.
- Monorepo yapısının kod versiyonlaması ve CI/CD süreçlerindeki avantajları kavrandı.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Python FastAPI backend ortamının bağımlılıklarının tanımlanması ve konfigürasyon modülünün yazılması.

---

## Gün 2 — 22-07-2026

**Görev:**
Backend ortam bağımlılıklarının (`requirements.txt`) tanımlanması, `config.py` yapılandırma modülünün yazılması ve FastAPI ana uygulamasının başlatılması.

**Yapılan:**
- `backend/requirements.txt` içerisine `fastapi`, `uvicorn`, `onnxruntime`, `pillow`, `numpy`, `pydantic` ve `pytest` kütüphaneleri eklendi.
- `backend/app/config.py` modülü yazılarak ONNX model yolu, ImageNet normalizasyon değerleri (`mean`, `std`) ve varsayılan boyutlar (`224x224`) merkezi hale getirildi.
- `backend/main.py` dosyası oluşturuldu ve tarayıcı erişimi için CORS middleware (`CORSMiddleware`) yapılandırıldı.

**Öğrenilenler:**
- RESTful servislerde CORS (Cross-Origin Resource Sharing) politikalarının güvenli ve doğru yapılandırılmasının önemi öğrenildi.
- Ortam değişkenlerinin (Environment Variables) `os.getenv` ile esnek şekilde yönetimi tecrübe edildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- ONNX Runtime çıkarım motorunu (`onnxruntime.InferenceSession`) başlatan servis modülünün yazılması.

---

## Gün 3 — 23-07-2026

**Görev:**
ONNX Runtime çıkarım servisi (`backend/app/services/onnx_service.py`) mimarisinin kurulması ve modelin belleğe yüklenmesi.

**Yapılan:**
- `backend/app/services/onnx_service.py` modülü altında `ONNXInferenceService` sınıfı tanımlandı.
- `checkpoints/crop_disease_model.onnx` modeli `onnxruntime.InferenceSession` kullanılarak belleğe yüklendi; CUDA GPU desteği kontrol edildi, mevcut olmaması durumunda CPU yürütme sağlayıcısına (CPUExecutionProvider) yumuşak geçiş sağlandı.
- Modelin girdi ve çıktı tensör isimleri (`input`, `output`) dinamik olarak sorgulandı.

**Öğrenilenler:**
- ONNX Runtime'ın PyTorch bağımlılığı olmadan C++ optimizasyonlu tensör hesaplamaları sayesinde yüksek hızlı çıkarım sağladığı kavrandı.
- Yürütme sağlayıcılarının (Execution Providers) dinamik tespiti mekanizması öğrenildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Ön işleme (preprocessing) boru hattının ONNX numpy matris formatlarına uygun şekilde geliştirilmesi.

---

## Gün 4 — 24-07-2026

**Görev:**
Görsel ön işleme boru hattının ve Softmax olasılık hesaplama fonksiyonunun `onnx_service.py` içerisine entegrasyonu.

**Yapılan:**
- Yüklenen yaprak fotoğraflarını RGB formata dönüştüren, `224x224` boyutuna yeniden boyutlandıran ve ImageNet ortalama/standart sapma değerleriyle normalize eden `preprocess_image` fonksiyonu yazıldı.
- Görsel matrisi $[H, W, C]$ formatından PyTorch/ONNX uyumlu $[N, C, H, W]$ ($1, 3, 224, 224$) tensör biçimine dönüştürüldü.
- Çıkarım sonucunda elde edilen logit değerlerini %0-100 arasında olasılıklara dönüştüren nümerik olarak kararlı `softmax` fonksiyonu uygulandı.

**Öğrenilenler:**
- Numpy üzerinde kanal sırası dönüşümlerinin (`np.transpose(..., (2,0,1))`) ve batch boyutu eklemenin (`np.expand_dims`) matematiksel altyapısı pekiştirildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Tekli görüntü çıkarımı için Pydantic şemalarının ve `/api/v1/predict` endpoint'inin yazılması.

--- Hafta Sonu: 25, 26-07-2026

## Gün 5 — 27-07-2026

**Görev:**
Tekli görüntü çıkarımı sunan `/api/v1/predict` RESTful endpoint'inin ve Pydantic veri modellerinin geliştirilmesi.

**Yapılan:**
- `backend/app/schemas/predict.py` içerisinde `PredictionItem`, `DiseaseDetail`, `SinglePredictionResponse` Pydantic modelleri oluşturuldu.
- `backend/app/api/v1/endpoints/predict.py` dosyasına `POST /api/v1/predict` endpoint'i eklendi. Yüklenen dosyanın imaj formatı doğrulandı, ONNX çıkarımı koşturuldu ve en yüksek olasılıklı sınıf ile top-k listesi döndürüldü.
- Çıkarım süresi milisaniye cinsinden (`latency_ms`) ölçülerek yanıta eklendi.

**Öğrenilenler:**
- FastAPI ve Pydantic entegrasyonu sayesinde otomatik request/response veri doğrulama ve OpenAPI (Swagger) dokümantasyonu üretimi deneyimlendi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- 15 bitki hastalığı sınıfı için semptom ve tedavi önerilerini içeren tarımsal bilgi bankasının (`disease_db.py`) oluşturulması.

---

## Gün 6 — 28-07-2026

**Görev:**
15 sınıflı tarımsal veritabanının (`backend/app/services/disease_db.py`) Türkçe ve İngilizce detaylarla yazılması.

**Yapılan:**
- 15 sınıfa ait (Domates, Patates, Biber hastalıkları ve sağlıklı durumlar) detaylı semptomlar, hastalık nedenleri, organik mücadele yöntemleri, kimyasal ilaçlama rehberi ve önleyici tedbirleri içeren `DISEASE_KNOWLEDGE_BASE` sözlüğü oluşturuldu.
- Sınıf adına göre bilgi getiren `get_disease_info` yardımcı fonksiyonu yazıldı.
- Tahmin yanıtlarına tarımsal reçete bilgilerinin otomatik eklenmesi sağlandı.

**Öğrenilenler:**
- Derin öğrenme tahminlerinin son kullanıcı (çiftçi/ziraat mühendisi) için anlamlı kılınmasında uzman alan bilgisinin (domain knowledge) sistemle entegrasyonunun önemi kavrandı.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Toplu görüntü çıkarımı (`POST /api/v1/predict-batch`) ve sistem durum kontrolü (`GET /api/v1/health`) endpoint'lerinin geliştirilmesi.

---

## Gün 7 — 29-07-2026

**Görev:**
Toplu analiz (`/api/v1/predict-batch`), sistem sağlık kontrolü (`/api/v1/health`) ve hastalık listeleme (`/api/v1/disease`) endpoint'lerinin tamamlanması.

**Yapılan:**
- `backend/app/api/v1/endpoints/health.py` betiği ile ONNX modelinin bellekteki durumu ve aktif yürütme cihazını (CUDA/CPU) döndüren endpoint yazıldı.
- `backend/app/api/v1/endpoints/disease.py` ile veritabanındaki 15 sınıfı listeleyen ve detayını sorgulayan endpoint'ler eklendi.
- `POST /api/v1/predict-batch` endpoint'i ile birden fazla yaprak fotoğrafının tek istekte analiz edilmesine imkan tanındı.

**Öğrenilenler:**
- Çoklu dosya yüklemelerinde (multipart/form-data) liste işleme ve kümülatif gecikme süresi hesaplama teknikleri pekiştirildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Backend API servisleri için `pytest` otomasyon test paketinin yazılması.

---

## Gün 8 — 30-07-2026

**Görev:**
Backend otomatik entegrasyon testlerinin (`backend/tests/test_predict_api.py`) `pytest` ve `httpx` ile yazılması ve doğrulanması.

**Yapılan:**
- `backend/tests/test_predict_api.py` test dosyası yazıldı.
- Kök dizin (`/`), sağlık kontrolü (`/api/v1/health`), hastalık listeleme (`/api/v1/disease`) ve örnek görsel üzerinden çıkarım (`/api/v1/predict`) endpoint'leri için test senaryoları tanımlandı.
- `PYTHONPATH=. pytest backend/tests/test_predict_api.py` komutu çalıştırılarak tüm testlerin hatasız geçtiği (%100 başarı) doğrulandı.

**Öğrenilenler:**
- Yazılım geliştirmede Test Odaklı Geliştirme (TDD) ve otomatik API testlerinin kod kalitesine katkısı kavrandı.

**Engeller:**
- Pytest çalıştırmasında modül yolu (`ModuleNotFoundError`) hatası alındı; `PYTHONPATH=.` tanımlaması yapılarak sorun çözüldü.

**Sonraki Adım:**
- Frontend Next.js / TypeScript / Tailwind CSS projesinin yapılandırılması.

---

## Gün 9 — 31-07-2026

**Görev:**
Frontend Next.js (App Router) ve TypeScript projesinin (`frontend/`) başlatılması, Tailwind CSS ve ikon kütüphanelerinin kurulumu.

**Yapılan:**
- `frontend/package.json`, `tsconfig.json`, `tailwind.config.js` ve `postcss.config.js` yapılandırma dosyaları oluşturuldu.
- `next`, `react`, `typescript`, `tailwindcss`, `lucide-react` ve `axios` kütüphaneleri yüklendi (`npm install`).
- Projenin `npm run build` komutu ile sorunsuz derlendiği doğrulandı.

**Öğrenilenler:**
- Next.js App Router mimarisinin sunucu ve istemci bileşenleri (Server/Client Components) arasındaki ayrım ilkeleri kavrandı.

**Engeller:**
- `types/index.ts` dosyasındaki küçük bir TypeScript tip tanımı hatası derleme esnasında yakalandı ve düzeltildi.

**Sonraki Adım:**
- Tema renklerinin, glassmorphism stillerinin ve Navbar/Footer bileşenlerinin tasarımı.

--- Hafta Sonu: 01, 02-08-2026

## Gün 10 — 03-08-2026

**Görev:**
Tasarım sisteminin (`globals.css`) oluşturulması ve ana navigasyon (`Navbar.tsx`, `Footer.tsx`) bileşenlerinin geliştirilmesi.

**Yapılan:**
- `frontend/src/app/globals.css` içerisinde glassmorphism kart stilleri (`.glass-card`), özel kaydırma çubukları ve zümrüt yeşili parlama efektleri (`.glow-emerald`) tanımlandı.
- `Navbar.tsx` bileşeni yazıldı. Kök sayfa, toplu analiz ve geçmiş sayfaları için navigasyon bağlantıları ve canlı backend ONNX durum göstergesi (Status Badge) eklendi.
- `Footer.tsx` bileşeni ile telif ve teknoloji bilgileri eklendi.

**Öğrenilenler:**
- Modern web tasarımında cam efekti (glassmorphism) ve mikro-etkileşimlerin kullanıcı deneyimine (UX) olumlu katkısı deneyimlendi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- İstemci taraflı görsel yükleme ve sürükle-bırak bileşeninin (`ImageUploader.tsx`) geliştirilmesi.

---

## Gün 11 — 04-08-2026

**Görev:**
Sürükle-bırak destekli görsel yükleme bileşeninin (`frontend/src/components/ImageUploader.tsx`) yazılması.

**Yapılan:**
- `ImageUploader.tsx` bileşeni geliştirildi.
- Sürükle-bırak (Drag-and-Drop) alanları için `onDragOver` ve `onDrop` durumları işlendi; sürükleme esnasında zümrüt yeşili kenarlık parlama efekti sağlandı.
- İstemci tarafında dosya formatı (JPG/PNG) ve boyut (maks 15MB) kontrolleri eklendi.
- Yüklenen görselin anlık önizlemesi (preview) ve temizleme düğmesi entegre edildi.

**Öğrenilenler:**
- HTML5 Drag and Drop API kullanımı ve istemci tarafında `URL.createObjectURL` ile bellek dostu görsel önizleme teknikleri öğrenildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Frontend API servis istemcisinin (`apiService.ts`) yazılması ve backend ile iletişim kurulması.

---

## Gün 12 — 05-08-2026

**Görev:**
Frontend API servis katmanının (`frontend/src/services/apiService.ts`) Axios ile tip güvenli olarak geliştirilmesi.

**Yapılan:**
- `frontend/src/services/apiService.ts` modülü oluşturuldu.
- Backend FastAPI servislerine erişim sağlayan `getHealthStatus`, `predictSingleImage`, `predictBatchImages` ve `getDiseaseDetail` fonksiyonları yazıldı.
- `multipart/form-data` veri gönderimi ve hata yakalama mekanizmaları eklendi.

**Öğrenilenler:**
- TypeScript arayüzleri (interfaces) ile API yanıtlarının tip güvenli (type-safe) biçimde işlenmesinin çalışma zamanı hatalarını engellemedeki rolü kavrandı.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- İnteraktif tahmin sonuç kartının (`PredictionResult.tsx`) geliştirilmesi.

---

## Gün 13 — 06-08-2026

**Görev:**
İnteraktif tahmin sonuç kartının (`frontend/src/components/PredictionResult.tsx`) tasarlanması ve geliştirilmesi.

**Yapılan:**
- `PredictionResult.tsx` bileşeni yazıldı.
- Tahmin edilen en olası hastalık sınıfı için durum etiketi (Sağlıklı / Hastalıklı), model güven yüzdesi rozeti ve milisaniye bazlı çıkarım süresi göstergesi eklendi.
- En yüksek olasılıklı 3 sınıf için (Top-K) dinamik doluluk oranına sahip renkli ilerleme çubukları (progress bars) tasarlandı.

**Öğrenilenler:**
- Olasılık dağılımlarının görsel grafikle sunulmasının model kararlarının şeffaflığına (Explainable AI) katkısı kavrandı.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Ziraat tedavi ve koruma tedbirleri panelinin (`DiseaseDetailCard.tsx`) geliştirilmesi.

---

## Gün 14 — 07-08-2026

**Görev:**
Tarımsal tedavi ve koruma rehberi panelinin (`frontend/src/components/DiseaseDetailCard.tsx`) sekmeli yapı ile yazılması.

**Yapılan:**
- `DiseaseDetailCard.tsx` bileşeni oluşturuldu.
- Sekmeli (Tabbed) arayüz tasarımı ile "Belirtiler", "Organik Tedavi", "Kimyasal İlaçlar" ve "Önleyici Tedbirler" sekmeleri eklendi.
- Hastalığın şiddet derecesine göre renk kodlu uyarı kutuları entegre edildi.

**Öğrenilenler:**
- Karmaşık içeriklerin sekmeli arayüzler ile düzenlenmesinin mobil ve masaüstü arayüz ergonomisine katkısı deneyimlendi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Yerel tarayıcı geçmişi kaydedici bileşeninin (`HistoryTable.tsx`) ve geçmiş sayfasının yazılması.

--- Hafta Sonu: 08, 09-08-2026

## Gün 15 — 10-08-2026

**Görev:**
Yerel geçmiş kaydedici bileşeni (`HistoryTable.tsx`) ve geçmiş sayfasının (`frontend/src/app/history/page.tsx`) geliştirilmesi.

**Yapılan:**
- Gerçekleştirilen her başarılı analizi istemci tarayıcısının `localStorage` alanına kaydeden mekanizma kuruldu.
- `HistoryTable.tsx` bileşeni yazılarak geçmiş analizlerin tarihi, dosya adı, teşhis sonucu ve gecikme süresi tablo şeklinde listelendi.
- Geçmiş kayıtları temizleme işlevselliği eklendi.

**Öğrenilenler:**
- Tarayıcı `localStorage` API kullanımı ve istemci tarafında kalıcı veri tutma (persistence) teknikleri öğrenildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Çoklu yaprak analizi sayfasının (`frontend/src/app/batch/page.tsx`) geliştirilmesi.

---

## Gün 16 — 11-08-2026

**Görev:**
Çoklu yaprak fotoğrafı analizi sayfasının (`frontend/src/app/batch/page.tsx`) yazılması.

**Yapılan:**
- `frontend/src/app/batch/page.tsx` sayfası oluşturuldu.
- Kullanıcıların birden fazla görseli tek seferde seçip backend `/predict-batch` endpoint'ine göndermesini sağlayan arayüz tasarlandı.
- Gelen toplu analiz yanıtları ızgara (grid) düzeninde kartlar şeklinde kullanıcıya sunuldu.

**Öğrenilenler:**
- Asenkron toplu veri işleme süreçlerinin kullanıcı arayüzünde yüklenme durumları (loading states) ile yönetimi pekiştirildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Mobil cihaz ve tablet uyumluluğu için duyarlı (responsive) düzen optimizasyonu.

---

## Gün 17 — 12-08-2026

**Görev:**
Mobil arayüz uyumluluğu (Responsive Design) ve erişilebilirlik (Accessibility) optimizasyonlarının yapılması.

**Yapılan:**
- Tailwind CSS kırılma noktaları (`sm`, `md`, `lg`) kullanılarak tüm sayfalar mobil, tablet ve masaüstü ekran boyutlarına tam uyumlu hale getirildi.
- Dokunmatik ekranlar için buton boyutları ve tıklama alanları optimize edildi.
- Sayfa yüklenme performansı ve erişilebilirlik nitelikleri (aria-label, alt etiketleri) gözden geçirildi.

**Öğrenilenler:**
- Mobile-first tasarım yaklaşımının saha ortamında akıllı telefon kullanan çiftçiler için önemi kavrandı.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Uçtan uca sistem entegrasyonu, performans testleri ve gecikme (latency) kıyaslamaları.

---

## Gün 18 — 13-08-2026

**Görev:**
Uçtan uca (E2E) sistem entegrasyon testlerinin yürütülmesi ve çıkarım gecikmesinin (Inference Latency) doğrulanması.

**Yapılan:**
- İstemci tarafından görsel yüklemeden başlayan, FastAPI backend üzerinden geçip ONNX Runtime çıkarımı ile yanıt dönen tüm iş akışı test edildi.
- ONNX modellerinin ortalama <50ms çıkarım süresine ulaştığı teyit edildi.
- Sistem genelinde sıfır hata ile uçtan uca veri akışı doğrulandı.

**Öğrenilenler:**
- Tüm katmanların (Frontend -> API -> ONNX Engine) entegrasyonunda gecikme sürelerinin ölçülmesi ve sistem darboğazlarının tespiti deneyimlendi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Uygulamanın Docker ve Docker Compose ile konteynerleştirilmesi.

---

## Gün 19 — 14-08-2026

**Görev:**
Backend ve Frontend uygulamalarının Dockerfile dosyalarının yazılması ve `docker-compose.yml` ile konteynerleştirilmesi.

**Yapılan:**
- Backend için `backend/Dockerfile` (Python 3.12-slim tabanlı) yazıldı.
- Frontend için `frontend/Dockerfile` (Node 20-alpine çok aşamalı derleme tabanlı) oluşturuldu.
- Proje köküne `docker-compose.yml` eklenerek `docker compose up --build` tek komutu ile tüm sistemin ayağa kalkması sağlandı.

**Öğrenilenler:**
- Docker çok aşamalı (multi-stage) derleme teknikleri ile küçük boyutlu ve güvenli üretim imajları oluşturma yöntemi öğrenildi.

**Engeller:**
- Yaşanmadı.

**Sonraki Adım:**
- Staj-II genel gözden geçirme, LOGBOOK_STAJ2.md denetimi ve Staj-II aşamasının kapatılması.

--- Hafta Sonu: 15, 16-08-2026

## Gün 20 — 17-08-2026

**Görev:**
Staj-II sonuçlarının gözden geçirilmesi, `LOGBOOK_STAJ2.md` dosyasının son kontrolü ve resmi Staj-II raporu hazırlığı.

**Yapılan:**
- `LOGBOOK_STAJ2.md` içerisindeki 20 günlük tüm teknik kayıtlar gözden geçirildi ve eksiksiz olduğu teyit edildi.
- FastAPI backend, ONNX Runtime çıkarım motoru ve Next.js frontend mimarisinin %100 çalışır durumda olduğu doğrulandı.
- Üniversite resmi Staj-II Raporu hazırlığı için tüm mimari şemalar ve kod çıktıları düzenlenerek Staj-II aşaması başarıyla kapatıldı.

**Öğrenilenler:**
- Derin öğrenme model geliştirmeden üretim ortamında web servisi olarak sunmaya kadar olan tam yığın (Full-Stack AI Engine) yazılım süreçleri başarıyla tamamlandı.

**Engeller:**
- Yaşanmadı.
