# T.C. ÜNİVERSİTESİ MÜHENDİSLİK FAKÜLTESİ
## BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ
### STAJ-II (YAZILIM GELİŞTİRME VE MODEL DAĞITIMI) UYGULAMALI ÇALIŞMA VE STAJ RAPORU

**Öğrenci Adı Soyadı:** Anjaravel / Marc Ravel  
**Staj Türü:** Staj-II (Yazılım Geliştirme, Tam Yığın Web ve Yapay Zeka Model Dağıtımı)  
**Staj Başlangıç - Bitiş Tarihi:** 21 Temmuz 2026 – 17 Ağustos 2026  
**Proje Adı:** FastAPI ve Next.js Tabanlı ONNX Runtime Destekli Bitki Hastalık Teşhis Web Platformu  
**Yazılım Mimarısı:** FastAPI (Python 3.12 Backend), Next.js 14 / TypeScript / Tailwind CSS (Frontend), ONNX Runtime, Docker & Docker Compose  

---

# 1. GİRİŞ VE PROJE AMACI

Staj-II çalışmasının temel amacı; Staj-I kapsamında eğitilen ve ONNX formatına dönüştürülen derin öğrenme modelinin (`checkpoints/crop_disease_model.onnx`), üretim ortamlarında (production) yüksek hızlı ve düşük gecikmeli bir web uygulaması olarak son kullanıcılara (çiftçilere ve ziraat mühendislerine) sunulmasıdır.

Proje süresince modüler bir **Monorepo** mimarisi benimsenmiş; arka yüz (backend) servisi olarak **FastAPI** ve **ONNX Runtime**, ön yüz (frontend) kullanıcı arayüzü olarak **Next.js 14 (App Router)**, **TypeScript** ve **Tailwind CSS** teknolojileri kullanılmıştır. Sistem; tekli görsel analizi, çoklu yaprak (batch) analizi, milisaniye bazlı gecikme ölçümü, sekmeli tarımsal tedavi rehberi, yerel geçmiş takibi (`localStorage`), birim testler (`pytest`) ve **Docker Compose** konteynerleştirme bileşenlerini kapsamaktadır.

---

# 2. GÜNLÜK ÇALIŞMA VE TEKNİK FAALİYET RAPORU (DAYS 1 – 20)

---

## Gün 1 — 21-07-2026: Monorepo Mimarisi ve Dizin Yapısının Tasarımı

**Amaç ve Yapılan Çalışmalar:**  
Staj-II projesinin ilk gününde, backend ve frontend katmanlarının aynı depoda fakat bağımsız modüller halinde geliştirilmesini sağlayan Monorepo dizin mimarisi tasarlanmış ve kurulmuştur. Proje kök dizininde FastAPI servislerini barındıran `backend/` ve Next.js kullanıcı arayüzünü barındıran `frontend/` klasörleri oluşturulmuştur. Staj-I ürünleri (`checkpoints/crop_disease_model.onnx`) backend servisine bağlanmıştır.

```
crop-disease-detector/
├── checkpoints/
│   └── crop_disease_model.onnx   # Staj-I ONNX Model Artifact
├── backend/                      # Python FastAPI Backend
│   ├── app/
│   └── tests/
└── frontend/                     # Next.js / TypeScript Frontend
    └── src/
```

**Sonuç ve Çıkarımlar:**  
Monorepo yapısının ön yüz ve arka yüz geliştirme bağımsızlığını korurken kod versiyonlamasını tek merkezde toplama avantajı teyit edilmiştir.

---

## Gün 2 — 22-07-2026: FastAPI Ortam Bağımlılıkları ve Merkezi Yapılandırma Modülü (`config.py`)

**Amaç ve Yapılan Çalışmalar:**  
`backend/requirements.txt` dosyası oluşturularak `fastapi`, `uvicorn`, `onnxruntime`, `pillow`, `pydantic` ve `pytest` bağımlılıkları tanımlanmıştır. Ortam değişkenleri ve model yollarını merkezi olarak yöneten `backend/app/config.py` modülü yazılmıştır. `backend/main.py` içerisinde FastAPI ana uygulaması başlatılmış ve tarayıcı erişimleri için CORS politikaları (`CORSMiddleware`) tanımlanmıştır.

```python
# backend/app/config.py merkezi yapılandırma tanımı
import os

class Settings:
    PROJECT_NAME: str = "Crop Disease Detector API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    MODEL_PATH: str = os.getenv("MODEL_PATH", "checkpoints/crop_disease_model.onnx")
    MEAN: list = [0.485, 0.456, 0.406]
    STD: list = [0.229, 0.224, 0.225]
    IMAGE_SIZE: tuple = (224, 224)
    CORS_ORIGINS: list = ["*"]

settings = Settings()
```

**Sonuç ve Çıkarımlar:**  
Web servislerinde CORS yapılandırmasının önemi anlaşılmış, sabitlerin tek merkezden yönetimi sağlanmıştır.

---

## Gün 3 — 23-07-2026: ONNX Runtime Çıkarım Servisi Mimarisi (`onnx_service.py`)

**Amaç ve Yapılan Çalışmalar:**  
PyTorch bağımlılığı olmadan C++ optimizasyonlu ONNX modellerini belleğe yükleyen ve çıkarım koşturan `backend/app/services/onnx_service.py` modülü geliştirilmiştir. `ONNXInferenceService` sınıfı üzerinden `onnxruntime.InferenceSession` başlatılmış; GPU (CUDAExecutionProvider) ve CPU (CPUExecutionProvider) geçişleri otomatikleştirilmiştir.

```python
# ONNX Runtime oturumu başlatma ve sağlayıcı yönetimi
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
self.session = ort.InferenceSession(self.model_path, providers=providers)
self.input_name = self.session.get_inputs()[0].name
self.output_name = self.session.get_outputs()[0].name
```

**Sonuç ve Çıkarımlar:**  
ONNX Runtime'ın PyTorch'a kıyasla daha hafif bellek ayak izi (memory footprint) bıraktığı ve C++ altyapısı sayesinde daha düşük gecikme sağladığı gözlemlenmiştir.

---

## Gün 4 — 24-07-2026: Görsel Ön İşleme Boru Hattı ve Softmax Hesaplamaları

**Amaç ve Yapılan Çalışmalar:**  
Yüklenen yaprak fotoğraflarını PIL Image ile açan, RGB formata çeviren, `224x224` boyutuna getiren ve ImageNet ortalama/standart sapma değerleriyle normalize eden `preprocess_image` fonksiyonu yazılmıştır. Görsel matrisi $[H, W, C]$ biçiminden ONNX tensör formatı olan $[1, 3, 224, 224]$ ($NCHW$) biçimine dönüştürülmüştür. Logit çıktılarından %0-100 arası olasılık türeten nümerik olarak kararlı `softmax` fonksiyonu eklenmiştir.

```python
# ImageNet normalizasyonu ve NCHW matris dönüşümü
img_np = np.array(img, dtype=np.float32) / 255.0
img_np = (img_np - np.array(settings.MEAN)) / np.array(settings.STD)
img_np = np.transpose(img_np, (2, 0, 1))
img_np = np.expand_dims(img_np, axis=0)
```

**Sonuç ve Çıkarımlar:**  
Matris boyut dönüşümlerinin (transpose/expand_dims) C++ tabanlı ONNX çıkarım motoruyla uyumu sağlanmıştır.

---

## Gün 5 — 27-07-2026: Tekli Tahmin Endpoint'i ve Pydantic Şemaları (`POST /api/v1/predict`)

**Amaç ve Yapılan Çalışmalar:**  
`backend/app/schemas/predict.py` içerisinde `SinglePredictionResponse` ve `PredictionItem` Pydantic şemaları oluşturulmuştur. `backend/app/api/v1/endpoints/predict.py` dosyasına `POST /api/v1/predict` RESTful endpoint'i eklenmiştir. Dosya türü kontrol edilmiş, ONNX çıkarımı koşturulmuş ve en yüksek olasılıklı 3 sınıf (Top-K) ile milisaniye bazlı gecikme süresi (`latency_ms`) istemciye döndürülmüştür.

```python
# POST /api/v1/predict endpoint yapısı
@router.post("/predict", response_model=SinglePredictionResponse)
async def predict_single_image(file: UploadFile = File(...), top_k: int = Query(3)):
    contents = await file.read()
    result = onnx_service.predict(image_bytes=contents, filename=file.filename, top_k=top_k)
    return result
```

**Sonuç ve Çıkarımlar:**  
FastAPI'nin Pydantic entegrasyonu sayesinde otomatik request/response doğrulaması ve OpenAPI (Swagger UI) dokümantasyonu elde edilmiştir.

---

## Gün 6 — 28-07-2026: 15-Sınıflı Tarımsal Bilgi Bankası Modülü (`disease_db.py`)

**Amaç ve Yapılan Çalışmalar:**  
Derin öğrenme tahminlerini son kullanıcı için anlamlı kılmak amacıyla `backend/app/services/disease_db.py` yazılmıştır. 15 sınıfın tamamı için Türkçe ve İngilizce hastalık adları, semptomlar, organik tedavi yöntemleri, kimyasal ilaçlama önerileri ve koruyucu tedbirleri içeren `DISEASE_KNOWLEDGE_BASE` veri yapısı oluşturulmuştur.

```python
# Örnek patates erken yanıklığı tarımsal reçete verisi
"Potato___Early_blight": {
    "name_tr": "Patates Erken Yanıklığı",
    "crop_type": "Potato",
    "is_healthy": False,
    "symptoms": ["Yapraklarda halkalı kahverengi lekeler", "Alt yapraklarda sararma"],
    "organic_treatment": ["Bakır sülfat karışımı", "Hasta yaprakların budanması"],
    "chemical_treatment": ["Mancozeb veya Chlorothalonil etken maddeli fungusitler"]
}
```

**Sonuç ve Çıkarımlar:**  
Tahmin yanıtlarına uzman ziraat reçetelerinin eklenmesiyle uygulamanın tarladaki pratik değeri artırılmıştır.

---

## Gün 7 — 29-07-2026: Toplu Tahmin, Sistem Sağlık Kontrolü ve Hastalık Rehberi Endpoint'leri

**Amaç ve Yapılan Çalışmalar:**  
`backend/app/api/v1/endpoints/health.py`, `disease.py` ve `predict.py` modülleri tamamlanmıştır:
- `GET /api/v1/health`: Modelin yüklenme durumunu ve aktif yürütme cihazını (CUDA/CPU) döndürür.
- `GET /api/v1/disease`: 15 sınıfın tamamını ve detaylarını listeler.
- `POST /api/v1/predict-batch`: Birden fazla yaprak fotoğrafının tek istekte analiz edilmesini sağlar.

**Sonuç ve Çıkarımlar:**  
Toplu analiz servisi ile paralel çıkarım işlevselliği backend katmanına kazandırılmıştır.

---

## Gün 8 — 30-07-2026: Pytest Otomatik API Entegrasyon Test Paketinin Yazılması (`test_predict_api.py`)

**Amaç ve Yapılan Çalışmalar:**  
Backend servislerinin kararlılığını doğrulamak üzere `backend/tests/test_predict_api.py` yazılmıştır. `TestClient` ve `pytest` kullanılarak kök endpoint (`/`), sağlık kontrolü (`/health`), hastalık listeleme (`/disease`) ve görsel çıkarımı (`/predict`) test edilmiştir.

```bash
# Pytest entegrasyon testlerinin çalıştırılması
PYTHONPATH=. pytest backend/tests/test_predict_api.py
```

**Sonuç ve Çıkarımlar:**  
Tüm testler hatasız geçerek **4/4 passed in 0.88s** sonucu elde edilmiş, backend API'sinin kararlılığı kanıtlanmıştır.

---

## Gün 9 — 31-07-2026: Frontend Next.js 14, TypeScript ve Tailwind CSS Proje Yapılandırması

**Amaç ve Yapılan Çalışmalar:**  
`frontend/` klasörü altında Next.js 14 (App Router), TypeScript, Tailwind CSS, Lucide-React ikon kütüphanesi ve Axios kurulumları gerçekleştirilmiştir (`package.json`, `tsconfig.json`, `tailwind.config.js`).

```bash
# Frontend üretim derleme testi
cd frontend && npm run build
```

**Sonuç ve Çıkarımlar:**  
Next.js App Router mimarisinin Server ve Client bileşenleri arasındaki modüler çalışma prensibi kavranmıştır.

---

## Gün 10 — 03-08-2026: Tasarım Sistemi (`globals.css`), Navbar ve Footer Bileşenleri

**Amaç ve Yapılan Çalışmalar:**  
`frontend/src/app/globals.css` içerisinde cam efekti (glassmorphism: `.glass-card`), koyu tema renk paleti (`#0b1329`) ve zümrüt yeşili parlama efektleri tanımlanmıştır. `Navbar.tsx` bileşeninde canlı ONNX backend durum göstergesi (Active/CPU/CUDA Badge) ve `Footer.tsx` bileşeni yazılmıştır.

**Sonuç ve Çıkarımlar:**  
Modern glassmorphism UI tasarım ilkeleriyle kullanıcı odaklı estetik bir arayüz temeli atılmıştır.

---

## Gün 11 — 04-08-2026: Sürükle-Bırak İmaj Yükleme Bileşeni (`ImageUploader.tsx`)

**Amaç ve Yapılan Çalışmalar:**  
`frontend/src/components/ImageUploader.tsx` geliştirilmiştir. İstemci tarafında HTML5 Drag and Drop API ile sürükle-bırak alanı, dosya türü (JPG/PNG) ve boyut (maks 15MB) doğrulaması, anlık görsel önizlemesi ve yükleme durumu (loading spinner) entegre edilmiştir.

**Sonuç ve Çıkarımlar:**  
İstemci tarafında `URL.createObjectURL` kullanımıyla bellek dostu görsel önizleme sağlanmıştır.

---

## Gün 12 — 05-08-2026: Tip Güvenli Axios API Servis Katmanı (`apiService.ts`)

**Amaç ve Yapılan Çalışmalar:**  
`frontend/src/services/apiService.ts` modülü yazılarak FastAPI backend'i ile ön yüz arasında tip güvenli (type-safe) iletişim kurulmuştur. `predictSingleImage`, `predictBatchImages`, `getHealthStatus` ve `getDiseaseDetail` fonksiyonları Axios ile geliştirilmiştir.

```typescript
// İstemci tarafı tip güvenli API çağrısı
export const predictSingleImage = async (file: File, topK: number = 3): Promise<SinglePredictionResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await apiClient.post<SinglePredictionResponse>(`/predict?top_k=${topK}`, formData);
  return response.data;
};
```

**Sonuç ve Çıkarımlar:**  
TypeScript arayüzlerinin (interfaces) kullanımı sayesinde çalışma zamanı hataları (runtime type mismatch) engellenmiştir.

---

## Gün 13 — 06-08-2026: İnteraktif Tahmin Sonuç Kartı Bileşeni (`PredictionResult.tsx`)

**Amaç ve Yapılan Çalışmalar:**  
`frontend/src/components/PredictionResult.tsx` yazılmıştır. Teşhis edilen hastalık için renk kodlu durum rozeti (Sağlıklı / Hastalıklı), model güven yüzdesi göstergesi (%99.2), milisaniye bazlı çıkarım süresi (`14.8 ms`) ve Top-3 olasılık dağılım çubukları (progress bars) eklenmiştir.

**Sonuç ve Çıkarımlar:**  
Olasılık dağılımlarının görsel grafikle sunulması model kararlarının şeffaflığına (Explainable AI) katkı sağlamıştır.

---

## Gün 14 — 07-08-2026: Tarımsal Tedavi Rehberi Paneli (`DiseaseDetailCard.tsx`)

**Amaç ve Yapılan Çalışmalar:**  
`frontend/src/components/DiseaseDetailCard.tsx` geliştirilmiştir. Sekmeli (Tabbed) arayüz ile "Belirtiler", "Organik Tedavi", "Kimyasal İlaçlar" ve "Önleyici Tedbirler" sekmeleri tasarlanmış; ziraat reçeteleri kullanıcıya sunulmuştur.

**Sonuç ve Çıkarımlar:**  
Karmaşık reçete verilerinin sekmeli arayüzle düzenlenmesi mobil ve masaüstü arayüz kullanım kolaylığını artırmıştır.

---

## Gün 15 — 10-08-2026: İstemci Geçmiş Kaydedici ve Geçmiş Sayfası (`HistoryTable.tsx`, `/history`)

**Amaç ve Yapılan Çalışmalar:**  
Gerçekleştirilen her teşhisi istemci tarayıcısının `localStorage` alanına tarih, dosya adı, teşhis ve gecikme süresiyle kaydeden mekanizma kurulmuştur. `HistoryTable.tsx` bileşeni ve `/history` sayfası geliştirilmiştir.

**Sonuç ve Çıkarımlar:**  
Sunucu tarafında veritabanı yükü yaratmadan istemci tarafında kalıcı veri tutma (persistence) sağlanmıştır.

---

## Gün 16 — 11-08-2026: Çoklu Yaprak Analizi Portalı (`/batch`)

**Amaç ve Yapılan Çalışmalar:**  
`frontend/src/app/batch/page.tsx` sayfası geliştirilmiştir. Kullanıcıların tarladan topladıkları birden fazla yaprak fotoğrafını aynı anda yükleyip backend `/predict-batch` servisine göndermesini sağlayan arayüz tasarlanmıştır.

**Sonuç ve Çıkarımlar:**  
Toplu görsel işleme istekleri ızgara (grid) kart düzeninde kullanıcıya sunulmuştur.

---

## Gün 17 — 12-08-2026: Model Performans ve Saha Adaptasyon Metrikleri Sayfası (`/metrics`)

**Amaç ve Yapılan Çalışmalar:**  
Staj-I sonuçlarını ve alan kaymasını (Domain Shift) kullanıcıya aktarmak için `frontend/src/app/metrics/page.tsx` sayfası oluşturulmuştur.
- PlantVillage Doğruluğu: **%96.13**
- Eski Sıfır-Vuruş Saha Başarımı: **%15.69**
- Yeni Çoğullamalı Sıfır-Vuruş Saha Başarımı: **%26.47 (+%10.78 artış)**
- 15 sınıflı Precision, Recall, F1 tablosu ve Karmaşıklık Matrisi görseli eklenmiştir.

`[GÖRSEL: confusion_matrix.png buraya]`

**Sonuç ve Çıkarımlar:**  
Yapay zeka modellerinin laboratuvar ve saha başarımları arasındaki fark kullanıcıya şeffaf bir şekilde sunulmuştur.

---

## Gün 18 — 13-08-2026: Uçtan Uca Entegrasyon Testleri ve Performans Doğrulaması

**Amaç ve Yapılan Çalışmalar:**  
Ön yüzden görsel yükleme ile başlayan, FastAPI üzerinden geçip ONNX Runtime çıkarımı ile sonuçlanan tüm sistem uçtan uca (E2E) test edilmiştir. Ortalama çıkarım süresinin **<50 ms** olduğu teyit edilmiştir.

**Sonuç ve Çıkarımlar:**  
Tüm sistem bileşenlerinin kesintisiz ve kararlı çalıştığı doğrulanmıştır.

---

## Gün 19 — 14-08-2026: Docker ve Docker Compose Konteynerleştirme Mimarisi

**Amaç ve Yapılan Çalışmalar:**  
Uygulamanın her ortamda tek komutla çalışabilmesi için `backend/Dockerfile`, `frontend/Dockerfile` ve root `docker-compose.yml` yazılmıştır.

```yaml
# docker-compose.yml konteyner orkestrasyonu
version: '3.8'
services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
  frontend:
    build:
      context: .
      dockerfile: frontend/Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - backend
```

**Sonuç ve Çıkarımlar:**  
Docker çok aşamalı (multi-stage) derleme teknikleriyle hafif ve güvenli üretim konteynerleri elde edilmiştir.

---

## Gün 20 — 17-08-2026: Staj-II Genel Gözden Geçirme, LOGBOOK_STAJ2.md Denetimi ve Kapanış

**Amaç ve Yapılan Çalışmalar:**  
`LOGBOOK_STAJ2.md` dosyasındaki 20 günlük teknik kayıtlar kontrol edilmiştir. FastAPI backend, ONNX çıkarım motoru ve Next.js ön yüz mimarisinin %100 çalışır durumda olduğu ve GitHub deposuna (`https://github.com/marcravel/crop_disease_detection.git`) pushed edildiği teyit edilmiştir.

**Sonuç ve Çıkarımlar:**  
Staj-II çalışmaları başarıyla tamamlanmıştır.

---

# 3. SONUÇ VE DEĞERLENDİRME

Staj-II kapsamında, Staj-I'de eğitilen derin öğrenme modeli üretime hazır tam yığın bir web platformuna dönüştürülmüştür. Elde edilen temel teknik kazanımlar şunlardır:

1. **Yüksek Hızlı ONNX Çıkarımı:** PyTorch bağımlılığı olmadan ONNX Runtime ile **<50 ms** çıkarım süresi elde edilmiştir.
2. **Modern Monorepo Mimarisi:** FastAPI REST API arka yüzü ile Next.js / TypeScript / Tailwind CSS ön yüzü modüler yapıda entegre edilmiştir.
3. **Konteynerleştirme ve Test:** `pytest` ile %100 API test başarımı ve **Docker Compose** ile tek komutla dağıtım altyapısı kurulmuştur.
