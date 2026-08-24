# T.C. SELÇUK ÜNİVERSİTESİ
## TEKNOLOJİ FAKÜLTESİ
### ELEKTRİK-ELEKTRONİK MÜHENDİSLİĞİ BÖLÜMÜ
#### STAJ-II (YAZILIM GELİŞTİRME VE MODEL DAĞITIMI) UYGULAMALI ÇALIŞMA VE STAJ RAPORU

**Öğrenci Adı Soyadı:** MARC ANJANIAINA RAVELONTSALAMA  
**Staj Türü:** Staj-II (Yazılım Geliştirme, Tam Yığın Web ve Yapay Zeka Model Dağıtımı)  
**Staj Başlangıç - Bitiş Tarihi:** 21 Temmuz 2026 – 17 Ağustos 2026 (20 İş Günü)  
**Proje Adı:** FastAPI ve Next.js Tabanlı ONNX Runtime Destekli Bitki Hastalık Teşhis Web Platformu  
**Yazılım Mimarisi:** FastAPI (Python 3.12 Backend), Next.js 14 / TypeScript / Tailwind CSS (Frontend), ONNX Runtime, Docker & Docker Compose  

---

# 1. GİRİŞ VE PROJE AMACI

Staj-II çalışmasının temel amacı; Staj-I kapsamında eğitilen ve ONNX formatına dönüştürülen derin öğrenme modelinin (`checkpoints/crop_disease_model.onnx`), üretim ortamlarında (production) yüksek hızlı ve düşük gecikmeli bir web uygulaması olarak son kullanıcılara (çiftçilere ve ziraat mühendislerine) sunulmasıdır.

Proje süresince modüler bir **Monorepo** mimarisi benimsenmiş; arka yüz (backend) servisi olarak **FastAPI** ve **ONNX Runtime**, ön yüz (frontend) kullanıcı arayüzü olarak **Next.js 14 (App Router)**, **TypeScript** ve **Tailwind CSS** teknolojileri kullanılmıştır. Sistem; tekli görsel analizi, çoklu yaprak (batch) analizi, milisaniye bazlı gecikme ölçümü, sekmeli tarımsal tedavi rehberi, yerel geçmiş takibi (`localStorage`), birim entegrasyon testleri (`pytest`) ve **Docker Compose** konteynerleştirme bileşenlerini kapsamaktadır.

---

# 2. GÜNLÜK ÇALIŞMA VE TEKNİK FAALİYET RAPORU (DAYS 1 – 20)

---

## Gün 1 — 21-07-2026: Monorepo Mimarisi, Dizin Hiyerarşisi ve Dağıtım Stratejisinin Kurulması

**Problem ve Mühendislik Kısıtları:**  
Derin öğrenme modellerinin üretim ortamlarında yayınlanmasında en yaygın mimari hata, kullanıcı arayüzü (frontend) ile yapay zeka servis kütüphanelerini aynı devasa bağımlılık havuzunda toplamak veya iki ayrı bağımsız Git deposuna (polyrepo) bölmektir. Polyrepo yaklaşımı, API kontratı (schema) güncellendiğinde her iki depoda ayrı ayrı commit atılmasını gerektirir ve sürümler arası uyumsuzluklara yol açar. Projenin tek bir Git reposu altında, ancak bağımsız alt modüller halinde yönetildiği **Monorepo** mimarisi kurulmalıdır. Staj-I çıktısı olan `checkpoints/crop_disease_model.onnx` yapay zeka varlığı (model artifact) doğrudan arka yüz servisine bağlanmalıdır.

**Alternatif Analizi ve Seçim Gerekçesi:**  
1. **Polyrepo (Ayrı Depolar):** Frontend ve backend iki ayrı depoda tutulur. Ancak API veri modellerinde yapılan bir değişiklik her iki depoda manuel senkronizasyon gerektirir ve Docker Compose konteyner orkestrasyonunu zorlaştırır.
2. **Monorepo (Tek Depo, Modüler Klasörler):** Ön yüz, arka yüz ve model varlıkları tek bir kök dizinde (`crop-disease-detector/`) toplanır. Sürüm takibi tek bir commit noktası üzerinden sağlanır. Docker ortamı tek bir root `docker-compose.yml` ile yönetilebilir. Bu esneklik ve sürdürülebilirlik nedeniyle Monorepo mimarisi tercih edilmiştir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
Proje kök dizininde FastAPI servislerini barındıracak `backend/` ve Next.js kullanıcı arayüzünü barındıracak `frontend/` klasörleri oluşturulmuştur. Staj-II çalışma takvimi 21 Temmuz – 17 Ağustos 2026 tarihleri arasında 20 iş günü olacak şekilde planlanmıştır. `checkpoints/crop_disease_model.onnx` dosyası backend servisinin erişebileceği ortak konuma kopyalanmıştır.

[EKRAN GÖRÜNTÜSÜ: monorepo_structure — backend/ ve frontend/ monorepo dizin mimarisi]

```
crop-disease-detector/
├── checkpoints/
│   └── crop_disease_model.onnx   # Staj-I ONNX Model Artifact
├── backend/                      # Python FastAPI Backend
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── schemas/
│   │   └── services/
│   └── tests/
└── frontend/                     # Next.js / TypeScript Frontend
    └── src/
        ├── app/
        ├── components/
        └── services/
```

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Monorepo dizin mimarisi başarıyla kurulmuş, ön yüz ve arka yüz bağımsızlığı korunarak tek merkezden sürdürülebilir kod versiyonlaması sağlanmıştır. Modüler yapı sayesinde gelecekte mobil uygulama veya yeni mikroservislerin eklenmesi kolaylaşmıştır.

---

## Gün 2 — 22-07-2026: FastAPI Bağımlılıklarının Tanımlanması, Yapılandırma Modülü (`config.py`) ve CORS Yapılandırması

**Problem ve Mühendislik Kısıtları:**  
Web servislerinde dosya yollarının, API versiyon numaralarının ve ortam değişkenlerinin kod içerisinde sabit (hardcoded) olarak yazılması, uygulamanın farklı sunucularda veya Docker konteynerlerinde çalışmasını engeller. Ayrıca web tarayıcılarının güvenlik politikaları gereği, varsayılan olarak farklı portlarda çalışan ön yüz (Port 3000) ile arka yüz (Port 8000) arasındaki HTTP istekleri engellenir (Same-Origin Policy). Güvenli Çapraz Orijin Kaynak Paylaşımı (CORS) politikalarının doğru yapılandırılması zorunludur.

**Alternatif Analizi ve Seçim Gerekçesi:**  
Python tarafında web çerçevesi olarak **Flask**, **Django REST Framework** ve **FastAPI** değerlendirilmiştir:
- **Flask:** Çok hafif olmasına karşın asenkron (async/await) desteğinin olmaması ve Pydantic veri doğrulama entegrasyonunun eksikliği nedeniyle elenmiştir.
- **Django:** Monolitik yapısı ve ağır veritabanı bağımlılıkları nedeniyle mikroservis yapımıza uygun bulunmamıştır.
- **FastAPI:** Asenkron Asynchronous Server Gateway Interface (ASGI - `uvicorn`) üzerine kurulu olması, saniyede binlerce isteği işleyebilmesi, varsayılan Pydantic ve OpenAPI (Swagger UI) desteği sunması nedeniyle oy birliğiyle seçilmiştir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`backend/requirements.txt` dosyası oluşturularak bağımlılıklar tanımlanmıştır. Ortam değişkenlerini ve konfigürasyonu merkezi olarak yöneten `backend/app/config.py` modülü yazılmıştır. `backend/main.py` içerisinde FastAPI ana uygulaması başlatılmış ve `CORSMiddleware` yapılandırılmıştır.

[EKRAN GÖRÜNTÜSÜ: backend/app/config.py — Settings sınıfı, model yolları ve CORS yapılandırması]

[EKRAN GÖRÜNTÜSÜ: backend/main.py — FastAPI uygulama başlangıcı ve CORSMiddleware ekleme bloğu]

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

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Çapraz orijin istek izinleri verilerek frontend-backend iletişimi güvenli hale getirilmiş, merkezi konfigürasyon altyapısı tamamlanmıştır.

---

## Gün 3 — 23-07-2026: ONNX Runtime Çıkarım Servisi Mimarisi ve Yürütme Sağlayıcısı Yönetimi (`onnx_service.py`)

**Problem ve Mühendislik Kısıtları:**  
PyTorch modellerinin üretim web sunucularında doğrudan `.pth` olarak çalıştırılması, sunucuya birkaç gigabaytlık PyTorch, CUDA ve Torchvision kütüphanelerinin yüklenmesini gerektirir. Bu durum bellek kullanımını artırır ve sunucu başlatma süresini uzatır. C++ optimizasyonlu **ONNX Runtime** motoru kullanılarak çıkarım süresi milisaniyelere düşürülmeli ve PyTorch kütüphane bağımlılığı backend servisinden tamamen kaldırılmalıdır. Donanımda GPU mevcutsa `CUDAExecutionProvider`, aksi halde `CPUExecutionProvider` moduna dinamik geçiş sağlanmalıdır.

**Alternatif Analizi ve Seçim Gerekçesi:**  
1. **PyTorch Native Inference (`torch.load`):** Ağır kütüphane boyutu (2GB+) ve yüksek RAM tüketimi nedeniyle elenmiştir.
2. **TorchScript C++ LibTorch:** Performanslı olmasına karşın Python tarafında kurulum ve bindings karmaşıklığı yüksektir.
3. **ONNX Runtime:** C++ altyapısı ile çok hızlıdır, hafiftir ve CUDA/CPU yürütme sağlayıcılarını (Execution Providers) otomatik yönetir. Bu sebeplerle seçilmiştir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`backend/app/services/onnx_service.py` modülü altında `ONNXInferenceService` sınıfı geliştirilmiştir. `onnxruntime.InferenceSession` nesnesi başlatılarak model tensör girdi (`input`) ve çıktı (`output`) isimleri sorgulanmıştır.

[EKRAN GÖRÜNTÜSÜ: backend/app/services/onnx_service.py — ONNXInferenceService sınıfı ve session yükleme metodu]

```python
# ONNX Runtime oturumu başlatma ve dinamik provider yönetimi
import onnxruntime as ort

class ONNXInferenceService:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.session = None
        self.load_model()

    def load_model(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
```

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
ONNX Runtime oturumu başarıyla başlatılmış, PyTorch kütüphanesine ihtiyaç duymadan C++ seviyesinde yüksek hızlı çıkarım altyapısı kurulmuştur.

---

## Gün 4 — 24-07-2026: İstemci İmaj Ön İşleme Boru Hattı ve Nümerik Kararlı Softmax Hesaplaması

**Problem ve Mühendislik Kısıtları:**  
İstemciden ham ikili bayt (bytes) olarak gelen görsellerin ONNX modelinin beklediği $[1, 3, 224, 224]$ boyutlu normalize edilmiş float32 matris biçimine getirilmesi gerekir. Görsellerin RGB renk kanallarının Staj-I'de hesaplanan ImageNet ortalama (`[0.485, 0.456, 0.406]`) ve standart sapma (`[0.229, 0.224, 0.225]`) değerleriyle piksel seviyesinde normalize edilmesi şarttır. Ayrıca modelin ürettiği ham logit çıktılarını %0-100 arası olasılığa dönüştürürken büyük sayılarda sayısal taşmaları (overflow) önleyen kararlı bir Softmax fonksiyonu yazılmalıdır.

**Alternatif Analizi ve Seçim Gerekçesi:**  
Görsel işlemeyi OpenCV veya Pillow (PIL) ile yapma seçenekleri incelenmiştir. OpenCV görselleri varsayılan olarak BGR formatında açtığı için RGB dönüşümünün unutulması ciddi tahmin hatalarına yol açmaktadır. Pillow doğrudan RGB formatında çalıştığı ve bellek dostu olduğu için tercih edilmiştir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`onnx_service.py` içerisinde `preprocess_image` ve `softmax` fonksiyonları yazılmıştır. Ham görsel Pillow ile açılmış, RGB formata çevrilmiş, `224x224` boyutuna getirilmiş, ImageNet ortalama ve standart sapma değerleriyle normalize edilerek NCHW matris biçimine dönüştürülmüştür.

[EKRAN GÖRÜNTÜSÜ: backend/app/services/onnx_service.py — preprocess_image ve softmax yardımcı fonksiyonları]

```python
# ImageNet normalizasyonu ve NCHW matris dönüşümü
def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(settings.IMAGE_SIZE)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - np.array(settings.MEAN)) / np.array(settings.STD)
    img_np = np.transpose(img_np, (2, 0, 1))
    return np.expand_dims(img_np, axis=0)

@staticmethod
def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
```

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
İstemci tarafı görsellerinin ONNX tensör formatına eksiksiz dönüşümü sağlanmış, kararlı Softmax ile güven skorları elde edilmiştir.

---

## Gün 5 — 27-07-2026: Tekli Tahmin REST Endpoint'i ve Pydantic Doğrulama Şemaları (`POST /api/v1/predict`)

**Problem ve Mühendislik Kısıtları:**  
İstemciden gelen HTTP multipart/form-data görsel isteklerini kabul eden, veri tiplerini ve dosya formatlarını (JPG/PNG) doğrulayan ve yapılandırılmış JSON yanıtı dönen RESTful endpoint'in geliştirilmesi gerekmektedir. İstemciye tahmin sonucunun yanı sıra milisaniye cinsinden çıkarım süresinin de (`latency_ms`) sunulması hedeflenmiştir.

**Alternatif Analizi ve Seçim Gerekçesi:**  
Görselleri Base64 metni olarak JSON payload içerisinde göndermek veya `UploadFile` (multipart/form-data) kullanmak seçenekleri değerlendirilmiştir. Base64 kodlaması veri boyutunu %33 oranında büyüterek ağ gecikmesini artırdığı için doğrudan ikili (binary) multipart aktarımı tercih edilmiştir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`backend/app/schemas/predict.py` modülünde `SinglePredictionResponse` ve `PredictionItem` Pydantic şemaları tanımlanmıştır. `backend/app/api/v1/endpoints/predict.py` dosyasına `POST /api/v1/predict` endpoint'i eklenmiştir.

[EKRAN GÖRÜNTÜSÜ: backend/app/schemas/predict.py — SinglePredictionResponse ve PredictionItem Pydantic modelleri]

[EKRAN GÖRÜNTÜSÜ: backend/app/api/v1/endpoints/predict.py — POST /predict endpoint fonksiyonu]

```python
# POST /api/v1/predict endpoint mantığı
@router.post("/predict", response_model=SinglePredictionResponse)
async def predict_single_image(file: UploadFile = File(...), top_k: int = Query(3)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Geçerli bir görsel dosyası seçiniz.")
    contents = await file.read()
    result = onnx_service.predict(image_bytes=contents, filename=file.filename, top_k=top_k)
    return result
```

[GÖRSEL: fastapi_swagger_docs.png — FastAPI OpenAPI Swagger UI etkileşimli dokümantasyon ekranı]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Pydantic entegrasyonu ile otomatik istek doğrulaması sağlanmış, Swagger UI dokümantasyonu üzerinden tekli imaj tahminlerinin ~15 ms gecikmeyle çalıştığı teyit edilmiştir.

---

## Gün 6 — 28-07-2026: 15-Sınıflı Tarımsal Bilgi Bankası Modülü (`disease_db.py`)

**Problem ve Mühendislik Kısıtları:**  
Yapay zeka modelinin yalnızca İngilizce sınıf adı (ör. `Potato___Early_blight`) döndürmesi son kullanıcı (çiftçi/ziraat mühendisi) için yeterli değildir. Her hastalık sınıfı için semptomlar, organik tedavi yöntemleri, kimyasal ilaç önerileri ve koruyucu tedbirleri içeren uzman bir tarımsal veritabanı kurulmalıdır.

**Alternatif Analizi ve Seçim Gerekçesi:**  
İlişkisel veritabanı (PostgreSQL/SQLite) kullanmak veya Python içi in-memory dictionary kullanmak seçenekleri değerlendirilmiştir. 15 sınıfın veri boyutunun çok küçük olması ve veritabanı I/O sorgu gecikmesini sıfıra indirmek amacıyla bellek içi (in-memory) Python sözlük yapısı tercih edilmiştir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`backend/app/services/disease_db.py` modülü yazılarak 15 sınıfın tamamını kapsayan `DISEASE_KNOWLEDGE_BASE` veri yapısı ve `get_disease_info` erişim fonksiyonu geliştirilmiştir.

[EKRAN GÖRÜNTÜSÜ: backend/app/services/disease_db.py — DISEASE_KNOWLEDGE_BASE veri yapısı ve get_disease_info fonksiyonu]

```python
# Örnek Patates Erken Yanıklığı tarımsal reçete verisi
"Potato___Early_blight": {
    "disease_id": "POT_EB",
    "name_tr": "Patates Erken Yanıklığı",
    "name_en": "Potato Early Blight",
    "crop_type": "Patates",
    "is_healthy": False,
    "severity": "Orta - Yüksek",
    "description": "Alternaria solani mantarının yol açtığı, yapraklarda hedef tahtası benzeri halkalı lekeler oluşturan hastalık.",
    "symptoms": ["Yapraklarda konsantrik halkalı kahverengi lekeler", "Alt yapraklarda sararma ve kuruma"],
    "organic_treatment": ["Bakır sülfat (Bordo bulamacı) uygulaması", "Enfekte yaprakların derhal budanması"],
    "chemical_treatment": ["Mancozeb veya Chlorothalonil etken maddeli fungusitler"],
    "prevention": ["Ekim nöbeti (münavebe) uygulanması", "Damlama sulama tercih edilmesi"]
}
```

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Teşhis sonuçlarına uzman ziraat reçetelerinin otomatik eklenmesiyle uygulamanın pratik değeri artırılmıştır.

---

## Gün 7 — 29-07-2026: Toplu Tahmin, Sistem Durum Kontrolü ve Hastalık Rehberi Endpoint'leri

**Amaç ve Yapılan Çalışmalar:**  
Sistem mimarisini tamamlamak üzere ek RESTful endpoint'ler geliştirilmiştir:
- `GET /api/v1/health`: Modelin yüklenme durumunu ve aktif yürütme cihazını (CUDA/CPU) döndürür.
- `GET /api/v1/disease`: 15 sınıfın tamamını ve detaylarını listeler.
- `POST /api/v1/predict-batch`: Birden fazla yaprak fotoğrafının tek istekte paralel analiz edilmesini sağlar.

[EKRAN GÖRÜNTÜSÜ: backend/app/api/v1/endpoints/health.py — GET /health endpoint fonksiyonu]

[EKRAN GÖRÜNTÜSÜ: backend/app/api/v1/endpoints/predict.py — POST /predict-batch toplu tahmin endpoint fonksiyonu]

```python
# GET /health endpoint mantığı
@router.get("/health", response_model=HealthCheckResponse)
def health_check():
    model_loaded = onnx_service.session is not None
    providers = onnx_service.session.get_providers() if model_loaded else []
    return HealthCheckResponse(
        status="healthy", version=settings.VERSION,
        model_loaded=model_loaded, device=providers[0] if providers else "None"
    )
```

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Toplu analiz servisi ve sistem durum izleme endpoint'leri `api_router` altında başarıyla birleştirilmiştir.

---

## Gün 8 — 30-07-2026: Pytest Otomatik API Entegrasyon Test Paketinin Yazılması (`test_predict_api.py`)

**Problem ve Mühendislik Kısıtları:**  
Backend servislerinin sürdürülebilirliği ve refaktör süreçlerinde kırılmaların önlenmesi için otomatik entegrasyon testlerinin (integration tests) yazılması gerekmektedir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`backend/tests/test_predict_api.py` modülü yazılmıştır. `fastapi.testclient.TestClient` kullanılarak kök endpoint (`/`), sağlık kontrolü (`/health`), hastalık listeleme (`/disease`) ve görsel çıkarımı (`/predict`) otomatik test edilmiştir.

[EKRAN GÖRÜNTÜSÜ: backend/tests/test_predict_api.py — pytest API test senaryoları]

[GÖRSEL: pytest_terminal_output.png — Pytest test çalıştırılması ve 4/4 passed terminal ekranı]

```bash
# Pytest entegrasyon testlerinin koşturulması
PYTHONPATH=. pytest backend/tests/test_predict_api.py
```

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Tüm testler hatasız geçerek **4/4 passed in 0.88s** sonucu elde edilmiş, backend servislerinin kararlılığı kanıtlanmıştır.

---

## Gün 9 — 31-07-2026: Frontend Next.js 14, TypeScript ve Tailwind CSS Proje Yapılandırması

**Problem ve Mühendislik Kısıtları:**  
Kullanıcıların mobil ve masaüstü cihazlardan rahatça erişebileceği, hızlı, arama motoru dostu (SEO) ve tip güvenli bir web arayüzünün kurulması gerekmektedir.

**Alternatif Analizi ve Seçim Gerekçesi:**  
Create React App (Vite) ile Next.js karşılaştırılmıştır. Vite yalnızca istemci taraflı render (CSR) yaparken, Next.js 14 App Router sunucu taraflı ön-render (SSR/SSG), entegre yönlendirme ve optimizasyon imkanları sunduğu için tercih edilmiştir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`frontend/` klasörü altında Next.js 14 (App Router), TypeScript, Tailwind CSS, Lucide-React ikon kütüphanesi ve Axios kurulumları gerçekleştirilmiştir (`package.json`, `tsconfig.json`, `tailwind.config.js`, `postcss.config.js`).

[EKRAN GÖRÜNTÜSÜ: frontend/package.json — Bağımlılıklar, bağımlılık sürümleri ve derleme betikleri]

[EKRAN GÖRÜNTÜSÜ: frontend/tailwind.config.js — Özel renk paleti ve tema genişletmeleri]

```bash
# Frontend üretim derleme testi
cd frontend && npm run build
```

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Next.js App Router yapısı kurulmuş, derleme testi yapılarak sorunsuz çalıştığı doğrulanmıştır.

---

## Gün 10 — 03-08-2026: Tasarım Sistemi (`globals.css`), Navbar ve Footer Bileşenleri

**Problem ve Mühendislik Kısıtları:**  
Uygulamanın kullanıcı deneyimini (UX) artırmak için modern, şık ve göz yormayan bir tasarım sistemine (Design System) ihtiyaç vardır. Ayrıca canlı backend durumunu gösteren dinamik bir navigasyon çubuğu tasarlanmalıdır.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`frontend/src/app/globals.css` içerisinde cam efekti (glassmorphism: `.glass-card`), koyu tema renk paleti (`#0b1329`) ve zümrüt yeşili parlama efektleri tanımlanmıştır. `Navbar.tsx` bileşeninde canlı ONNX backend durum göstergesi (Active/CPU/CUDA Badge) ve `Footer.tsx` bileşeni yazılmıştır.

[EKRAN GÖRÜNTÜSÜ: frontend/src/app/globals.css — Cam efekti (.glass-card) ve özel kaydırma çubuğu stilleri]

[EKRAN GÖRÜNTÜSÜ: frontend/src/components/Navbar.tsx — Canlı ONNX backend durum göstergesi bileşeni]

[GÖRSEL: navbar_footer_preview.png — Navbar ve Footer bileşenlerinin arayüz önizlemesi]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Cam efektli modern UI bileşenleri geliştirilmiş, canlı backend bağlantı rozeti entegre edilmiştir.

---

## Gün 11 — 04-08-2026: Sürükle-Bırak İmaj Yükleme Bileşeni (`ImageUploader.tsx`)

**Problem ve Mühendislik Kısıtları:**  
Kullanıcıların yaprak fotoğraflarını kolayca yükleyebilmeleri için sürükle-bırak (Drag and Drop) alanına ihtiyaç vardır. Geçersiz dosya türlerinin (PDF, TXT) veya aşırı büyük dosyaların (>15MB) sunucuya gönderilmeden önce istemci tarafında engellenmesi gerekmektedir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`frontend/src/components/ImageUploader.tsx` geliştirilmiştir. HTML5 Drag and Drop API event'leri (`onDragOver`, `onDrop`) işlenmiş; sürükleme esnasında yeşil kenarlık parlama efekti verilmiştir. İstemci tarafında `URL.createObjectURL` ile anlık görsel önizleme sağlanmıştır.

[EKRAN GÖRÜNTÜSÜ: frontend/src/components/ImageUploader.tsx — Sürükle-bırak event handler'ları ve dosya doğrulama mantığı]

[GÖRSEL: ImageUploader.tsx — Sürükle-bırak arayüzü ve istemci görsel önizleme ekran görüntüsü]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
İstemci tarafı doğrulama kuralları kurulmuş, bellek dostu önizleme yapısıyla kullanıcı deneyimi yükseltilmiştir.

---

## Gün 12 — 05-08-2026: Tip Güvenli Axios API Servis Katmanı (`apiService.ts`)

**Problem ve Mühendislik Kısıtları:**  
Ön yüzün arka yüzle doğrudan `fetch` üzerinden ham veri alması, tip uyumsuzluklarına (type errors) ve kopyalanmış koda yol açar. Tüm HTTP isteklerini tip güvenli bir servis katmanında toplamak gerekir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`frontend/src/services/apiService.ts` modülü yazılarak Axios istemcisi yapılandırılmıştır. `getHealthStatus`, `predictSingleImage`, `predictBatchImages` ve `getDiseaseDetail` fonksiyonları geliştirilmiştir.

[EKRAN GÖRÜNTÜSÜ: frontend/src/services/apiService.ts — Axios istemci konfigürasyonu ve tip güvenli istek fonksiyonları]

```typescript
// Tip güvenli Axios tahmin servisi
export const predictSingleImage = async (file: File, topK: number = 3): Promise<SinglePredictionResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await apiClient.post<SinglePredictionResponse>(`/predict?top_k=${topK}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};
```

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
TypeScript arayüzleri ile tam uyumlu, tip güvenli API haberleşme katmanı tamamlanmıştır.

---

## Gün 13 — 06-08-2026: İnteraktif Tahmin Sonuç Kartı Bileşeni (`PredictionResult.tsx`)

**Problem ve Mühendislik Kısıtları:**  
Modelden dönen tahmin sonuçlarının (Top-K olasılıklar, durum etiketi, gecikme süresi) kullanıcıya anlaşılır ve görsel grafiklerle sunulması gerekmektedir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`frontend/src/components/PredictionResult.tsx` yazılmıştır. Teşhis edilen hastalık için renk kodlu durum rozeti (Sağlıklı / Hastalıklı), model güven yüzdesi göstergesi (%99.2), milisaniye bazlı çıkarım süresi (`14.8 ms`) ve Top-3 olasılık dağılım çubukları (progress bars) eklenmiştir.

[EKRAN GÖRÜNTÜSÜ: frontend/src/components/PredictionResult.tsx — Top-K olasılık çubukları ve güven rozeti render fonksiyonu]

[GÖRSEL: PredictionResult.tsx — Top-K olasılık dağılımı ve milisaniye bazlı çıkarım süresi ekranı]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Olasılık dağılımlarının görsel grafiklerle sunulması model kararlarının şeffaflığına (Explainable AI) katkı sağlamıştır.

---

## Gün 14 — 07-08-2026: Tarımsal Tedavi Rehberi Paneli (`DiseaseDetailCard.tsx`)

**Problem ve Mühendislik Kısıtları:**  
Teşhis sonucuna ait semptom, organik tedavi, kimyasal ilaç ve önleme verilerinin tek sayfada karmaşaya yol açmadan sekmeli (Tabbed) yapıda gösterilmesi gerekmektedir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`frontend/src/components/DiseaseDetailCard.tsx` geliştirilmiştir. React `useState` kullanılarak "Belirtiler", "Organik Tedavi", "Kimyasal İlaçlar" ve "Önleyici Tedbirler" sekmeleri tasarlanmıştır.

[EKRAN GÖRÜNTÜSÜ: frontend/src/components/DiseaseDetailCard.tsx — Sekmeli arayüz ve tarımsal reçete render bloğu]

[GÖRSEL: DiseaseDetailCard.tsx — Sekmeli tarımsal tedavi rehberi ekran görüntüsü]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Tarımsal reçetelerin sekmeli yapıda düzenlenmesiyle kullanıcı arayüz ergonomisi artırılmıştır.

---

## Gün 15 — 10-08-2026: İstemci Geçmiş Kaydedici ve Geçmiş Sayfası (`HistoryTable.tsx`, `/history`)

**Problem ve Mühendislik Kısıtları:**  
Çiftçilerin daha önce yaptıkları analizleri tekrar inceleyebilmeleri için sunucu tarafında veritabanı karmaşıklığı yaratmadan yerel bir kayıt mekanizması kurulmalıdır.

**Uygulanan Yöntem ve Teknik Detaylar:**  
Başarılı her teşhisi istemci tarayıcısının `localStorage` alanına kaydeden yapı kurulmuştur. `HistoryTable.tsx` bileşeni ve `/history` sayfası geliştirilmiştir.

[EKRAN GÖRÜNTÜSÜ: frontend/src/components/HistoryTable.tsx — localStorage veri okuma/yazma ve geçmiş tablosu render bileşeni]

[GÖRSEL: HistoryTable.tsx — localStorage geçmiş analiz tablosu ekranı]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Tarayıcı `localStorage` API kullanımı ile kalıcı ve performanslı yerel geçmiş takibi sağlanmıştır.

---

## Gün 16 — 11-08-2026: Çoklu Yaprak Analizi Portalı (`/batch`)

**Problem ve Mühendislik Kısıtları:**  
Toplu yaprak fotoğraflarının tek tek yüklenmesi zaman alacağından, birden fazla görselin aynı anda seçilip backend `/predict-batch` servisine gönderilmesi gerekmektedir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`frontend/src/app/batch/page.tsx` sayfası geliştirilmiştir. Çoklu dosya seçimi, toplu analiz isteği ve gelen sonuçların ızgara (grid) kartlar halinde listelenmesi sağlanmıştır.

[EKRAN GÖRÜNTÜSÜ: frontend/src/app/batch/page.tsx — Çoklu dosya seçimi ve toplu analiz istek yönetimi]

[GÖRSEL: batch_page — Çoklu yaprak analiz portalı ekran görüntüsü]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Toplu analiz servisi başarıyla entegre edilmiştir.

---

## Gün 17 — 12-08-2026: Model Performans ve Saha Adaptasyon Metrikleri Sayfası (`/metrics`)

**Problem ve Mühendislik Kısıtları:**  
Modelin laboratuvar ve saha başarımlarının, alan kayması (Domain Shift) analizlerinin ve karmaşıklık matrislerinin kullanıcıya şeffafça sunulacağı özel bir metrik sayfasına ihtiyaç vardır.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`frontend/src/app/metrics/page.tsx` geliştirilmiştir. KPI kartları, 15 sınıflı metrik tablosu ve görseller (`confusion_matrix.png`, `learning_curves.png`) eklenmiştir.

[EKRAN GÖRÜNTÜSÜ: frontend/src/app/metrics/page.tsx — Metrik KPI kartları ve 15 sınıflı tablo render bileşeni]

[GÖRSEL: metrics_page — Model metrikleri ve saha adaptasyon analizi ekranı]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Saha çoğullamalı modelin %26.47 sıfır-vuruş başarımı ve %96.13 PlantVillage doğruluğu şeffafça sunulmuştur.

---

## Gün 18 — 13-08-2026: Uçtan Uca Entegrasyon Testleri ve Gecikme Doğrulaması

**Problem ve Mühendislik Kısıtları:**  
Ön yüzden başlayan, FastAPI üzerinden geçip ONNX Runtime çıkarımı ile sonuçlanan tüm ağ döngüsünün gecikme süresinin <50ms olduğu teyit edilmelidir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
Tarayıcı geliştirici araçları (Network Tab) ve backend logları üzerinden E2E testler yapılmıştır.

[EKRAN GÖRÜNTÜSÜ: e2e_network_tab.png — Tarayıcı geliştirici konsolunda /predict isteği yanıt süresi ve payload ekranı]

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Çıkarım süresinin ortalama 14.8 ms olduğu teyit edilmiş, kesintisiz veri akışı sağlanmıştır.

---

## Gün 19 — 14-08-2026: Docker ve Docker Compose Konteynerleştirme Mimarisi

**Problem ve Mühendislik Kısıtları:**  
Uygulamanın farklı sunucu ortamlarında bağımlılık hatası olmaksızın tek komutla ayağa kaldırılabilmesi için Docker ile konteynerleştirilmesi gerekmektedir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
`backend/Dockerfile` (Python 3.12-slim tabanlı), `frontend/Dockerfile` (Node 20-alpine çok aşamalı derleme tabanlı) ve root `docker-compose.yml` yazılmıştır.

[EKRAN GÖRÜNTÜSÜ: backend/Dockerfile — Python 3.12-slim tabanlı backend konteyner yapılandırması]

[EKRAN GÖRÜNTÜSÜ: frontend/Dockerfile — Node 20-alpine çok aşamalı derleme Dockerfile]

[EKRAN GÖRÜNTÜSÜ: docker-compose.yml — Backend me Frontend servislerinin Docker Compose orkestrasyonu]

```yaml
# docker-compose.yml orkestrasyon dosyası
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

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
`docker compose up --build` komutu ile tüm sistemin sorunsuz konteynerize çalıştığı doğrulanmıştır.

---

## Gün 20 — 17-08-2026: Uçtan Uca Sistem Denetimi, LOGBOOK_STAJ2.md Doğrulanması, GitHub Repozituvar Senkronizasyonu ve Staj-II Resmi Kapanışı

**Problem ve Mühendislik Kısıtları:**  
Staj-II süresince geliştirilen FastAPI backend servisleri, ONNX çıkarım motoru, Next.js frontend arayüzü, Docker konteynerleştirme bileşenleri ve otomatik test paketlerinin üretim ortamlarında hatasız ve tam performansla çalıştığının doğrulanması; günlük staj seyir defterinin (`LOGBOOK_STAJ2.md`) eksiksiz denetlenmesi ve tüm yazılım varlıklarının uzaktan Git repozituvarına (`https://github.com/marcravel/crop_disease_detection.git`) pushed edilmesi gerekmektedir.

**Uygulanan Yöntem ve Teknik Detaylar:**  
Stajın son gününde, projeye ait tüm bileşenler sırasıyla uçtan uca denetlenmiştir. 
1. `backend/tests/test_predict_api.py` entegrasyon test paketi tekrar çalıştırılarak `4/4 passed` (%100 başarı) teyit edilmiştir.
2. Next.js istemci derlemesi `npm run build` ile koşturulmuş, tüm dinamik ve statik sayfaların (`/`, `/batch`, `/history`, `/metrics`) sıfır tür/lint hatasıyla derlendiği doğrulanmıştır.
3. `LOGBOOK_STAJ2.md` dosyasındaki 20 günlük teknik kayıtlar denetlenmiş ve Staj-II çıkış koşulları kontrol edilmiştir.
4. Git çalışma ağacı temizlenmiş, tüm yeni kaynak kodlar ve dokümanlar `origin/main` dalına đẩy edilmiştir.

[EKRAN GÖRÜNTÜSÜ: terminal — git status ve git push origin main onay ekranı]

[EKRAN GÖRÜNTÜSÜ: LOGBOOK_STAJ2.md — 20 günlük staj seyir defteri ve tamamlanan görevler kontrol listesi]

```bash
# Son doğrulama ve Git senkronizasyonu
PYTHONPATH=. pytest backend/tests/test_predict_api.py
cd frontend && npm run build
git status
git push origin main
```

**Elde Edilen Sonuçlar ve Mühendislik Çıkarımları:**  
Staj-II kapsamındaki tüm yazılım geliştirme, API tasarımı, web ön yüz entegrasyonu, ONNX model dağıtımı ve konteynerleştirme hedefleri eksiksiz olarak başarıyla tamamlanmış; projenin üretim ortamında yayına hazır olduğu teyit edilerek Staj-II resmi olarak kapatılmıştır.

---

# 3. SONUÇ VE DEĞERLENDİRME

Staj-II kapsamında, Staj-I'de eğitilen derin öğrenme modeli üretime hazır tam yığın bir web platformuna dönüştürülmüştür. Elde edilen temel teknik kazanımlar şunlardır:

1. **Yüksek Hızlı ONNX Çıkarımı:** PyTorch bağımlılığı olmadan ONNX Runtime ile **<50 ms** çıkarım süresi elde edilmiştir.
2. **Modern Monorepo Mimarisi:** FastAPI REST API arka yüzü ile Next.js / TypeScript / Tailwind CSS ön yüzü modüler yapıda entegre edilmiştir.
3. **Konteynerleştirme ve Test:** `pytest` ile %100 API test başarımı ve **Docker Compose** ile tek komutla dağıtım altyapısı kurulmuştur.
