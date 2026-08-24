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

Staj-II döneminin ilk gününde, Staj-I kapsamında eğittiğim, doğruladığım ve ONNX formatına dönüştürdüğüm derin öğrenme modelini (`checkpoints/crop_disease_model.onnx`) üretim ortamında web tabanlı modern bir platforma dönüştürmek için sistem mimarisini tasarlamaya başladım. Derin öğrenme modellerinin dağıtımında en sık karşılaşılan mimari hata, kullanıcı arayüzü (frontend) ile yapay zeka servis kütüphanelerini aynı devasa bağımlılık havuzunda toplamak veya iki ayrı bağımsız Git deposuna (polyrepo) bölmektir. Polyrepo yaklaşımı, API kontratı (şeması) güncellendiğinde her iki depoda ayrı ayrı senkronizasyon gerektirmekte ve Docker konteyner orkestrasyonunu zorlaştırmaktadır.

Bu mimari ikilemi çözmek adına projenin tek bir Git reposu altında, ancak bağımsız alt modüller halinde yönetildiği **Monorepo** mimarisini kurmaya karar verdim. Monorepo yapısı sayesinde sürüm takibi tek bir commit noktası üzerinden sağlanabilmekte ve tüm sistem kök dizindeki tek bir `docker-compose.yml` orkestrasyon dosyasıyla yönetilebilir hale gelmektedir.

Bu doğrultuda proje kök dizininde FastAPI servislerini barındıracak `backend/` ve Next.js kullanıcı arayüzünü barındıracak `frontend/` klasörlerini oluşturdum. Staj-II çalışma takvimini 21 Temmuz – 17 Ağustos 2026 tarihleri arasında 20 iş günü olacak şekilde planladım. Staj-I çıktısı olan `checkpoints/crop_disease_model.onnx` yapay zeka varlığını backend servisinin doğrudan erişebileceği ortak konuma bağladım.

[EKRAN GÖRÜNTÜSÜ: monorepo_structure — backend/ ve frontend/ monorepo dizin mimarisi]

Monorepo dizin mimarisi başarıyla kurulmuş, ön yüz ve arka yüz bağımsızlığı korunarak tek merkezden sürdürülebilir kod versiyonlaması sağlanmıştır. Modüler yapı sayesinde gelecekte mobil uygulama veya yeni mikroservislerin eklenmesi kolaylaştırılmıştır.

---

## Gün 2 — 22-07-2026: FastAPI Bağımlılıklarının Tanımlanması, Yapılandırma Modülü (`config.py`) ve CORS Yapılandırması

Monorepo dizin mimarisini oluşturduktan sonra, bugün arka yüz (backend) servislerinin temelini oluşturacak FastAPI çatısının yapılandırılması ve güvenlik ayarlarının yapılmasına geçtim. Web servislerinde dosya yollarının, API versiyon numaralarının ve ortam değişkenlerinin kod içerisinde sabit (hardcoded) olarak yazılması, uygulamanın farklı sunucularda veya Docker konteynerlerinde çalışmasını engeller. Ayrıca web tarayıcılarının güvenlik politikaları gereği, varsayılan olarak farklı portlarda çalışan ön yüz (Port 3000) ile arka yüz (Port 8000) arasındaki HTTP istekleri engellenir (Same-Origin Policy).

Python tarafında web çerçevesi olarak Flask, Django REST Framework ve FastAPI alternatiflerini değerlendirdim. Flask'ın asenkron (async/await) desteğinin eksikliği ve Django'nun monolitik hantallığı karşısında; Asynchronous Server Gateway Interface (ASGI - `uvicorn`) üzerine kurulu olması, saniyede binlerce isteği asenkron işleyebilmesi ve varsayılan Pydantic veri doğrulama desteği sunması nedeniyle FastAPI'yi seçtim.

`backend/requirements.txt` dosyasını oluşturarak FastAPI, Uvicorn, Onnxruntime, Pillow, NumPy ve Pydantic bağımlılıklarını tanımladım. Ortam değişkenlerini ve konfigürasyonu merkezi olarak yöneten `backend/app/config.py` modülünü yazdım. `backend/main.py` içerisinde FastAPI ana uygulamasını başlattım ve istemci ile sunucu arasındaki çapraz orijin isteklerini güvenle yönetmek için `CORSMiddleware` yapılandırmasını sisteme entegre ettim.

[EKRAN GÖRÜNTÜSÜ: backend/app/config.py — Settings sınıfı, model yolları ve CORS yapılandırması]

[EKRAN GÖRÜNTÜSÜ: backend/main.py — FastAPI uygulama başlangıcı ve CORSMiddleware ekleme bloğu]

Çapraz orijin istek izinleri verilerek frontend-backend iletişimi güvenli hale getirilmiş, merkezi konfigürasyon altyapısı tamamlanmıştır.

---

## Gün 3 — 23-07-2026: ONNX Runtime Çıkarım Servisi Mimarisi ve Yürütme Sağlayıcısı Yönetimi (`onnx_service.py`)

FastAPI ana yapısını ve CORS ayarlarını kurduktan sonra, bugün yapay zeka modelini PyTorch kütüphane bağımlılığı olmaksızın yüksek hızda çalıştıracak ONNX Runtime çıkarım motorunu geliştirdim. PyTorch modellerini üretim sunucusunda doğrudan `.pth` olarak çalıştırmak, sunucuya birkaç gigabaytlık PyTorch, CUDA ve Torchvision kütüphanelerinin yüklenmesini gerektirir. Bu durum bellek kullanımını artırmakta ve sunucu başlatma süresini uzatmaktadır.

PyTorch bağımlılığı yerine C++ optimizasyonlu **ONNX Runtime** motorunu tercih ettim. ONNX Runtime, donanımda GPU mevcutsa `CUDAExecutionProvider`, aksi halde `CPUExecutionProvider` moduna dinamik geçiş sağlayarak yüksek verimlilik sunmaktadır.

Bu doğrultuda `backend/app/services/onnx_service.py` modülü altında `ONNXInferenceService` sınıfını geliştirdim. `onnxruntime.InferenceSession` nesnesi başlatılarak model tensör girdi (`input`) ve çıktı (`output`) adlarını dinamik sorgulayan bir yapı kurdum. Oturum ilklendirmesinde kullanılabilir execution provider listesini denetleyen ve uygun donanım hızlandırıcısını seçen mantığı entegre ettim.

[EKRAN GÖRÜNTÜSÜ: backend/app/services/onnx_service.py — ONNXInferenceService sınıfı ve session yükleme metodu]

ONNX Runtime oturumu başarıyla başlatılmış, PyTorch kütüphanesine ihtiyaç duymadan C++ seviyesinde yüksek hızlı, hafif ve dinamik donanım destekli çıkarım altyapısı kurulmuştur. Bu sayede web servisinin bellek ayak izi ciddi oranda küçültülmüştür.

---

## Gün 4 — 24-07-2026: İstemci İmaj Ön İşleme Boru Hattı ve Nümerik Kararlı Softmax Hesaplaması

ONNX Runtime oturumunu başarıyla başlattıktan sonra, bugün istemciden gelen ham görsel verilerini modelin beklediği tensör formatına dönüştüren ön işleme ve olasılık hesaplama boru hattını geliştirdim. İstemciden ham ikili bayt (bytes) olarak gelen görsellerin ONNX modelinin beklediği $[1, 3, 224, 224]$ boyutlu normalize edilmiş float32 matris biçimine getirilmesi gerekir. Ayrıca modelin ürettiği ham logit çıktılarını %0-100 arası olasılığa dönüştürürken büyük sayılarda sayısal taşmaları (overflow) önleyen kararlı bir Softmax fonksiyonu yazılmalıdır.

Görsel işlemeyi OpenCV veya Pillow (PIL) ile yapma seçeneklerini inceledim. OpenCV görselleri varsayılan olarak BGR formatında açtığı için RGB dönüşümünün unutulması ciddi renk uzayı ve teşhis hatalarına yol açmaktadır. Pillow doğrudan RGB formatında çalıştığı ve bellek dostu olduğu için tercih edilmiştir.

`onnx_service.py` içerisine `preprocess_image` ve `softmax` fonksiyonlarını ekledim. Ham görsel Pillow ile açılmış, RGB formata çevrilmiş, $224 	imes 224$ boyutuna getirilmiş, Staj-I'de hesaplanan ImageNet ortalama (`[0.485, 0.456, 0.406]`) ve standart sapma (`[0.229, 0.224, 0.225]`) değerleriyle normalize edilerek NCHW matris biçimine dönüştürülmüştür. Sayısal taşmaları önlemek için logitlerden maksimum değerin çıkarıldığı kararlı Softmax algoritması ($rac{e^{x - \max(x)}}{\sum e^{x - \max(x)}}$) uygulanmıştır.

[EKRAN GÖRÜNTÜSÜ: backend/app/services/onnx_service.py — preprocess_image ve softmax yardımcı fonksiyonları]

İstemci tarafı görsellerinin ONNX tensör formatına eksiksiz dönüşümü sağlanmış, kararlı Softmax ile güvenilir olasılık skorları elde edilmiştir.

---

## Gün 5 — 27-07-2026: Tekli Tahmin REST Endpoint'i ve Pydantic Doğrulama Şemaları (`POST /api/v1/predict`)

İmaj ön işleme ve Softmax algoritmalarını tamamladıktan sonra, yeni haftada istemcilerin görsel yükleyip anlık teşhis almasını sağlayan tekli tahmin REST endpoint'ini geliştirmeye odaklandım. İstemciden gelen HTTP multipart/form-data görsel isteklerini kabul eden, veri tiplerini ve dosya formatlarını (JPG/PNG) doğrulayan ve yapılandırılmış JSON yanıtı dönen RESTful endpoint'in geliştirilmesi gerekiyordu.

Görselleri Base64 metni olarak JSON payload içerisinde göndermek veya `UploadFile` (multipart/form-data) kullanmak seçeneklerini değerlendirdim. Base64 kodlaması veri boyutunu %33 oranında büyüterek ağ gecikmesini artırdığı için doğrudan ikili (binary) multipart aktarımı tercih ettim.

`backend/app/schemas/predict.py` modülünde `SinglePredictionResponse` ve `PredictionItem` Pydantic şemalarını tanımladım. `backend/app/api/v1/endpoints/predict.py` dosyasına `POST /api/v1/predict` endpoint'ini ekledim. Endpoint, gelen dosyanın `content-type` başlığını denetlemekte, geçersiz dosyaları HTTP 400 hatasıyla reddetmekte; geçerli görselleri ONNX çıkarım servisine iletip tahmin edilen hastalık sınıfını, güven skorunu ve milisaniye bazlı çıkarım süresini (`latency_ms`) döndürmektedir.

[EKRAN GÖRÜNTÜSÜ: backend/app/schemas/predict.py — SinglePredictionResponse ve PredictionItem Pydantic modelleri]

[EKRAN GÖRÜNTÜSÜ: backend/app/api/v1/endpoints/predict.py — POST /predict endpoint fonksiyonu]

[GÖRSEL: fastapi_swagger_docs.png — FastAPI OpenAPI Swagger UI etkileşimli dokümantasyon ekranı]

Pydantic entegrasyonu ile otomatik istek doğrulaması sağlanmış, Swagger UI dokümantasyonu üzerinden tekli imaj tahminlerinin ~15 ms gecikmeyle çalıştığı teyit edilmiştir.

---

## Gün 6 — 28-07-2026: 15-Sınıflı Tarımsal Bilgi Bankası Modülü (`disease_db.py`)

Tekli tahmin API'sini çalışır hale getirdikten sonra, bugün modelin sadece teknik sınıf adı dönmesinin ötesine geçerek çiftçilere ve ziraat uzmanlarına rehberlik edecek kapsamlı tarımsal bilgi bankası modülünü geliştirdim. Yapay zeka modelinin yalnızca İngilizce sınıf adı (ör. `Potato___Early_blight`) döndürmesi son kullanıcı (çiftçi/ziraat mühendisi) için yeterli değildir. Her hastalık sınıfı için semptomlar, organik tedavi yöntemleri, kimyasal ilaç önerileri ve koruyucu tedbirleri içeren uzman bir tarımsal veritabanı kurulması gerekiyordu.

İlişkisel veritabanı (PostgreSQL/SQLite) kullanmak veya Python içi in-memory dictionary kullanmak seçeneklerini değerlendirdim. 15 sınıfın veri boyutunun çok küçük olması ve veritabanı I/O sorgu gecikmesini sıfıra indirmek amacıyla bellek içi (in-memory) Python sözlük yapısını tercih ettim.

`backend/app/services/disease_db.py` modülü yazılarak 15 sınıfın tamamını kapsayan `DISEASE_KNOWLEDGE_BASE` veri yapısı ve `get_disease_info` erişim fonksiyonu geliştirilmiştir. Her sınıf için Türkçe/İngilizce isimler, hastalık açıklaması, semptom listesi, organik tedavi yöntemleri (ör. Bordo bulamacı), kimyasal fungusitler (ör. Mancozeb) ve kültürel önleyici tedbirler detaylandırılmıştır.

[EKRAN GÖRÜNTÜSÜ: backend/app/services/disease_db.py — DISEASE_KNOWLEDGE_BASE veri yapısı ve get_disease_info fonksiyonu]

Teşhis sonuçlarına uzman ziraat reçetelerinin otomatik eklenmesiyle uygulamanın pratik ve tarımsal değeri önemli ölçüde artırılmıştır.

---

## Gün 7 — 29-07-2026: Toplu Tahmin, Sistem Durum Kontrolü ve Hastalık Rehberi Endpoint'leri

Tarımsal bilgi bankasını kurduktan sonra, bugün backend API mimarisini tamamlamak üzere sistem durum izleme, hastalık listeleme ve toplu görsel analizi endpoint'lerini geliştirdim. Web arayüzünün sistem sağlığını anlık izleyebilmesi, katalogdaki tüm hastalıkları listeleyebilmesi ve tarladan çekilen çok sayıda yaprak fotoğrafının tek seferde işlenebilmesi için ek RESTful servislere ihtiyaç vardı.

Bu doğrultuda 3 ana endpoint geliştirdim:
1. `GET /api/v1/health`: Modelin yüklenme durumunu, API sürümünü ve aktif donanım yürütücüsünü (CUDA/CPU) döndürür.
2. `GET /api/v1/disease`: Bilgi bankasındaki 15 sınıfın tamamını ve tarımsal tedavi detaylarını listeler.
3. `POST /api/v1/predict-batch`: Birden fazla yaprak fotoğrafının tek HTTP isteğinde paralel analiz edilmesini sağlar.

Tüm bu endpoint'ler `backend/app/api/v1/api.py` içerisindeki `api_router` altında modüler olarak toplanmış ve ana FastAPI uygulamasına bağlanmıştır.

[EKRAN GÖRÜNTÜSÜ: backend/app/api/v1/endpoints/health.py — GET /health endpoint fonksiyonu]

[EKRAN GÖRÜNTÜSÜ: backend/app/api/v1/endpoints/predict.py — POST /predict-batch toplu tahmin endpoint fonksiyonu]

Toplu analiz servisi ve sistem durum izleme endpoint'leri başarıyla entegre edilmiş, backend servis omurgası tamamlanmıştır.

---

## Gün 8 — 30-07-2026: Pytest Otomatik API Entegrasyon Test Paketinin Yazılması (`test_predict_api.py`)

Tüm backend RESTful endpoint'lerini tamamladıktan sonra, bugün servislerin kararlılığını doğrulamak ve olası regresyonları önlemek amacıyla otomatik API entegrasyon test paketini geliştirdim. Backend servislerinin sürdürülebilirliği, refaktör süreçlerinde kırılmaların önlenmesi ve sürekli entegrasyon (CI) süreçlerine hazır olunması için otomatik testlerin yazılması kritik bir mühendislik zorunluluğudur.

Pytest çatısı ve `fastapi.testclient.TestClient` kütüphanesini kullanarak `backend/tests/test_predict_api.py` modülünü yazdım. Test paketi içerisinde kök karşılama endpoint'i (`/`), sistem sağlık kontrolü (`/health`), hastalık kataloğu sorgulama (`/disease`) ve tekli görsel tahmini (`/predict`) senaryolarını test eden kapsamlı test fonksiyonları geliştirdim. Geçersiz dosya formatı ve eksik parametre durumlarında doğru HTTP hata kodlarının döndüğü doğrulandı.

**Kullanılan Linux Komutu:**
```bash
# Pytest entegrasyon testlerinin koşturulması
PYTHONPATH=. pytest backend/tests/test_predict_api.py
```

[EKRAN GÖRÜNTÜSÜ: backend/tests/test_predict_api.py — pytest API test senaryoları]

[GÖRSEL: pytest_terminal_output.png — Pytest test çalıştırılması ve 4/4 passed terminal ekranı]

Tüm testler hatasız geçerek **4/4 passed in 0.88s** sonucu elde edilmiş, backend servislerinin kararlılığı ve hata toleransı kanıtlanmıştır.

---

## Gün 9 — 31-07-2026: Frontend Next.js 14, TypeScript ve Tailwind CSS Proje Yapılandırması

Backend servislerini ve test paketini başarıyla tamamladıktan sonra, haftanın son gününde kullanıcıların tarayıcı üzerinden sisteme erişeceği modern web ön yüzünün kurulumuna başladım. Kullanıcıların mobil ve masaüstü cihazlardan rahatça erişebileceği, hızlı, arama motoru dostu (SEO) ve tip güvenli bir web arayüzünün kurulması gerekiyordu.

Create React App (Vite) ile Next.js karşılaştırması yaptım. Vite yalnızca istemci taraflı render (CSR) yaparken, Next.js 14 App Router sunucu taraflı ön-render (SSR/SSG), entegre yönlendirme, görsel optimizasyonu ve modern React Server Components desteği sunduğu için tercih edilmiştir.

`frontend/` klasörü altında Next.js 14 (App Router), TypeScript, Tailwind CSS, modern Lucide-React ikon kütüphanesi ve Axios kurulumlarını gerçekleştirdim (`package.json`, `tsconfig.json`, `tailwind.config.js`, `postcss.config.js`).

**Kullanılan Linux Komutu:**
```bash
# Frontend üretim derleme testi
cd frontend && npm run build
```

[EKRAN GÖRÜNTÜSÜ: frontend/package.json — Bağımlılıklar, bağımlılık sürümleri ve derleme betikleri]

[EKRAN GÖRÜNTÜSÜ: frontend/tailwind.config.js — Özel renk paleti ve tema genişletmeleri]

Next.js App Router yapısı kurulmuş, derleme testi koşturularak ortamın sıfır hata ile üretim paketine derlendiği teyit edilmiştir.

---

## Gün 10 — 03-08-2026: Tasarım Sistemi (`globals.css`), Navbar ve Footer Bileşenleri

Frontend altyapısını kurduktan sonra, yeni haftada kullanıcı deneyimini (UX) üst seviyeye taşıyacak modern tasarım sistemini ve ana navigasyon bileşenlerini geliştirmeye odaklandım. Uygulamanın tarımsal teknoloji kimliğini yansıtan, modern, koyu temalı ve şık bir görsel dil oluşturulması hedeflenmiştir.

`frontend/src/app/globals.css` içerisinde yarı saydam cam efekti (glassmorphism: `.glass-card`), koyu tema arka plan renk paleti (`#0b1329`) ve zümrüt yeşili parlama efektleri tanımlanmıştır. `Navbar.tsx` bileşeninde aktif sayfaları gösteren yönlendirme linklerinin yanı sıra backend `/health` servisini belirli aralıklarla sorgulayarak sunucu durumunu canlı rozetle (Active / CPU / CUDA) gösteren dinamik durum göstergesi geliştirilmiştir. Alt bilgi için telif ve teknoloji yığınını barındıran `Footer.tsx` bileşeni yazılmıştır.

[EKRAN GÖRÜNTÜSÜ: frontend/src/app/globals.css — Cam efekti (.glass-card) ve özel kaydırma çubuğu stilleri]

[EKRAN GÖRÜNTÜSÜ: frontend/src/components/Navbar.tsx — Canlı ONNX backend durum göstergesi bileşeni]

[GÖRSEL: navbar_footer_preview.png — Navbar ve Footer bileşenlerinin arayüz önizlemesi]

Cam efektli modern UI bileşenleri geliştirilmiş, canlı backend bağlantı rozeti entegre edilerek kullanıcı arayüzü temel görsel iskeletine kavuşturulmuştur.

---

## Gün 11 — 04-08-2026: Sürükle-Bırak İmaj Yükleme Bileşeni (`ImageUploader.tsx`)

Tasarım sistemi ve navigasyon bileşenlerini tamamladıktan sonra, bugün kullanıcıların yaprak fotoğraflarını kolayca yükleyebilmelerini sağlayan etkileşimli yükleme bileşenini geliştirdim. Kullanıcıların yaprak fotoğraflarını sürükle-bırak yöntemiyle veya dosya gezgininden kolayca seçebilmeleri gerekiyordu. Ayrıca geçersiz dosya türlerinin (PDF, TXT) veya sunucu belleğini zorlayacak 15MB üzeri aşırı büyük dosyaların sunucuya gönderilmeden önce istemci tarafında engellenmesi şarttı.

`frontend/src/components/ImageUploader.tsx` bileşeni geliştirilmiştir. HTML5 Drag and Drop API event'leri (`onDragOver`, `onDragLeave`, `onDrop`) işlenmiş; sürükleme esnasında kullanıcıya görsel geri bildirim sunan zümrüt yeşili kenarlık parlama efekti verilmiştir. İstemci tarafında dosya türü ve boyut doğrulama kuralları işletilmiş, `URL.createObjectURL` API'si ile seçilen görsel tarayıcı belleğinde anlık olarak önizleme kartında gösterilmiştir.

[EKRAN GÖRÜNTÜSÜ: frontend/src/components/ImageUploader.tsx — Sürükle-bırak event handler'ları ve dosya doğrulama mantığı]

[GÖRSEL: ImageUploader.tsx — Sürükle-bırak arayüzü ve istemci görsel önizleme ekran görüntüsü]

İstemci tarafı doğrulama kuralları kurulmuş, bellek dostu önizleme yapısıyla kullanıcı deneyimi ve arayüz tepkiselliği yükseltilmiştir.

---

## Gün 12 — 05-08-2026: Tip Güvenli Axios API Servis Katmanı (`apiService.ts`)

Görsel yükleme arayüzünü hazırladıktan sonra, bugün ön yüz ile FastAPI backend arasındaki HTTP iletişimini yönetecek tip güvenli API servis katmanını geliştirdim. Ön yüzün arka yüzle doğrudan ham `fetch` üzerinden iletişim kurması tip uyumsuzluklarına (type errors), kopyalanmış kodlara ve merkezi hata yönetimi eksikliğine yol açar. Tüm HTTP isteklerini tip güvenli bir servis katmanında toplamak yazılım sürdürülebilirliği için esastır.

`frontend/src/services/apiService.ts` modülü yazılarak Axios istemcisi yapılandırılmıştır. Backend Pydantic şemalarına tam karşılık gelen TypeScript arayüzleri (`SinglePredictionResponse`, `BatchPredictionResponse`, `DiseaseInfo`, `HealthCheckResponse`) tanımlanmıştır. Modül içerisinde `getHealthStatus`, `predictSingleImage`, `predictBatchImages` ve `getDiseaseDetail` fonksiyonları multipart/form-data desteğiyle yazılmıştır.

[EKRAN GÖRÜNTÜSÜ: frontend/src/services/apiService.ts — Axios istemci konfigürasyonu ve tip güvenli istek fonksiyonları]

TypeScript arayüzleri ile tam uyumlu, merkezi hata yakalama mekanizmasına sahip tip güvenli API haberleşme katmanı başarıyla tamamlanmıştır. Bu katman sayesinde ön yüz bileşenleri backend veri yapılarıyla tam uyumlu hale getirilmiştir.

---

## Gün 13 — 06-08-2026: İnteraktif Tahmin Sonuç Kartı Bileşeni (`PredictionResult.tsx`)

API haberleşme katmanını kurduktan sonra, bugün modelden dönen tahmin sonuçlarını kullanıcıya şık ve anlaşılır grafiklerle sunan sonuç kartı bileşenini geliştirdim. Modelden dönen tahmin çıktılarının (Top-K sınıflar, olasılık dağılımları, teşhis edilen durum, gecikme süresi) son kullanıcıya Açıklanabilir Yapay Zeka (XAI) prensiplerine uygun olarak görselleştirilmesi gerekiyordu.

`frontend/src/components/PredictionResult.tsx` bileşeni yazılmıştır. Teşhis edilen durum için renk kodlu rozet (Yeşil: Sağlıklı / Kırmızı: Hastalıklı), model güven yüzdesi göstergesi (%99.2), milisaniye bazlı çıkarım süresi rozeti (`14.8 ms`) ve Top-3 en olası sınıfı gösteren animasyonlu ilerleme çubukları (progress bars) eklenmiştir.

[EKRAN GÖRÜNTÜSÜ: frontend/src/components/PredictionResult.tsx — Top-K olasılık çubukları ve güven rozeti render fonksiyonu]

[GÖRSEL: PredictionResult.tsx — Top-K olasılık dağılımı ve milisaniye bazlı çıkarım süresi ekranı]

Olasılık dağılımlarının görsel grafiklerle sunulması model kararlarının şeffaflığına katkı sağlamış, kullanıcıların yapay zeka tahminlerine duyduğu güven pekiştirilmiştir.

---

## Gün 14 — 07-08-2026: Tarımsal Tedavi Rehberi Paneli (`DiseaseDetailCard.tsx`)

Tahmin sonuç kartını geliştirdikten sonra, haftanın son gününde teşhis edilen hastalığa ait uzman tedavi ve bakım önerilerini sunan sekmeli rehber panelini geliştirdim. Teşhis sonucuna ait semptom, organik tedavi, kimyasal ilaç ve koruyucu tedbir verilerinin tek sayfada karmaşaya yol açmadan ergonomik bir biçimde sunulması gerekiyordu.

`frontend/src/components/DiseaseDetailCard.tsx` bileşeni geliştirilmiştir. React `useState` kullanılarak "Belirtiler", "Organik Tedavi", "Kimyasal İlaçlar" ve "Önleyici Tedbirler" sekmeleri tasarlanmıştır. Kullanıcı tek tıkla sekmeler arasında geçiş yaparak teşhis edilen hastalığa karşı uygulanacak Bordo bulamacı gibi organik çözümleri veya Mancozeb gibi kimyasal fungusit önerilerini detaylıca inceleyebilmektedir.

[EKRAN GÖRÜNTÜSÜ: frontend/src/components/DiseaseDetailCard.tsx — Sekmeli arayüz ve tarımsal reçete render bloğu]

[GÖRSEL: DiseaseDetailCard.tsx — Sekmeli tarımsal tedavi rehberi ekran görüntüsü]

Tarımsal reçetelerin sekmeli yapıda düzenlenmesiyle kullanıcı arayüz ergonomisi artırılmış ve sahadaki çiftçilere hızlı uygulanabilir uzman rehberlik sağlanmıştır.

---

## Gün 15 — 10-08-2026: İstemci Geçmiş Kaydedici ve Geçmiş Sayfası (`HistoryTable.tsx`, `/history`)

Teşhis ve tedavi panellerini tamamladıktan sonra, yeni haftada kullanıcıların daha önce yaptıkları analizleri tekrar inceleyebilmelerini sağlayan yerel geçmiş takip mekanizmasını geliştirdim. Çiftçilerin daha önce yaptıkları yaprak analizlerini geriye dönük inceleyebilmeleri için sunucu tarafında veritabanı karmaşıklığı yaratmadan tarayıcı tabanlı kalıcı bir çözüm üretilmesi hedeflenmiştir.

Başarılı her teşhisi istemci tarayıcısının `localStorage` alanına görsel önizlemesi, teşhis edilen sınıf, güven skoru ve zaman damgasıyla kaydeden bir yardımcı modül kurulmuştur. `HistoryTable.tsx` bileşeni ve `/history` sayfası geliştirilmiştir. Sayfa üzerinde geçmiş kayıtların tarihe ve güven skoruna göre sıralanabilmesi, hastalık adına göre aranabilmesi ve tek tıkla temizlenebilmesi sağlanmıştır.

[EKRAN GÖRÜNTÜSÜ: frontend/src/components/HistoryTable.tsx — localStorage veri okuma/yazma ve geçmiş tablosu render bileşeni]

[GÖRSEL: HistoryTable.tsx — localStorage geçmiş analiz tablosu ekranı]

Tarayıcı `localStorage` API kullanımı ile sunucuya ek yük bindirmeden kalıcı, gizlilik odaklı ve yüksek performanslı yerel geçmiş takibi sağlanmıştır.

---

## Gün 16 — 11-08-2026: Çoklu Yaprak Analizi Portalı (`/batch`)

Tekil analiz ve geçmiş yönetimini kurduktan sonra, bugün sahada birden fazla yaprak fotoğrafının tek seferde incelenmesini sağlayan çoklu analiz portalını geliştirdim. Toplu yaprak fotoğraflarının tek tek yüklenmesi zaman alacağından, birden fazla görselin aynı anda seçilip backend `/predict-batch` servisine gönderilmesi gerekiyordu.

`frontend/src/app/batch/page.tsx` sayfası geliştirilmiştir. Çoklu dosya seçimi, toplu analiz isteği yönetimi ve gelen sonuçların ızgara (grid) kartlar halinde listelenmesi sağlanmıştır. Sayfa üzerinde toplam incelenen yaprak sayısı, sağlıklı/hastalıklı oranları ve ortalama çıkarım gecikmesini özetleyen üst istatistik kartları yerleştirilmiştir.

[EKRAN GÖRÜNTÜSÜ: frontend/src/app/batch/page.tsx — Çoklu dosya seçimi ve toplu analiz istek yönetimi]

[GÖRSEL: batch_page — Çoklu yaprak analiz portalı ekran görüntüsü]

Toplu analiz servisi başarıyla entegre edilmiş, geniş tarım alanlarında çalışan ziraat mühendislerinin toplu numune değerlendirmeleri son derece pratik hale getirilmiştir.

---

## Gün 17 — 12-08-2026: Model Performans ve Saha Adaptasyon Metrikleri Sayfası (`/metrics`)

Çoklu analiz portalını tamamladıktan sonra, bugün sistemin yapay zeka arka planını ve bilimsel başarımlarını şeffafça sergileyen performans metrikleri sayfasını geliştirdim. Modelin laboratuvar ve saha başarımlarının, alan kayması (Domain Shift) analizlerinin ve karmaşıklık matrislerinin kullanıcıya şeffafça sunulacağı özel bir metrik sayfasına ihtiyaç vardı.

`frontend/src/app/metrics/page.tsx` geliştirilmiştir. Sayfaya KPI özet kartları (%96.13 PlantVillage doğruluğu, %26.47 PlantDoc sıfır-vuruş başarımı, 14.8 ms ortalama çıkarım süresi), 15 sınıflı detaylı Precision/Recall/F1 metrik tablosu ve statik analiz görselleri (`confusion_matrix.png`, `learning_curves.png`) entegre edilmiştir.

[EKRAN GÖRÜNTÜSÜ: frontend/src/app/metrics/page.tsx — Metrik KPI kartları ve 15 sınıflı tablo render bileşeni]

[GÖRSEL: metrics_page — Model metrikleri ve saha adaptasyon analizi ekranı]

Saha çoğullamalı modelin %26.47 sıfır-vuruş başarımı ve %96.13 PlantVillage doğruluğu şeffafça sunulmuş, projenin bilimsel derinliği web platformuna taşınmıştır.

---

## Gün 18 — 13-08-2026: Uçtan Uca Entegrasyon Testleri ve Gecikme Doğrulaması

Tüm web arayüz sayfalarını ve bileşenlerini geliştirdikten sonra, bugün istemciden başlayan ve ONNX çıkarımı ile sonlanan tüm ağ akışının gecikme ve kararlılık testlerini gerçekleştirdim. Ön yüzden başlayan, FastAPI üzerinden geçip ONNX Runtime çıkarımı ile sonuçlanan tüm ağ döngüsünün gecikme süresinin <50ms olduğu teyit edilmeliydi.

Tarayıcı geliştirici araçları (Chrome DevTools Network Tab) ve backend erişim logları üzerinden tekli ve toplu görsel tahmin testleri koşturulmuştur. İsteklerin yük boyutu, TLS/HTTP el sıkışma süreleri ve ONNX çıkarım hızları ölçülmüştür. Yapılan testlerde ONNX çıkarım süresinin ortalama 14.8 ms, ağ döngüsü dahil toplam istemci yanıt süresinin ise ~35 ms seviyesinde kaldığı tespit edilmiştir.

[EKRAN GÖRÜNTÜSÜ: e2e_network_tab.png — Tarayıcı geliştirici konsolunda /predict isteği yanıt süresi ve payload ekranı]

Sistemin hedeflenen <50 ms kısıtının altında kaldığı teyit edilmiş, gerçek zamanlı ve kesintisiz bir kullanıcı deneyimi sağlandığı doğrulanmıştır.

---

## Gün 19 — 14-08-2026: Docker ve Docker Compose Konteynerleştirme Mimarisi

Entegrasyon testlerini başarıyla tamamladıktan sonra, bugün tüm platformun farklı sunucu ortamlarında tek komutla ayağa kaldırılabilmesini sağlayan Docker konteynerleştirme altyapısını kurdum. Uygulamanın farklı sunucu ve bulut ortamlarında bağımlılık hatası olmaksızın tek komutla ayağa kaldırılabilmesi için konteynerleştirilmesi şarttı.

Backend için hafif Python 3.12-slim tabanlı `backend/Dockerfile`, frontend için Node 20-alpine çok aşamalı derleme (multi-stage build) tabanlı `frontend/Dockerfile` ve servisleri izole köprü ağında birleştiren kök `docker-compose.yml` yazılmıştır. `docker-compose.yml` dosyası backend servisini Port 8000, frontend servisini Port 3000 üzerinden yayınlayacak ve bağımlılık sırasını (`depends_on`) gözetecek şekilde yapılandırılmıştır.

**Kullanılan Linux Komutu:**
```bash
docker compose build
docker compose up -d
```

[EKRAN GÖRÜNTÜSÜ: backend/Dockerfile — Python 3.12-slim tabanlı backend konteyner yapılandırması]

[EKRAN GÖRÜNTÜSÜ: frontend/Dockerfile — Node 20-alpine çok aşamalı derleme Dockerfile]

[EKRAN GÖRÜNTÜSÜ: docker-compose.yml — Backend ve Frontend servislerinin Docker Compose orkestrasyonu]

`docker compose up --build` komutu ile tüm sistemin sorunsuz konteynerize çalıştığı doğrulanmış ve üretime hazır dağıtım standardı tamamlanmıştır.

---

## Gün 20 — 17-08-2026: Uçtan Uca Sistem Denetimi, LOGBOOK_STAJ2.md Doğrulanması, GitHub Repozituvar Senkronizasyonu ve Staj-II Resmi Kapanışı

Konteynerleştirme altyapısını başarıyla test ettikten sonra, Staj-II'nin son gününde tüm sistemin uçtan uca denetimini gerçekleştirdim, staj günlüğünü doğruladım ve tüm kaynak kodları uzaktaki Git repozituvarına senkronize ederek stajı tamamladım.

Staj-II süresince geliştirilen FastAPI backend servisleri, ONNX çıkarım motoru, Next.js frontend arayüzü, Docker konteynerleştirme bileşenleri ve otomatik test paketlerinin üretim ortamlarında hatasız ve tam performansla çalıştığının doğrulanması; günlük staj seyir defterinin (`LOGBOOK_STAJ2.md`) eksiksiz denetlenmesi ve tüm yazılım varlıklarının uzaktaki Git repozituvarına (`https://github.com/marcravel/crop_disease_detection.git`) push edilmesi adımları sırasıyla yürütülmüştür.

1. `backend/tests/test_predict_api.py` entegrasyon test paketi tekrar çalıştırılarak `4/4 passed` (%100 başarı) teyit edilmiştir.
2. Next.js istemci derlemesi `npm run build` ile koşturulmuş, tüm dinamik ve statik sayfaların (`/`, `/batch`, `/history`, `/metrics`) sıfır tür/lint hatasıyla derlendiği doğrulanmıştır.
3. `LOGBOOK_STAJ2.md` dosyasındaki 20 günlük teknik kayıtlar denetlenmiş ve Staj-II çıkış koşulları kontrol edilmiştir.
4. Git çalışma ağacı temizlenmiş, tüm yeni kaynak kodlar ve dokümanlar `origin/main` dalına senkronize edilmiştir.

**Kullanılan Linux Komutu:**
```bash
PYTHONPATH=. pytest backend/tests/test_predict_api.py
cd frontend && npm run build
git status
git push origin main
```

[EKRAN GÖRÜNTÜSÜ: terminal — git status ve git push origin main onay ekranı]

[EKRAN GÖRÜNTÜSÜ: LOGBOOK_STAJ2.md — 20 günlük staj seyir defteri ve tamamlanan görevler kontrol listesi]

Staj-II kapsamındaki tüm yazılım geliştirme, API tasarımı, web ön yüz entegrasyonu, ONNX model dağıtımı ve konteynerleştirme hedefleri eksiksiz olarak başarıyla tamamlanmış; projenin üretim ortamında yayına hazır olduğu teyit edilerek Staj-II resmi olarak kapatılmıştır.

---

# 3. SONUÇ VE DEĞERLENDİRME

Staj-II kapsamında, Staj-I'de eğitilen derin öğrenme modeli üretime hazır tam yığın bir web platformuna dönüştürülmüştür. Elde edilen temel teknik kazanımlar şunlardır:

1. **Yüksek Hızlı ONNX Çıkarımı:** PyTorch bağımlılığı olmadan ONNX Runtime ile **<50 ms** çıkarım süresi elde edilmiştir.
2. **Modern Monorepo Mimarisi:** FastAPI REST API arka yüzü ile Next.js / TypeScript / Tailwind CSS ön yüzü modüler yapıda entegre edilmiştir.
3. **Konteynerleştirme ve Test:** `pytest` ile %100 API test başarımı ve **Docker Compose** ile tek komutla dağıtım altyapısı kurulmuştur.
