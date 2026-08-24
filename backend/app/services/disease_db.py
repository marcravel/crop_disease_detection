"""
Agricultural Disease Knowledge Base for 15 PlantVillage Classes.
Provides symptoms, organic treatments, chemical treatments, and prevention protocols.
"""

DISEASE_KNOWLEDGE_BASE = {
    "Pepper__bell___Bacterial_spot": {
        "disease_id": "Pepper__bell___Bacterial_spot",
        "name_tr": "Biber Bakteriyel Leke Hastalığı",
        "name_en": "Pepper Bell Bacterial Spot",
        "crop_type": "Pepper",
        "is_healthy": False,
        "severity": "Moderate to High",
        "description": "Xanthomonas bakterisinin neden olduğu, yapraklarda sulu koyu lekeler ve meyvelerde kabarık lekelere yol açan yaygın bir hastalık.",
        "symptoms": [
            "Yapraklarda küçük, yağmsı/sulu sarımsı yeşil lekeler",
            "Lekelerin zamanla kahverengileşmesi ve kuruması",
            "Yaprakların dökülmesi ve meyvede içbükey koyu noktalar"
        ],
        "organic_treatment": [
            "Bakır bazlı organik fungisitler uygulanması",
            "Enfekte olmuş yaprak ve bitki kalıntılarının derhal imha edilmesi",
            "Neem yağı (tesbih ağacı yağı) püskürtülmesi"
        ],
        "chemical_treatment": [
            "Bakır hidroksit veya bakır oksiklorür içerikli bakterisitler",
            "Mankozeb ile kombine bakır spreyleri"
        ],
        "prevention": [
            "Sertifikalı hastalıksız tohum kullanımı",
            "Damlama sulama kullanarak yaprakların ıslanmasını önleme",
            "Bitkiler arası yeterli havalandırma mesafesi bırakma"
        ]
    },
    "Pepper__bell___healthy": {
        "disease_id": "Pepper__bell___healthy",
        "name_tr": "Sağlıklı Biber Yaprağı",
        "name_en": "Pepper Bell Healthy",
        "crop_type": "Pepper",
        "is_healthy": True,
        "severity": "None",
        "description": "Bitkide herhangi bir hastalık veya zararlı belirtisi tespit edilmemiştir. Canlı yeşil doku ve düzenli gelişim mevcuttur.",
        "symptoms": ["Leke, sararma veya deformasyon yoktur."],
        "organic_treatment": ["Rutin organik gübreleme ve kompost takviyesi."],
        "chemical_treatment": ["Kimyasal müdahaleye gerek yoktur."],
        "prevention": [
            "Düzenli sulama ve dengeli besleme",
            "Toprak neminin ve havalandırmasının korunması"
        ]
    },
    "Potato___Early_blight": {
        "disease_id": "Potato___Early_blight",
        "name_tr": "Patates Erken Yaprak Yanıklığı",
        "name_en": "Potato Early Blight",
        "crop_type": "Potato",
        "is_healthy": False,
        "severity": "Moderate",
        "description": "Alternaria solani mantarının yol açtığı, yaşlı yapraklarda konsantrik (hedef tahtası şeklinde) halkalı kahverengi lekelere neden olan fungal hastalık.",
        "symptoms": [
            "Yapraklarda konsantrik daireli koyu kahverengi lekeler",
            "Leke etrafında sarı hale oluşumu",
            "Alt yapraklardan başlayarak yukarı doğru kuruma"
        ],
        "organic_treatment": [
            "Kükürt veya bakır sülfat spreyleri",
            "Aşırı azotlu gübrelemeden kaçınma"
        ],
        "chemical_treatment": [
            "Chlorothalonil veya Mancozeb etken maddeli fungisitler",
            "Azoxystrobin veya Pyraclostrobin uygulaması"
        ],
        "prevention": [
            "En az 3 yıllık ekim nöbeti (münavebe) uygulanması",
            "Hasat sonrası bitki artıklarının temizlenmesi"
        ]
    },
    "Potato___Late_blight": {
        "disease_id": "Potato___Late_blight",
        "name_tr": "Patates Geç Yaprak Yanıklığı (Mildiyö)",
        "name_en": "Potato Late Blight",
        "crop_type": "Potato",
        "is_healthy": False,
        "severity": "Severe",
        "description": "Phytophthora infestans oomiseti tarafından oluşturulan, elverişli nemli koşullarda tarlayı günler içinde yok edebilen yıkıcı hastalık.",
        "symptoms": [
            "Yaprak uçlarında ve kenarlarında düzensiz su emmiş koyu lekeler",
            "Nemli havalarda yaprak altında beyaz küf tabakası",
            "Yumrularda kahverengi çürüklük ve kötü koku"
        ],
        "organic_treatment": [
            "Erken evrede koruyucu bordo bulamacı uygulaması",
            "Hasta bitki kısımlarının yakılarak imhası"
        ],
        "chemical_treatment": [
            "Metalaxyl + Mancozeb kombinasyonları",
            "Dimethomorph veya Propamocarb sistemik fungisitleri"
        ],
        "prevention": [
            "Dirençli patates çeşitlerinin seçilmesi",
            "Yaprakların kuru kalması için damlama sulama tercihi"
        ]
    },
    "Potato___healthy": {
        "disease_id": "Potato___healthy",
        "name_tr": "Sağlıklı Patates Yaprağı",
        "name_en": "Potato Healthy",
        "crop_type": "Potato",
        "is_healthy": True,
        "severity": "None",
        "description": "Patates bitkisi yaprakları sağlıklı, lekesiz ve güçlü yapılıdır.",
        "symptoms": ["Hastalık belirtisi gözlenmemektedir."],
        "organic_treatment": ["Dengeli organik gübreleme."],
        "chemical_treatment": ["Gerek yoktur."],
        "prevention": ["Rutin tarımsal bakım ve kontrol."]
    },
    "Tomato_Bacterial_spot": {
        "disease_id": "Tomato_Bacterial_spot",
        "name_tr": "Domates Bakteriyel Leke Hastalığı",
        "name_en": "Tomato Bacterial Spot",
        "crop_type": "Tomato",
        "is_healthy": False,
        "severity": "Moderate to High",
        "description": "Xanthomonas vesicatoria türlerinin neden olduğu, yaprak ve meyvede koyu lekeler ile verim kaybına yol açan bakteriyel enfeksiyon.",
        "symptoms": [
            "Yapraklarda ve gövdede küçük (1-3 mm) siyah/kahverengi lekeler",
            "Lekelerin çevresinde sarı halkalar ve yaprak delinmesi",
            "Meyve üzerinde pürüzlü kabarık lekelenmeler"
        ],
        "organic_treatment": [
            "Bakır içerikli bilesikler ve bakır sabunu",
            "Bakteriyel yayılımı azaltmak için budama araçlarının sterilizasyonu"
        ],
        "chemical_treatment": [
            "Bakır hidroksit + Mankozeb sprey karışımları",
            "Streptomisin (izin verilen bölgelerde)"
        ],
        "prevention": [
            "Yağmurlama sulamadan kaçınılması",
            "Temiz tohum ve fidelik kullanımı"
        ]
    },
    "Tomato_Early_blight": {
        "disease_id": "Tomato_Early_blight",
        "name_tr": "Domates Erken Yaprak Yanıklığı",
        "name_en": "Tomato Early Blight",
        "crop_type": "Tomato",
        "is_healthy": False,
        "severity": "Moderate",
        "description": "Alternaria solani mantarının yol açtığı, hedef tahtası desenli halkalı lekeler içeren fungal hastalık.",
        "symptoms": [
            "Yaşlı yapraklarda halkalı siyah-kahverengi lekeler",
            "Yaprakların sararıp kuruması ve dökülmesi",
            "Gövdede içbükey lezyonlar"
        ],
        "organic_treatment": [
            "Kükürt tozu veya bakır spreyi",
            "Toprağın yapraklara sıçramasını önlemek için malçlama"
        ],
        "chemical_treatment": [
            "Klorotalonil, Mankozeb veya Difenokonazol fungisitleri"
        ],
        "prevention": [
            "Bitki alt yapraklarının budanması",
            "Solanaceae familyası dışındaki ürünlerle ekim nöbeti"
        ]
    },
    "Tomato_Late_blight": {
        "disease_id": "Tomato_Late_blight",
        "name_tr": "Domates Geç Yaprak Yanıklığı (Mildiyö)",
        "name_en": "Tomato Late Blight",
        "crop_type": "Tomato",
        "is_healthy": False,
        "severity": "Severe",
        "description": "Phytophthora infestans kökenli, soğuk ve nemli havalarda hızla yayılarak tüm tarlayı tehdit eden tehlikeli hastalık.",
        "symptoms": [
            "Gri-yeşil renkli, su emmiş hızla büyüyen lekeler",
            "Yaprak altında nemli koşullarda beyaz spor tabakası",
            "Gövde ve meyvede sert kahverengi çürümeler"
        ],
        "organic_treatment": [
            "Bordo bulamacı koruyucu uygulaması",
            "Enfekte olmuş tüm bitki organlarının sökülüp yakılması"
        ],
        "chemical_treatment": [
            "Metalaksil, Kymoxanil, Dimetomorf etken maddeli fungisitler"
        ],
        "prevention": [
            "Sera ve tarla havalandırmasının artırılması",
            "Erken uyarı sistemlerinin izlenmesi"
        ]
    },
    "Tomato_Leaf_Mold": {
        "disease_id": "Tomato_Leaf_Mold",
        "name_tr": "Domates Yaprak Küfü",
        "name_en": "Tomato Leaf Mold",
        "crop_type": "Tomato",
        "is_healthy": False,
        "severity": "Moderate",
        "description": "Passalora fulva (Cladosporium fulvum) mantarının özellikle seralarda yüksek nem koşullarında oluşturduğu yaprak küfü.",
        "symptoms": [
            "Yaprak üst yüzeyinde soluk yeşil/sarı lekeler",
            "Yaprak alt yüzeyinde kadifemsi zeytin yeşili/kahverengi küf tabakası",
            "Yaprakların kıvrılması ve kuruması"
        ],
        "organic_treatment": [
            "Sera neminin %85'in altına düşürülmesi",
            "Bakır esaslı mantar ilaçları"
        ],
        "chemical_treatment": [
            "Difenoconazole, Chlorothalonil veya Myclobutanil fungisitleri"
        ],
        "prevention": [
            "Sera havalandırma fanlarının çalıştırılması",
            "Bitki sıklığının azaltılması"
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "disease_id": "Tomato_Septoria_leaf_spot",
        "name_tr": "Domates Septoria Yaprak Lekesi",
        "name_en": "Tomato Septoria Leaf Spot",
        "crop_type": "Tomato",
        "is_healthy": False,
        "severity": "Moderate",
        "description": "Septoria lycopersici mantarının neden olduğu, çok sayıda küçük gri merkezli yuvarlak lekelerle karakterize hastalık.",
        "symptoms": [
            "Yapraklarda koyu kenarlı, gri/beyaz merkezli çok sayıda küçük nokta",
            "Merkezde siyah küçük piknit yapıları",
            "Erken yaprak dökümü"
        ],
        "organic_treatment": [
            "Bakır bazlı fungisit çözeltileri",
            "Malç uygulaması ile spora temasın kesilmesi"
        ],
        "chemical_treatment": [
            "Mancozeb, Chlorothalonil veya Azoxystrobin"
        ],
        "prevention": [
            "Hasat artıkları imhası",
            "Sulamanın sabah saatlerinde yapılması"
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "disease_id": "Tomato_Spider_mites_Two_spotted_spider_mite",
        "name_tr": "Domates İki Noktalı Kırmızı Örümcek Zararlısı",
        "name_en": "Tomato Two-Spotted Spider Mite",
        "crop_type": "Tomato",
        "is_healthy": False,
        "severity": "Moderate to High",
        "description": "Tetranychus urticae isimli küçük akarın yaprak özsuyunu emerek sararmalara ve ince ağ yapısına yol açması durumu.",
        "symptoms": [
            "Yaprak üst yüzeyinde sarı/gümüşi ince noktalanmalar",
            "Yaprak altlarında ve filizlerde ince ağ yapıları",
            "Yaprakların bronzlaşıp kuruması"
        ],
        "organic_treatment": [
            "Potasyum sabunu (Akarasit sabunlar)",
            "Neem yağı veya doğal avcı akar (Phytoseiulus persimilis) salımı"
        ],
        "chemical_treatment": [
            "Abamectin, Spiromesifen veya Bifenazate içerikli akarisitler"
        ],
        "prevention": [
            "Sera neme dengesinin korunması (kuru sıcak havayı severler)",
            "Yabancı ot mücadelesi"
        ]
    },
    "Tomato__Target_Spot": {
        "disease_id": "Tomato__Target_Spot",
        "name_tr": "Domates Hedef Leke Hastalığı",
        "name_en": "Tomato Target Spot",
        "crop_type": "Tomato",
        "is_healthy": False,
        "severity": "Moderate",
        "description": "Corynespora cassiicola mantarının neden olduğu, yapraklarda ve meyvelerde hedef tahtasına benzer lekeler oluşturan hastalık.",
        "symptoms": [
            "Yapraklarda açık kahverengi merkezli, belirgin kenarlı dairesel lekeler",
            "Meyvelerde çökük çürüklük noktaları",
            "Gövde üzerinde lezyonlar"
        ],
        "organic_treatment": [
            "Bakır bazlı organik mantar koruyucuları",
            "Budama ile hava akışının artırılması"
        ],
        "chemical_treatment": [
            "Boscalid, Pyraclostrobin veya Chlorothalonil"
        ],
        "prevention": [
            "Yüksek nemden kaçınma",
            "Münavebe uygulanması"
        ]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "disease_id": "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "name_tr": "Domates Sarı Yaprak Kıvırcıklık Virüsü (TYLCV)",
        "name_en": "Tomato Yellow Leaf Curl Virus",
        "crop_type": "Tomato",
        "is_healthy": False,
        "severity": "Severe",
        "description": "Beyazsinek (Bemisia tabaci) vektörü ile taşınan, bitkide bodurlaşma, yaprak sararması ve kıvrılmaya yol açan virüs hastalığı.",
        "symptoms": [
            "Yaprak kenarlarında yukarı doğru içbükey kıvrılma ve çanaklaşma",
            "Yaprak damar aralarında şiddetli sararma",
            "Bitkide bodurlaşma ve meyve bağlama durması"
        ],
        "organic_treatment": [
            "Sarı yapışkan tuzaklar ile beyazsinek mücadelesi",
            "Virüslü bitkilerin derhal sökülüp imha edilmesi"
        ],
        "chemical_treatment": [
            "Virüslere doğrudan ilaç yoktur; vektör beyazsinek için Imidacloprid, Acetamiprid veya Spirotetramat insektisitleri"
        ],
        "prevention": [
            "Tül/tülnet ile seraların beyazsineğe karşı korunması",
            "Dirençli (TYLCV tolerant) fide kullanımı"
        ]
    },
    "Tomato__Tomato_mosaic_virus": {
        "disease_id": "Tomato__Tomato_mosaic_virus",
        "name_tr": "Domates Mozaik Virüsü (ToMV)",
        "name_en": "Tomato Mosaic Virus",
        "crop_type": "Tomato",
        "is_healthy": False,
        "severity": "High",
        "description": "Mekanik temas ve tohumla kolayca bulaşan, yapraklarda alacalı yeşil-sarı mozaik desenleri oluşturan virüs.",
        "symptoms": [
            "Yapraklarda açık ve koyu yeşil alacalı mozaik deseni",
            "Yapraklarda büzüşme, daralma (eğrelti otu görünümü)",
            "Meyve iç dokusunda kahverengileşme"
        ],
        "organic_treatment": [
            "Hastalıklı bitkilerin sökülmesi",
            "İşlemler sırasında ellerin ve aletlerin süt veya çamaşır suyu çözeltisi ile dezenfeksiyonu"
        ],
        "chemical_treatment": [
            "Virüslere karşı doğrudan kimyasal ilaç yoktur. Hijyen önlemleri esastır."
        ],
        "prevention": [
            "Sertifikalı virüssüz tohum",
            "Tütün ürünleri kullanan çalışanların hijyen kurallarına uyması"
        ]
    },
    "Tomato_healthy": {
        "disease_id": "Tomato_healthy",
        "name_tr": "Sağlıklı Domates Yaprağı",
        "name_en": "Tomato Healthy",
        "crop_type": "Tomato",
        "is_healthy": True,
        "severity": "None",
        "description": "Domates bitkisi yaprakları canlı yeşil, düzgün formda ve tamamen sağlıklıdır.",
        "symptoms": ["Herhangi bir enfeksiyon veya zararlı izi bulunmamaktadır."],
        "organic_treatment": ["Düzenli kompost şerbeti veya sıvı organik gübre."],
        "chemical_treatment": ["İlaçlama gereksizdir."],
        "prevention": [
            "Düzenli sulama ve dengeli gübreleme",
            "Periyodik tarla/sera kontrolleri"
        ]
    }
}

def get_disease_info(class_name: str) -> dict:
    """
    Retrieves agricultural disease information for a given class name.
    """
    return DISEASE_KNOWLEDGE_BASE.get(class_name, {
        "disease_id": class_name,
        "name_tr": class_name.replace("_", " "),
        "name_en": class_name.replace("_", " "),
        "crop_type": "Unknown",
        "is_healthy": "healthy" in class_name.lower(),
        "severity": "Unknown",
        "description": "Detaylı bilgi bulunamadı.",
        "symptoms": [],
        "organic_treatment": [],
        "chemical_treatment": [],
        "prevention": []
    })
