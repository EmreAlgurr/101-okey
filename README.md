# Resmi Profesyonel 101 Okey — Oyun Motoru, Tam El Çözücü, Yapay Zekâ ve Matematiksel Strateji El Kitabı

Python ile sıfırdan geliştirilmiş, **Resmi Profesyonel 101 Okey** kurallarını (ve opsiyonel **Katlamalı + 10× Taş Yedirme Cezalı** ev kurallarını) eksiksiz uygulayan; milisaniye düzeyinde çalışan tam el çözücülü (*Branch & Bound*), bilgi-kümesi tabanlı yapay zekâ ajanlarına (`PlannerAgent`, `IS-MCTS`, `Neural IS-MCTS`, `Heuristic`) ve yerel bir web arayüzüne sahip açık kaynaklı oyun motoru ve matematiksel strateji araştırma projesi.

---

## 1. Proje Mimarisi

```text
101 okey/
├── tiles.py                   # 106 taşlık deste, Gösterge/Okey/Sahte Okey, per/seri/çift doğrulama
├── solver.py                  # Tam Branch & Bound el çözücü (Seri >=101, Çift >=5, min deadwood)
├── engine.py                  # 4 oyunculu headless oyun motoru, cezalar, çarpanlar, ev kuralları
├── benchmark.py               # Koltuk rotasyonlu çoklu ajan turnuva kıyaslayıcısı
├── analyze_strategy.py        # Karar telemetrisi & Counterfactual EV strateji raporu çıkarıcı
├── agents/
│   ├── base.py                # Ortak Agent arayüzü (yalnızca Observation görür, hile yapamaz)
│   ├── heuristic_agent.py     # Güçlü kural tabanlı taban ajan + taş sayma savunması
│   ├── ismcts_agent.py        # PUCT + Top-K budamalı Single-Observer IS-MCTS
│   ├── neural_ismcts_agent.py # PyTorch PolicyValueNet güdümlü sıfır-rollout IS-MCTS
│   └── planner_agent.py       # CRN (Common Random Numbers) tabanlı PlannerAgent v1 & v2.1
├── ml/
│   ├── encoding.py            # 400 boyutlu durum tensörü & 61 boyutlu maskelenmiş aksiyon uzayı
│   ├── model.py               # LayerNorm + GELU ResMLP PolicyValueNet
│   └── train_supervised.py    # CUDA destekli supervised warm-start eğitim boru hattı
├── web/
│   ├── server.py              # FastAPI backend (Oyna & Seyret modları)
│   └── static/index.html      # Sıfır-build-step koyu zümrüt çuha temalı Web UI
└── tests/
    ├── test_rules.py          # Resmi kurallar, 12-13-1 yasağı, Sahte Okey, çarpanlar, 10× ceza
    ├── test_agents.py         # Ajan yasallığı, determinizasyon, hız ve bilgi-gizliliği testleri
    └── test_ml.py             # Tensör boyutları, aksiyon maskeleme ve Neural IS-MCTS testleri
```

---

## 2. Oyun Motoru (`engine.py` & `solver.py`) Nasıl Çalışır?

### Taşlar, Gösterge ve Sahte Okey (`tiles.py`)
- **106 Taş:** 4 renk (`RED`, `YELLOW`, `BLUE`, `BLACK`), 1–13 arası ikişer kopya (104 taş) + **2 Sahte Okey**.
- **Gösterge ve Okey:** Ortadaki desteden açılan gösterge taşının aynı renginde `(numara % 13) + 1` numaralı taş **Okey** olur (`13 → 1` döner). **Sahte Okey**, joker değildir; yalnızca aktif Okey'in rengini ve numarasını birebir temsil eder.
- **Kesin Kurallar:** `12-13-1` sarmal serisi hem `build_run`, hem `validate_meld`, hem de `solver.py` içinde kesin olarak reddedilir.

### Tam El Çözücü (`solver.py`)
- Pivot-taş sıralı **Branch & Bound** algoritması kullanır; 6+ uzunluktaki serileri 3–5'lik alt serilere bölerek (aynı puan ve aynı taş çoklu kümesiyle) dallanmayı minimize eder.
- `solve_best_opening`: El açma için maksimum seri/per puanını bulur (`>= 101` veya Katlamalı barajı).
- `solve_best_pairs`: En fazla çift sayısını bulur; eldeki Okey'leri en yüksek tekil taşlarla eşleyerek kalan deadwood cezasını azaltır.
- `solve_min_deadwood`: Açılmış ellerde kalan cezayı minimize eden perleri (kullanılmayan Okey'in `+101` cezasını hesaba katarak) çözer.
- **Performans:** 2 Okey içeren 22 taşlık ellerde dahi ortalama **1.61 ms** (maksimum **8.49 ms**).

### Tur Döngüsü, Cezalar ve Skorlama (`engine.py`)
- **Dağıtım:** Başlayan oyuncuya 22, diğer 3 oyuncuya 21 taş; ortada 21 taşlık çekme destesi.
- **Standart Arayüz:** `reset()`, `step(action)`, `get_legal_actions()`, `get_observation(player_id)` ve hızlı simülasyonlar için `clone()`.
- **Yandan Taş Alma (`TAKE_DISCARD`):** Solundaki oyuncunun attığı taşı almak **yalnızca** o taş aynı tur içinde hemen açılışta, yeni perde veya işlek olarak kullanılabiliyorsa yasaldır (`must_use_tile_id` ile zorunlu kılınır).
- **Cezalar (`penalties[player_id]`):**
  - Masadaki açık perlere işlenebilen (**işlek**) bir taşı bitmeden atmak: **`+101`**
  - Bitmeden **Okey** atmak: **`+101`**
  - **Ev Kuralı (`feed_penalty_multiplier=10`):** Attığın taşı bir sonraki oyuncu `TAKE_DISCARD` ile alırsa, taşı atan oyuncuya anında **`taş_değeri × 10`** ceza yazılır (ör. `12` yedirirsen `+120`).
- **El Sonu Puanlama (En Düşük Skor Kazanır):**
  - Normal bitme: **`-101`**
  - **Okeyle Bitme** / **Elden Bitme** / **Çiftle Bitme**: **`-202`** ve tüm rakiplerin el cezaları **×2** (Elden + Okey birlikte **`-404`** ve rakipler **×4**).
  - Açamayan rakip: Sabit **`+202`** (çarpanlarla `+404` / `+808`).
  - Seri açmış rakip: Kalan taşların toplamı (elde kalan Okey **`+101`**).
  - Çift açmış rakip: Kalan taşların toplamı **×2** (kazanan çarpanı ile üst üste biner).

---

## 3. Yapay Zekâ Gelişim Süreci & Benchmark Sonuçları

Tüm turnuvalar **koltuk sırası döndürülerek** (4 koltuğun tamamı eşit paylaşılarak) ve ajanların yalnızca kendi `Observation` nesnesini görmesi zorunlu tutularak koşulmuştur.

| Aşama / Ajan | Ort. Skor (↓ iyi) | Kazanma % | Açma % | +101 Ceza/El | Karar Süresi |
|---|---:|---:|---:|---:|---:|
| **Faz 2:** `IS-MCTS` (UCB1, 50 iter) | 99.65 | %5.0 | %60.0 | 0.000 | 82.2 ms |
| **Faz 2.5:** `IS-MCTS` (PUCT + Top-K, 120 iter) | 65.00 | %10.0 | %72.5 | 0.000 | 500.5 ms |
| **Faz 3:** `Neural-ISMCTS` (ResMLP, 120 iter) | 78.42 | %10.0 | %67.5 | 0.000 | 809.2 ms |
| **Taban Ajan:** `HeuristicAgent` (3x ortalama) | 61.51 | %8.9 | %80.6 | 0.000 | 0.24 ms |
| **Faz 4:** **`PlannerAgent v2` (80 CRN senaryo)** | **40.48** | **%16.7** | %78.3 | **0.000** | **199.5 ms** |

### Neden `PlannerAgent` Diğerlerini Açık Ara Geçti?
1. **Kısa Oyun Ufku & Çekiş Varyansı:** Ortada yalnızca 21 taş vardır; yani bir oyuncu el boyunca ortalama **sadece ~5 kez taş çeker**. Rastgele rakip elleri üzerinden 120 ağaç iterasyonu bölmek (IS-MCTS), her aksiyon başına ~20 gürültülü çekiş simülasyonu düşürdüğü için taş atma kararlarında yüksek varyans üretiyordu.
2. **Supervised Öğrenmenin Sınırı (Faz 3):** `PolicyValueNet` (RTX 4060 CUDA üzerinde 1.500 oyundan üretilen 49.979 örnekte **%68.94 doğrulama başarısı**), `HeuristicAgent`'ı taklit ettiği için öğretmeninin üst sınırını aşamadı ve geciktirilmiş açılış durumlarını veri setinde hiç göremedi.
3. **Common Random Numbers (CRN) ile Doğrudan EV Optimizasyonu (`PlannerAgent v2 / v2.1`):**
   - `PlannerAgent`, rastgele rakip hamleleri simüle etmek yerine eldeki görünmeyen taş havuzundan **aynı 80 çekiş senaryosunu (CRN)** örnekleyerek her aday taş atışı, yandan alma ve çift-açma kararının **gerçek beklenen el sonu skorunu** doğrudan hesaplar.
   - `v2.1` yükseltmesiyle: eli `≤ 4` taşa düşen açık rakiplere karşı **erken bitme riski (horizon truncation)** ve **sağdaki oyuncuya taş yedirme maliyeti (`next_player_feed_penalty`)** doğrudan EV hesabına eklenir.
   - Sonuç: Heuristic taban ajana karşı **el başına -21.03 puan fark** (`40.48` vs `61.51`) ve **1.88× daha yüksek kazanma oranı** (`%16.7` vs `%8.9`).

---

## 4. Ampirik 101 Okey Strateji Bulguları (CRN EV Analizi)

`PlannerAgent v2.1` ile **Resmi Kurallar** ve **Katlamalı + 10× Yedirme Cezası** altında 150'şer el (600 oyuncu-el) simüle edilerek karar anlarındaki beklenen değer (EV) eğilimleri, ayrıca 80'er el (320 oyuncu-el) üzerinde **Counterfactual Okey Testi** ve **1–13 Taş Bazlı Atış Dağılımı** çıkarılmıştır:

| Karar / Metrik | Resmi Kurallar | Katlamalı + 10× Ceza | Pratik Çıkarım |
|---|---:|---:|---|
| **Açma Oranı** | ~%80.0 | **%63.5** | Katlamalıda ellerin ~%36'sı açamadan bitiyor; barajı geçme fırsatı nadir. |
| **101+ Seri Gelince Hemen Açma** | **%100** | **%100** | Kısa tur sayısı (~5 çekiş) yüzünden 1 tur bile beklemek net `-EV`. |
| **5 Çift Gelince Hemen Açma** | %92.6 | **%94.0** | Sadece elde `≥90` seri puanı **ve** yerde `≥18` taş varsa seri beklenir. |
| **Açmadan Önce 1–3 Atış Oranı** | %31.1 | **%58.3** | 10× cezada max risk `+10..+30` olduğu için 1–3'ler ilk yakılan taşlardır. |
| **Açmadan Önce 6–11 Atış Oranı** | %36.8 | **%13.6** | Canlı 6–11 atmak hem `+60..+110` ceza hem rakibe katlamalı açılış riskidir. |
| **13 Paradoksu (Açmadan Önce)** | %11.0 | **%11.1** | `12-13-1` yasağı yüzünden 13 tek taraflı uç taştır; canlı 8–10'dan güvenlidir. |
| **Counterfactual Okey Testi** *(Aynı El)* | **`+15.80` EV** (%62.2) | **`+26.01` EV** (%88.9) | Okey'le açabiliyorken harcayıp açmak (`+15.36`), saklayıp beklemekten (`+41.37`) **+26.01 puan** kârlıdır. |
| **Yandan Taş Alma Kazancı (Açılış)** | `+63.87` EV | **`+60.32` EV** | Açılış için soldan taş almak kör çekişe göre ~60 puan avantaj sağlar. |

### 1'den 13'e Tek Tek Taş Atış Yüzdeleri ve Ortalama ΔEV (60 El / 240 Oyuncu-El)

Aşağıdaki tablo, seçilen en iyi atışa göre (`ΔEV = 0.00` en iyi aday) aday taşların ortalama EV farkını (`Avg ΔEV`, düşük = atması daha iyi / ucuz) ve gerçekleşen atış yüzdelerini gösterir:

| Taş (r) | Resmi: Açmamış % | Resmi: Açmış % | Resmi: Avg ΔEV | House: Açmamış % | House: Açmış % | House: Avg ΔEV |
|---:|---:|---:|---:|---:|---:|---:|
| **1** | %12.23 | %1.27 | +7.45 | **%26.11** | %3.11 | +7.94 |
| **2** | %10.07 | %1.98 | +7.33 | **%19.12** | %4.35 | +7.83 |
| **3** | %5.40 | %2.27 | +7.39 | **%9.56** | %4.35 | +6.63 |
| **4** | %4.68 | %5.52 | +6.35 | %5.56 | %4.97 | +6.02 |
| **5** | %7.37 | %7.37 | +6.99 | %5.71 | %4.66 | +6.58 |
| **6** | %6.47 | %7.08 | +6.73 | %4.28 | %7.30 | +5.31 |
| **7** | %4.50 | %8.07 | +5.42 | **%1.85** | %6.68 | +3.87 |
| **8** | %7.19 | %8.36 | +8.65 | **%2.85** | %8.54 | +4.70 |
| **9** | %8.99 | %10.76 | +6.95 | **%1.71** | %8.23 | +6.11 |
| **10** | %9.17 | %11.47 | +7.70 | **%2.43** | %12.42 | +5.72 |
| **11** | %7.19 | %12.89 | +9.53 | **%2.85** | %13.04 | +6.55 |
| **12** | %7.55 | %12.75 | +10.13 | %5.71 | %12.58 | +10.38 |
| **13** | **%9.17** | %10.20 | +10.40 | **%12.27** | %9.78 | +8.61 |

> *Metodoloji Notu: Bu sonuçlar kesin Nash Dengesi değil, `PlannerAgent v2.1`'in CRN (Common Random Numbers) simülasyonlarından elde edilen ampirik EV eğilimleridir.*

---

## 5. 101 Okey Matematiksel Strateji El Kitabı (EV & Olasılık Analizi)

Bu rehber, 101 Okey'deki kritik karar anlarını (**Resmi Kurallar** ve **Katlamalı + 10× Taş Yedirme Cezası** altında) kapalı-form olasılık ve Beklenen Değer (EV) eşikleriyle modeller.

### 5.1. Oyunun Temel Kısıtı: Kısa Ufuk ve Asimetri

- **Çekiş Ufku (Horizon):** 106 taşın 85'i dağıtılır ($22 + 21 + 21 + 21$). Ortada yalnızca $D = 21$ taş kalır; her oyuncunun el boyunca göreceği maksimum çekiş sayısı:

$$k = \left\lfloor \frac{D}{4} \right\rfloor \approx 5 \text{ çekiş}$$

- **Ödeme Asimetrisi:** El açamamanın cezası $+202$, açıp elinde taş kalmasının tipik maliyeti $+25$ ile $+55$, bitmenin ödülü $-101$ (Elden/Okeyle $-202$). Kısa ufuk + ağır $+202$ cezası, 101 Okey'i uzun vadeli kurulum oyunundan ziyade **hasar kontrolü (damage control)** oyununa dönüştürür.

### 5.2. Teorem 1: 10× Ceza Kuralında "%10 Sınırı"

Sağındaki oyuncu senin attığın $r \in \{1..13\}$ değerli taşı alıp açar veya işlerse $10r$ ceza yediğin kuralda, elini açmışken o taşı elde tutup el sonu yazmakla atmak arasındaki başabaş noktası taşın değerinden ($r$) bağımsızdır.

- **Seri açmışsan:** Elde tutmanın tavan maliyeti $r$, atmanın beklenen ceza maliyeti $10r \cdot p$ ($p$: sağındaki oyuncunun o taşı alma olasılığı):

$$10r \cdot p > r \implies p > 0.10 \text{ (\%10)}$$

- **Çift açmışsan:** Elde tutmanın tavan maliyeti $2r$ olduğundan sınır ikiye katlanır:

$$10r \cdot p > 2r \implies p > 0.20 \text{ (\%20)}$$

> **Kural:** Elini seri açtıktan sonra, sağındaki oyuncunun alma ihtimali **%10'u geçen** hiçbir canlı taşı (özellikle 9–12) atma; elinde tutup el sonu $r$ puan yazmak matematiksel olarak daha ucuzdur.

### 5.3. Teorem 2: Taş Bağlanabilirliği ve "13 Paradoksu"

Herhangi bir taşın rakip elinde bir 3'lü per (seri veya grup) tamamlama esnekliği, katılabileceği farklı 3'lü kombinasyon sayısıyla doğru orantılıdır:

- **Grup (Set) esnekliği:** Her sayı için diğer 3 renkten $\binom{3}{2} = 3$ farklı 3'lü grup vardır.
- **Seri (Run) esnekliği (`12-13-1` yasak):**
  - **1 ve 13 (Uç taşlar):** Yalnızca 1 seri (`1-2-3` veya `11-12-13`) $\rightarrow$ **Toplam 4 kombinasyon**.
  - **2 ve 12:** 2 seri $\rightarrow$ **Toplam 5 kombinasyon**.
  - **3–11 (İç taşlar):** 3 seri $\rightarrow$ **Toplam 6 kombinasyon**.

$$\text{10× Ceza Tehlike İndeksi} = (\text{Toplam 3'lü Kombinasyon}) \times (10r)$$

| Taş Değeri ($r$) | 3'lü Kombinasyon | Max Ceza ($10r$) | Tehlike İndeksi | Karar Sınıfı |
|---:|---:|---:|---:|---|
| **1** | 4 | +10 | **40** | En güvenli yakmalık |
| **2** | 5 | +20 | **100** | Güvenli yakmalık |
| **3** | 6 | +30 | **180** | Düşük risk |
| **4** | 6 | +40 | **240** | Düşük–orta risk |
| **5** | 6 | +50 | **300** | Orta risk |
| **6** | 6 | +60 | **360** | Orta risk |
| **7** | 6 | +70 | **420** | Orta–yüksek risk |
| **8** | 6 | +80 | **480** | Yüksek risk |
| **13** | **4** | +130 | **520** | **13 Paradoksu:** 9–12'den güvenli |
| **9** | 6 | +90 | **540** | Çok tehlikeli |
| **10** | 6 | +100 | **600** | Çok tehlikeli |
| **12** | 5 | +120 | **600** | Çok tehlikeli |
| **11** | **6** | +110 | **660** | **Oyunun en zehirli taşı** |

> **Sonuç:** `12-13-1` yasağı `13`'ün seri esnekliğini üçte birine düşürür. Canlı (komşuları çıkmamış) bir `9, 10, 11` veya `12` atmak, izole bir `13` atmaktan matematiksel olarak daha yüksek beklenen ceza taşır.

### 5.4. Teorem 3: Hipergeometrik "Out" Tablosu (Çift mi, Seri mi?)

Kapalı (görünmeyen) $U$ taş içinden seni 101'e (veya baraja) ulaştıracak $m$ tane "out" taş varsa, kalan $k$ çekişte bunlardan en az birini çekme olasılığın kesin olarak:

$$P(\text{seri}) = 1 - \frac{\binom{U - m}{k}}{\binom{U}{k}}$$

| Kalan Çekiş ($k$) | Kapalı Taş ($U$) | $m=2$ Out | $m=4$ Out | $m=6$ Out | $m=8$ Out |
|---|---:|---:|---:|---:|---:|
| **$k=5$ (Oyun başı)** | 75 | %13.0 | %24.6 | %34.9 | %44.0 |
| **$k=3$ (Orta oyun)** | 65 | %9.1 | %17.6 | %25.6 | %33.0 |
| **$k=1$ (Son tur)** | 55 | %3.6 | %7.3 | %10.9 | %14.5 |

#### 5 Çift Varken Seri Bekleme Şartı
Çift açtığında elinde kalacak puan $D_{\text{çift}}$ (el sonu ceza $2D_{\text{çift}}$), seri açarsan kalacak puan $D_{\text{seri}}$ olsun. Seri beklemenin Çift açmaktan kârlı olması için gereken minimum olasılık:

$$P(\text{seri}) \cdot D_{\text{seri}} + (1 - P(\text{seri})) \cdot 202 < 2D_{\text{çift}} \implies P_{\min} = \frac{202 - 2D_{\text{çift}}}{202 - D_{\text{seri}}}$$

- **Sayısal Örnek:** $D_{\text{çift}} = 45$ ($2D_{\text{çift}} = 90$) ve $D_{\text{seri}} = 35$ ise:

$$P_{\min} = \frac{202 - 90}{202 - 35} = \%67.1$$

Oyun başında ($k=5$) 8 tane out taşın olsa bile seri yakalama ihtimalin yalnızca **%44.0**'tür ($44.0 < 67.1$). Bu nedenle 5 çift geldiğinde seri beklemek, yalnızca elin $2D_{\text{çift}} > 130$ civarında çok ağır ve $k=5$ (ilk tur) iken haklı çıkabilir; diğer neredeyse tüm durumlarda anında Çift açmak zorunlu $+EV$'dir.

### 5.5. Teorem 4: 101'i Bulunca 1 Tur Bekleme Eşiği (Elden / Bloke)

Şu an elinde $D$ deadwood bırakarak açabiliyorsun. 1 tur bekleyip Elden bitme ($-202$) kovalamak için gereken minimum bitme olasılığı ($p_{\text{kilit}}$: o 1 tur içinde elin bitmesi veya katlamalı barajının seni kilitlemesi olasılığı):

$$p_{\text{bit}} > \frac{p_{\text{kilit}} \cdot (202 - D)}{202 + D + p_{\text{kilit}} \cdot (202 - D)}$$

$D = 30$ ve katlamalıda kilitlenme/bitme riski yalnızca $p_{\text{kilit}} = 0.20$ iken:

$$p_{\text{bit}} > \frac{0.20 \cdot 172}{232 + 34.4} \approx \%12.9$$

Tek çekişte ($U \approx 75$) %12.9 bitme olasılığı için seni doğrudan elden bitirecek $\ge 10$ farklı out gerekir. 21 taşlık açılmamış elde bu pratikte oluşmadığından, barajı bulduğun an beklemeden açmak kesin optimaldir.

### 5.6. Teorem 5: Okey (Joker) Saklama vs. Harcama Koşulu

- **Henüz açmamışken:** Okey'i harcayıp açmak seni $+202$ riskinden kurtarır; Teorem 4 gereği açılış sağlayan Okey'i elinde tutmak neredeyse her zaman ağır $-EV$'dir (Counterfactual testte **+15.80** / **+26.01 EV** kaybı).
- **Zaten açmışken (Okeyle Bitme Kovalamak):** Okeyle bitmek sana normal bitmeye göre ekstra $-101$ bonus verir; ancak elinde Okey varken başka biri bitirirse (veya taşlar biterse) Okey elinde kaldığı için $+101$ ek ceza yersin. Kazanç ve ceza simetrik olduğundan karar kuralı nettir:

$$P(\text{Okeyle bitmem}) > P(\text{El bitmeden Okey'le yakalanmam})$$

Masada 2–3 taşı kalmış açmış bir rakip varsa veya yerde $\le 8$ taş ($k \le 2$) kaldıysa, Okey'i saklamayı bırakıp işlek/per olarak harcamak $+EV$'dir.

### 5.7. Masada Tek Bakışta Karar Özeti

1. **Barajı bulduğun an aç:** Tek tur beklemenin başabaş şartı (%12+ tek çekişte elden bitme) pratikte sağlanmaz.
2. **5 Çift = İniş:** İlk turda $\ge 90$ seri puanın ve çok yüksek çift cezan ($2D_{\text{çift}} > 130$) yoksa seri bekleme.
3. **Açmadan önce yakılacaklar:** Önce `1–3`, sonra komşusu çıkmış ölü taşlar, sonra izole `13`. Canlı `9–11`'ler en son atılır.
4. **Açtıktan sonra "%10 Kuralı":** Sağındaki oyuncunun alma ihtimali **%10**'u geçen (Seri) veya **%20**'yi geçen (Çift) hiçbir taşı atma; elinde tut.
5. **Okey egosu yapma:** Açılışa yarıyorsa anında harca; açmışsan da yalnızca rakiplerden önce bitme şansın **%50**'nin üzerindeyken sakla.

---

## 6. Kurulum ve Çalıştırma

### Testlerin Çalıştırılması
Tüm kural, çözücü, ajan ve ML testlerini çalıştırmak için:
```bash
pytest -v
```
*(Mevcut tüm 30 birim testi başarıyla geçmektedir.)*

### Web Arayüzünü Başlatma (Oyna veya Botları İzle)
`PlannerAgent` veya `HeuristicAgent` botlarına karşı oynamak ya da 4 botu **Spectator Mode**'da (oynat/duraklat/hız ayarı ve "Rakiplerin Ellerini Göster" seçeneği ile) izlemek için:
```bash
uvicorn web.server:app --reload --port 8000
```
Tarayıcıdan **`http://127.0.0.1:8000`** adresine gidin. Üst menüden **Katlamalı** veya **House (Katlamalı + 10× Feed)** modlarını açıp kapatabilirsiniz.

### Benchmark ve Strateji Analizi Komutları
```bash
# 1 PlannerAgent v2 vs 3 HeuristicAgent (60 el)
python benchmark.py --agent planner --rounds 60

# 2 PlannerAgent v2 vs 2 PlannerAgent v1
python benchmark.py --agent v2_vs_v1 --rounds 40

# House Rules (Katlamalı + 10x Yedirme Cezası) Strateji Raporu
python analyze_strategy.py --rounds 150 --katlamali --feed-penalty-mult 10
```
