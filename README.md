# Resmi Profesyonel 101 Okey — Oyun Motoru, Tam El Çözücü, Yapay Zekâ ve Strateji Analizi

Python ile sıfırdan geliştirilmiş, **Resmi Profesyonel 101 Okey** kurallarını (ve opsiyonel **Katlamalı + 10× Taş Yedirme Cezalı** ev kurallarını) eksiksiz uygulayan; milisaniye düzeyinde çalışan tam el çözücülü (*Branch & Bound*), bilgi-kümesi tabanlı yapay zekâ ajanlarına (`PlannerAgent`, `IS-MCTS`, `Neural IS-MCTS`, `Heuristic`) ve yerel bir web arayüzüne sahip açık kaynaklı oyun ve simülasyon motoru.

---

## 1. Proje Mimarisi

```text
101 okey/
├── tiles.py                  # 106 taşlık deste, Gösterge/Okey/Sahte Okey, per/seri/çift doğrulama
├── solver.py                 # Tam Branch & Bound el çözücü (Seri >=101, Çift >=5, min deadwood)
├── engine.py                 # 4 oyunculu headless oyun motoru, cezalar, çarpanlar, ev kuralları
├── benchmark.py              # Koltuk rotasyonlu çoklu ajan turnuva kıyaslayıcısı
├── analyze_strategy.py       # Karar telemetrisi ile Optimal 101 Okey Strateji Raporu çıkarıcı
├── agents/
│   ├── base.py               # Ortak Agent arayüzü (yalnızca Observation görür, hile yapamaz)
│   ├── heuristic_agent.py    # Güçlü kural tabanlı taban ajan + taş sayma savunması
│   ├── ismcts_agent.py       # PUCT + Top-K budamalı Single-Observer IS-MCTS
│   ├── neural_ismcts_agent.py# PyTorch PolicyValueNet güdümlü sıfır-rollout IS-MCTS
│   └── planner_agent.py      # CRN (Common Random Numbers) tabanlı PlannerAgent v1 & v2.1
├── ml/
│   ├── encoding.py           # 400 boyutlu durum tensörü & 61 boyutlu maskelenmiş aksiyon uzayı
│   ├── model.py              # LayerNorm + GELU ResMLP PolicyValueNet
│   └── train_supervised.py   # CUDA destekli supervised warm-start eğitim boru hattı
├── web/
│   ├── server.py             # FastAPI backend (Oyna & Seyret modları)
│   └── static/index.html     # Sıfır-build-step koyu zümrüt çuha temalı Web UI
└── tests/
    ├── test_rules.py         # Resmi kurallar, 12-13-1 yasağı, Sahte Okey, çarpanlar, 10× ceza
    ├── test_agents.py        # Ajan yasallığı, determinizasyon, hız ve bilgi-gizliliği testleri
    └── test_ml.py            # Tensör boyutları, aksiyon maskeleme ve Neural IS-MCTS testleri
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

## 4. Çıkarılan Optimal 101 Okey Stratejisi (600 El Analizi)

`analyze_strategy.py` ile hem **Resmi Kurallar** hem de **Katlamalı + 10× Yedirme Cezası** altında 150'şer el (600 oyuncu-el) simüle edilerek aşağıdaki matematiksel sonuçlar elde edilmiştir:

| Stratejik Kriter | Resmi Kurallar | House Rules (Katlamalı + 10× Ceza) | Optimal Çıkarım |
|---|---:|---:|---|
| **Ortalama Açma Oranı** | ~%80.0 | **%63.5** | Katlamalıda ~1–2 kişi `+202` yer; ilk açan olmak kritiktir. |
| **Seri 101+ Geldiğinde Hemen Açma** | **%100** | **%100** | `+202` riski ve barajın kilitlenmesi, Elden beklemeyi ezer. |
| **5 Çift Geldiğinde Hemen Açma** | %92.6 | **%94.0** | Sadece ortada `≥18` taş VE elde `≥90` seri varsa 1 tur beklenir. |
| **Açmamışken 1–3 Atış Oranı** | %31.1 | **%58.3** | Küçük taşlar hem 101'e az katkı verir hem yedirme riski düşüktür (`+10..+30`). |
| **Açmamışken 6–11 Atış Oranı** | %36.8 | **%13.6** | Canlı 6–11 atmak hem `+60..+110` ceza hem rakibe açılış hediyesi demektir. |
| **"13 Paradoksu" (Uç Taş)** | %11.0 | **%11.1** | `12-13-1` yasağı yüzünden `13` tek taraflı uç taştır; canlı `8–10`'lardan daha güvenlidir. |
| **Açtıktan Sonra 1–4 Atış Oranı** | %11.6 | **%21.0** | 10× cezalı masada ölü düşük taş atmak, canlı `10–12` atmaktan güvenlidir. |
| **Okey'i Açılışta Harcama vs Açamama** | `+6.78` vs `+202.0` | **`+51.60` vs `+219.95`** | Okey'i harcayıp elini açabilen ellerde ort. puan `+6.78`/`+51.60`'a düşer; el açamayıp Okey'le yakalanmanın bedeli ağırdır. |
| **Yandan Taş Alma Kazancı (Açılış)** | `+63.87` EV | **`+60.32` EV** | Seni açtırıyorsa solundan taş almak hem sana `+60` kazandırır hem rakibe `10×` ceza yazar. |

> *Metodoloji Notu: Bu sonuçlar kesin Nash Dengesi değil, `PlannerAgent v2.1`'in 600'er eldeki CRN (Common Random Numbers) simülasyonlarından türetilmiş ampirik EV eğilimleridir.*

---

## 5. Kurulum ve Çalıştırma

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
