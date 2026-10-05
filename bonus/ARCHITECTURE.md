# Kiến trúc — Trợ lý AI cá nhân với Hybrid Memory

**Contributors:** K4 — Track 2 (bài cá nhân).
**Repo:** cùng repo Lab 19 (`bonus/`).
**Mục tiêu:** POC tối thiểu cho một trợ lý tiếng Việt có **episodic memory**
(vector store) + **stable profile** (feature store) + **recent activity**
(streaming feature view), ghép thành context trước khi gọi LLM.

---

## 1. Sơ đồ kiến trúc

```
                    ┌──────────────────────────────────────────────────────┐
   USER (VN/EN)     │  WRITE PATH                                          │
      │             │                                                      │
      │ chat/doc    ▼                                                      │
      │      ┌──────────────┐   chunk    ┌──────────────┐   embed   ┌──────┴─────┐
      └─────▶│  Ingestion   │───────────▶│  Chunker     │──────────▶│  Embedder  │
             │  (agent)     │            │  384/1024d   │           │ (bge-m3)   │
             └──────┬───────┘            └──────────────┘           └──────┬─────┘
                    │ profile signals                                     │
                    │ (lang, wpm, topic, ts)                              │ upsert
                    ▼                                                     ▼
             ┌──────────────┐                              ┌────────────────────────┐
             │ Push/Stream  │                              │  Qdrant  (episodic)    │
             │ → Feast      │                              │  payload: user_id,     │
             │ online store │                              │  ts, topic, source     │
             └──────┬───────┘                              └───────────┬────────────┘
                    │ materialize                                      │
                    ▼                                                  │
   ┌───────────────────────────────┐                                   │
   │  FEAST FEATURE STORE          │                                   │
   │  ├ user_profile   (TTL 30d)   │  stable, batch/daily              │
   │  ├ query_velocity (TTL 1h)    │  streaming, sub-second            │
   │  └ doc_engagement (TTL 24h)   │  batch + stream                   │
   └───────────────┬───────────────┘                                   │
                   │ online lookup < 10ms                              │
                   ▼                                                   │
   USER ──query──▶┌────────────────────────────────────────────────────┴──────┐
                  │  READ PATH — HybridMemoryAgent.recall()                   │
                  │                                                           │
                  │  (a) profile = Feast.get_online_features(user_id)         │
                  │  (b) query_hybrid = BM25 ⊕ vector → RRF(k=60)             │
                  │  (c) re-rank: boost docs khớp topic_affinity              │
                  │  (d) assemble context string                              │
                  └───────────────────────────┬───────────────────────────────┘
                                              ▼
                                   ┌────────────────────┐
                                   │  LLM (final answer)│
                                   └────────────────────┘
```

**Hai nửa trả lời hai câu hỏi khác nhau:**

| Nửa | Câu hỏi | Cơ chế | Latency |
|---|---|---|---|
| Feature store | *"User này là ai?"* | online lookup theo `user_id` | < 10 ms |
| Vector store | *"Cái gì liên quan?"* | hybrid BM25 + ANN + RRF | ~10–30 ms |

Điểm nối là **RRF**: danh sách BM25 và danh sách vector được hợp nhất bằng
`1/(k+rank)`, đúng công thức NB2. Profile không tham gia vào *retrieval* mà
tham gia vào *re-ranking* và *context assembly*.

---

## 2. Ba quyết định kiến trúc (kèm tradeoff)

### QĐ1 — Chunking: **per-message + semantic break ở ranh giới chủ đề**

Tôi chọn **chunk theo message, gộp các message liên tiếp cùng chủ đề thành
một block ~256 token, cắt ở semantic break**, thay vì mỗi message một vector
hoặc mỗi conversation một vector.

- **Per-message:** retrieval rất chính xác (granular) nhưng *storage cost*
  cao gấp N và mỗi mảnh quá ngắn → embedding nhiễu, thiếu ngữ cảnh. Một câu
  "ok, tiếp đi" thành một memory vô nghĩa.
- **Per-conversation:** rẻ, giữ ngữ cảnh, nhưng embedding của cả hội thoại dài
  rơi vào "khoảng giữa" nhiều chủ đề → gần tất cả, thuộc về không cụm nào
  (đúng lỗi NB6 mô tả cho compound query). Đồng thời *context window* bị đốt:
  kéo 1 memory = kéo cả hội thoại.
- **Chọn block ~256 token:** là điểm cân bằng. 256 token vừa đủ để một mảnh
  có *một* ý trọn vẹn (retrieval quality tốt), vừa đủ nhỏ để top-5 nhét gọn
  trong context window, vừa rẻ hơn per-message ~5–8× về số vector.

**Tradeoff tường minh:** *retrieval quality* cao hơn per-conversation, *storage
cost* thấp hơn per-message, *context window* kiểm soát được. Cái mất: chunk
có thể cắt giữa một ý nếu semantic break đoán sai — chấp nhận được vì
overlap 32 token giữa các block bù lại.

### QĐ2 — Feature schema: **tabular features làm nền, embedding feature chỉ cho topic**

| Feature | Entity | TTL | Source | Pattern |
|---|---|---|---|---|
| `preferred_language` | `user` | 30d | batch (profile form) | tabular |
| `reading_speed_wpm` | `user` | 30d | batch (behavioral log) | tabular |
| `topic_affinity` | `user` | 7d | batch (topic counts) | tabular |
| `queries_last_hour` | `user` | 1h | **streaming** | tabular |
| `distinct_topics_24h` | `user` | 1h | **streaming** | tabular |
| `active_hours` | `user` | 30d | batch | tabular |
| `interest_vector` | `user` | 7d | batch (mean of read-doc vectors) | **embedding** |

Tôi chọn **tabular cho gần hết, chỉ một embedding feature**, vì:

- Tabular **giải thích được** và **rẻ**: `topic_affinity` là một string, lookup
  < 10 ms, và khi user hỏi "sao lại recommend cái này" thì trả lời được.
- Embedding feature (`interest_vector`) là *latent preference* — nó bắt được
  sở thích chưa thành nhãn (user đọc nhiều bài cloud nhưng chưa từng nói
  thích cloud). Nhưng nó **không materialize rẻ** và **không debug được**, nên
  chỉ dùng cho re-ranking, không dùng làm nguồn chân lý.

**Tradeoff tường minh:** tabular (simple, explainable, rẻ) vs embedding
(latent, mạnh hơn nhưng mờ). Tôi chọn **tabular-first** vì POC ưu tiên
*clarity*; embedding feature là lớp tăng cường optional, không phải xương sống.
Điều này khớp với lời khuyên trong brief: "don't optimize tốc độ; optimize clarity".

### QĐ3 — Freshness: **phân tầng theo use case, không một ngưỡng chung**

Không có một con số freshness đúng cho mọi thứ — nó phụ thuộc *cái gì đang
được hỏi*. Ba use case, ba mức:

1. **Sub-second (streaming Push API)** — user vừa đọc xong 1 tài liệu, hỏi
   ngay *"tóm tắt cái tôi vừa đọc"*. Đây là **episodic memory**, đi thẳng vào
   Qdrant ở write path, không qua batch. Freshness phải sub-second vì câu trả
   lời *là* tài liệu đó. Profile (`queries_last_hour`) cũng nên streaming.
2. **5 phút (micro-batch)** — *"dạo này tôi quan tâm gì?"*. Topic spike cần
   tổng hợp vài query gần đây; refresh 5 phút là đủ và tránh ghi online store
   liên tục.
3. **Daily (batch)** — *"tôi thích đọc gì?"*. `topic_affinity`,
   `reading_speed_wpm` là slow-moving; tính lại mỗi ngày. Refresh mỗi giây
   chỉ tốn I/O mà không đổi kết quả.

**Tradeoff tường minh:** sub-second (đúng tức thì, nhưng đắt + phức tạp) vs
daily (rẻ, đơn giản, nhưng stale). Tôi **không** chọn một mức cho tất cả —
chọn mức theo *temporal semantics của từng feature*. Đây chính là lý do
`query_velocity_features` có TTL 1h còn `user_profile_features` có TTL 30d
trong `app/feast_repo/feature_views.py`: TTL sai = fraud detection bỏ lỡ tín
hiệu real-time.

---

## 3. Loại bỏ một lựa chọn sai (explicit)

**Tôi đã xem xét lưu episodic memory như một *embedding feature view* ngay
trong feature store, nhưng tách nó ra vector store riêng.** Lý do:

- **Chu kỳ re-index khác hẳn nhau.** Profile đổi theo *tuần* (batch, 100 user
  × vài feature); memory mới mỗi *giờ* (mỗi hội thoại, mỗi tài liệu = một
  vector mới). Nếu nhồi memory vào Feast, mỗi lần `materialize` phải ghi lại
  toàn bộ vector — Feast tối ưu cho *scalar online lookup*, không cho *ANN
  search trên không gian vector*.
- **Truy vấn khác hẳn nhau.** Feature store trả lời "cho tôi giá trị của key
  X"; vector store trả lời "cho tôi top-K gần query Q nhất". Nhồi cái sau vào
  cái trước là bắt một KV-store làm ANN — sai công cụ.
- **Vòng đời khác nhau.** Memory cần TTL/decay/CRUD riêng; profile cần PIT
  join để training không leak.

Nói ngắn: **feature store là nguồn chân lý cho thuộc tính, vector store là
nguồn chân lý cho ngữ nghĩa.** Trộn hai cái làm mất cả hai tính chất.

---

## 4. Vietnamese-context considerations

- **Code-switching (vi/en mix):** user VN thật viết "deploy cái service lên
  k8s", "check giúp tôi cái config". Whitespace tokenizer tách "k8s" và
  "config" ổn, nhưng BM25 coi "service" và "dịch vụ" là hai từ khác nhau →
  hybrid (BM25 ⊕ vector) là bắt buộc, không phải optional. Đây đúng là bài
  học NB2: `mixed` queries là chỗ hybrid thắng rõ nhất.
- **Phonetic typo:** người VN gõ nhanh hay bỏ dấu ("toi muon doc ve cloud").
  bge-m3 chịu được phần nào; một lớp *normalize dấu* trước khi embed giúp
  recall nhưng có thể phá nghĩa ("má" vs "ma"). Tôi chọn **không** normalize
  dấu ở POC, và ghi nhận là limitation.
- **Tokenizer:** whitespace split (nhanh, đủ cho baseline) vs `pyvi` /
  `underthesea` (tách từ ghép tiếng Việt đúng hơn → BM25 chính xác hơn).
  POC dùng whitespace vì corpus nhỏ; production nên dùng `underthesea` cho
  BM25 và **giữ nguyên câu** cho embedding (embedding model đã học ngữ nghĩa
  tốt hơn tokenizer thủ công).
- **Privacy / Nghị định 13:** dữ liệu cá nhân (hội thoại, tài liệu user đọc)
  là dữ liệu cá nhân theo NĐ 13/2023. Mọi truy vấn **phải** filter theo
  `user_id` — đây là isolation **cứng**, không phải soft metadata filter kiểu
  NB7. Rò chéo user trong memory = sự cố pháp lý, không chỉ là bug.

---

## 5. Liên kết với concept trong lab

| Concept lab | Dùng ở đâu trong kiến trúc này |
|---|---|
| **RRF (NB2)** | Hợp nhất BM25 + vector ở read path |
| **Filtered search (NB5)** | `user_id` filter là pre-filter bắt buộc, không post-filter |
| **Agentic retrieval (NB6)** | Planner tách "recommend gì tiếp" thành 2 tool call (profile + vector) |
| **Semantic cache (NB7)** | Cache câu trả lời, nhưng **namespace theo user_id** — không bao giờ `namespaced=False` |
| **TTL (NB4)** | `query_velocity` TTL 1h vs `user_profile` TTL 30d |
| **PIT join (NB4/NB8)** | Training re-ranker: không được dùng memory ghi *sau* nhãn |
| **Streaming (NB6)** | `queries_last_hour` cập nhật sub-second |

---

## 6. What this POC doesn't handle yet

POC này chứng minh *data flow* và *ba quyết định*, không phải một hệ thống
production. Những gì còn thiếu và tôi biết rõ:

- **Privacy isolation thật:** POC filter theo `user_id` trong payload, nhưng
  tất cả memory nằm chung một collection. Multi-tenant thật cần per-user
  collection hoặc encryption per-user — và một filter quên là rò toàn bộ.
- **Encryption at rest / in transit:** chưa có. NĐ 13 yêu cầu điều này.
- **CRUD on memories:** POC chỉ `remember()` (append). Chưa có *forget* /
  *update* / *delete*, cũng chưa có memory decay (TTL cho episodic).
- **Multi-device sync:** memory ghi ở máy A chưa chắc có ở máy B.
- **Consolidation:** chưa gộp 5 memory tương tự thành 1 summary tuần.
- **Re-ranking thật:** hiện chỉ boost theo `topic_affinity`; `interest_vector`
  chưa được dùng để re-rank.
- **Đo lường:** chưa có golden set để đo recall của memory retrieval.

---

## 7. Vibe-coding workflow log (optional)

- **Prompt hiệu quả nhất:** *"Cho tôi interface `HybridMemoryAgent` với 2
  method `remember/recall`, tái dùng `app.search.Searcher` cho hybrid và
  `feast.FeatureStore` cho online lookup; đừng gọi LLM thật, chỉ trả context
  string."* — spec rõ ràng về *interface* và *tái dùng cái có sẵn*, AI viết
  đúng pattern ngay.
- **Prompt fail:** *"Thiết kế memory system cho AI assistant"* — AI trả về
  một bài blog chung chung về RAG, không có tradeoff nào cụ thể. Bài học:
  AI viết *code* tốt khi có spec; *quyết định kiến trúc* vẫn phải tự nghĩ.
