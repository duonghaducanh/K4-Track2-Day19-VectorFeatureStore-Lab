# Reflection — Lab 19

**Tên:** _Duong Ha Duc Anh_
**Cohort:** _A20-K4_
**Path đã chạy:** _lite_

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên 50 queries (P@10 trung bình): hybrid 78.6% > BM25 77.8% > vector 73.2%.

Theo lát cắt: `exact` BM25 và hybrid **hoà** (96.7%) — lexical match đã bão hoà
nên vector không thêm tín hiệu. `mixed` hybrid thắng tuyệt đối **100%** (BM25
97.0%, vector 98.5%): RRF cộng được cả hai tín hiệu khi query vừa có keyword vừa
cần ngữ nghĩa. `paraphrase` cả ba đều yếu (24–33%), BM25 nhích nhất: corpus
paraphrase tiếng Việt vượt khả năng của `bge-small-en-v1.5` (train tiếng Anh),
nên vector không bắt được tương đồng.

**Không dùng hybrid khi:** (1) query `exact`/tên riêng/mã sản phẩm — BM25 rẻ,
nhanh, khỏi embed; (2) embedding model lệch ngôn ngữ với dữ liệu (paraphrase
tiếng Việt ở đây) — vector hạng thấp kéo RRF xuống, phải đổi sang bge-m3 trước;
(3) ngân sách latency sub-ms — hai retrieval + fusion không đáng.

---

## Điều ngạc nhiên nhất khi làm lab này

Rò rỉ target-encoding mạnh đến mức phi lý: `session_id` cho train AUC ≈ 0.99
nhưng test ≈ 0.52 — mức rò rỉ tỉ lệ thuận với cardinality của khoá. Và join
"giá trị mới nhất" tạo lift ảo hoàn toàn biến mất khi chuyển sang PIT join.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [x] Pair work với: _không (làm cá nhân)_
