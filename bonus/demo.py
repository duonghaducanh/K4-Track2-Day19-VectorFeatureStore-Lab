"""5-query demo for HybridMemoryAgent (bonus).

Run:  python bonus/demo.py        (exits 0 on success)

Seeds a handful of episodic memories for one user, then walks the five
scenarios from the brief. Prints the assembled context for each -- no LLM
call, which is the point: the hard part is *retrieval + assembly*, not
generation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent  # noqa: E402

USER = "u_001"

# (text, topic) -- a plausible reading history for a cloud-leaning user.
MEMORIES = [
    ("Đọc bài về Kubernetes: Pod lifecycle, Deployment, tự động mở rộng "
     "theo lưu lượng với HorizontalPodAutoscaler.", "cloud"),
    ("Tài liệu AWS: tối ưu chi phí hạ tầng bằng spot instance và "
     "auto-scaling group, tách biệt môi trường dev và prod.", "cloud"),
    ("Ghi chú: cloud security cơ bản -- mã hoá dữ liệu khi lưu trữ, "
     "xác thực hai yếu tố, quản lý secret bằng vault.", "security"),
    ("Bài về cân bằng tải giữa nhiều region, giảm độ trễ end-to-end "
     "cho người dùng ở Việt Nam.", "networking"),
    ("Hướng dẫn tối ưu truy vấn PostgreSQL với chỉ mục B-tree.", "database"),
    ("Đọc về mô hình ngôn ngữ lớn và cách phát hiện hallucination.", "ai_ml"),
]

QUERIES = [
    ("1. simple (vector only)",  "Tôi đã đọc gì về Kubernetes?"),
    ("2. profile needed",        "Recommend đọc gì tiếp"),
    ("3. fresh activity",        "Tôi đang quan tâm gì gần đây?"),
    ("4. paraphrase (vector)",   "Tài liệu về tự động mở rộng hạ tầng?"),
    ("5. mixed (hybrid+profile)", "Cho tôi summary cloud security"),
]


def main() -> int:
    agent = HybridMemoryAgent()

    print("=" * 70)
    print("HybridMemoryAgent demo -- seeding episodic memory")
    print("=" * 70)
    for text, topic in MEMORIES:
        agent.remember(text, user_id=USER, topic=topic)
    print(f"seeded {len(MEMORIES)} memories for {USER}\n")

    for label, query in QUERIES:
        print("=" * 70)
        print(f"QUERY {label}: {query!r}")
        print("=" * 70)
        print(agent.recall(query, user_id=USER))
        print()

    # Privacy check: another user must NOT see u_001's memories.
    other = agent.recall("Kubernetes là gì?", user_id="u_999")
    leaked = "Kubernetes" in other and "Pod lifecycle" in other
    print("=" * 70)
    print(f"isolation check: u_999 sees u_001 memory? {leaked}  (must be False)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
