"""NSMC(Naver Sentiment Movie Corpus) 다운로드 + jsonl 변환.

공개 GitHub 레포(e9t/nsmc)에서 탭 구분 텍스트 파일을 받아, 우리 파이프라인이
쓰는 jsonl 포맷({"text": ..., "label": "긍정"|"부정"})으로 바꾼다. API 호출이
아니라 정적 파일 다운로드 한 번이라 속도 제한·봇 차단 걱정이 없다
(startup-sector-classifier에서 겪었던 문제들과 대조됨).

train 15만 개 / test 5만 개, 별점에서 자동 추출된 이진 라벨이라 업종 분류
때와 달리 라벨 중복/애매함 문제도 없다.

사용:
    python scripts/download_data.py --out-dir data/raw
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import requests

BASE = "https://raw.githubusercontent.com/e9t/nsmc/master"
FILES = {"train": "ratings_train.txt", "test": "ratings_test.txt"}
LABEL_NAMES = {"0": "부정", "1": "긍정"}


def convert(raw_text: str) -> list[dict]:
    records = []
    reader = csv.DictReader(raw_text.splitlines(), delimiter="\t")
    for row in reader:
        text = (row.get("document") or "").strip()
        label = row.get("label")
        if not text or label not in LABEL_NAMES:
            continue
        records.append({"text": text, "label": LABEL_NAMES[label]})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split, filename in FILES.items():
        url = f"{BASE}/{filename}"
        print(f"{split} 다운로드 중... ({url})", file=sys.stderr)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        records = convert(resp.text)

        out_path = args.out_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        pos = sum(1 for r in records if r["label"] == "긍정")
        print(f"{split}: {len(records)}개 (긍정 {pos} / 부정 {len(records) - pos}) -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
