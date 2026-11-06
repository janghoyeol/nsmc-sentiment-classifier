# nsmc-sentiment-classifier

남이 학습시킨 모델 없이, 토크나이저와 작은 GPT를 **처음부터** 만들어 영화 리뷰 긍정/부정을 분류하는 모델

**[데모](https://nsmc-sentiment-classifier.onrender.com)** — 리뷰를 넣으면 긍정·부정과 확률을 보여줌 (Render 무료라 첫 접속은 깨우는 데 시간이 걸림)

## 왜 만들었나

모델이 정말 뭘 배우는지 보고 싶어서
→ 사전학습이 실제로 도움이 되는지, 같은 구조를 사전학습 없이 학습한 것과 비교해 숫자로 확인

## 결과

| | 테스트 정확도 |
| --- | --- |
| 위키백과 사전학습 → 미세조정 | **82.4%** |
| 같은 구조, 사전학습 없이 | 71.2% |

같은 구조·같은 시드·같은 하이퍼파라미터, 백본 초기화만 다름 → 사전학습이 **11.3%p** 끌어올림
(NSMC 테스트 5만 개, 무작위로 찍으면 50%)

위키백과(문어체)로 배운 게 영화 리뷰(구어체·줄임말)에도 옮겨 가는지가 걱정이었는데, 격차가 사라질 정도는 아니었음
→ 자세한 내용: [ADR 0001](docs/decisions/0001-nsmc-and-checkpoint-reuse.md)

## 구조

```
한국어 문장
  → 자모 단위 토크나이저 (직접 구현, src/tokenizer.py)
  → 소형 GPT 백본 (직접 구현, src/model.py)
  → 분류 head → 긍정 / 부정
```

1. **사전학습**: 위키백과 한국어 텍스트로 다음 글자 맞히기
   (이전 프로젝트 [startup-sector-classifier](https://github.com/janghoyeol/startup-sector-classifier)에서 만든 체크포인트 재사용)
2. **미세조정**: NSMC 영화 리뷰 15만 개로 긍정/부정 분류. 백본은 head의 1/10 학습률

## 실행

```bash
pip install -r requirements.txt
python -m pytest tests                       # 토크나이저·모델 테스트

python scripts/download_data.py --out-dir data/raw
python scripts/train_finetune.py --pretrained checkpoints/pretrain.pt \
    --vocab artifacts/vocab.json --out checkpoints/finetune_pretrained.pt   # 사전학습 백본으로
python scripts/train_finetune.py \
    --vocab artifacts/vocab.json --out checkpoints/finetune_scratch.pt      # 비교군 (사전학습 없이)
```

웹: `uvicorn web.api:app` (학습된 모델은 Hugging Face에서 받아 옴, `render.yaml`)

## 기술

Python · PyTorch · FastAPI · Docker · Render · 학습은 Kaggle GPU
