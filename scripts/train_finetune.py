"""NSMC 감성분석(긍정/부정) 미세조정.

startup-sector-classifier와 같은 비교 실험 구조를 재사용한다 — `--pretrained`로
(그 프로젝트에서 만든) 1단계 체크포인트를 주면 그 백본으로 시작하고, 안 주면
같은 구조를 랜덤 초기화부터 바로 분류만 학습(베이스라인). 백본/분류head는
차등 학습률을 쓴다(사전학습 썼을 때 백본은 head의 1/10 속도) — 업종분류
프로젝트에서 이게 핵심이었다.

NSMC는 train/test가 이미 공식적으로 나뉘어 있어서(각각 15만/5만 개), 그걸
그대로 쓴다 — 자체적으로 섞어서 split할 필요가 없다. train 안에서 다시
일부를 떼어 val로 써서 epoch마다 모니터링하고, 최종 선택된 체크포인트만
test로 한 번 평가한다(과적합 모니터링용 val과 최종 성능 보고용 test를
분리 — startup-sector-classifier는 데이터가 적어 이 구분을 안 뒀었다).

사용:
    python scripts/train_finetune.py --pretrained checkpoints/pretrain.pt \
        --vocab artifacts/vocab.json \
        --out checkpoints/finetune_pretrained.pt \
        --metrics-out artifacts/metrics_pretrained.json

    python scripts/train_finetune.py \
        --vocab artifacts/vocab.json \
        --out checkpoints/finetune_scratch.pt \
        --metrics-out artifacts/metrics_scratch.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import GPTClassifier, GPTConfig  # noqa: E402
from tokenizer import JamoTokenizer  # noqa: E402


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def encode_records(
    records: list[dict], tokenizer: JamoTokenizer, label_map: dict[str, int], max_len: int
) -> list[tuple[list[int], int]]:
    data = []
    for r in records:
        ids = tokenizer.encode(r["text"], add_bos_eos=False)[:max_len]
        if not ids:
            continue
        data.append((ids, label_map[r["label"]]))
    return data


def split_train_val(data: list, val_fraction: float, seed: int) -> tuple[list, list]:
    idx = list(range(len(data)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(idx) * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [data[i] for i in train_idx], [data[i] for i in val_idx]


def iter_batches(data: list, batch_size: int, pad_id: int, device: str, shuffle: bool, rng: random.Random):
    order = list(range(len(data)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        batch = [data[i] for i in order[start : start + batch_size]]
        max_len = max(len(ids) for ids, _ in batch)
        x = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        lengths = torch.zeros(len(batch), dtype=torch.long)
        y = torch.zeros(len(batch), dtype=torch.long)
        for i, (ids, label) in enumerate(batch):
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            lengths[i] = len(ids)
            y[i] = label
        yield x.to(device), lengths.to(device), y.to(device)


@torch.no_grad()
def evaluate(model: GPTClassifier, data: list, batch_size: int, pad_id: int, device: str) -> dict:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    rng = random.Random(0)
    for x, lengths, y in iter_batches(data, batch_size, pad_id, device, shuffle=False, rng=rng):
        logits, loss = model(x, lengths, y)
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += len(y)
    model.train()
    return {"loss": loss_sum / total, "accuracy": correct / total, "n": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", type=Path, default=Path("data/raw/train.jsonl"))
    parser.add_argument("--test-data", type=Path, default=Path("data/raw/test.jsonl"))
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/finetune.pt"))
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--max-train", type=int, default=None, help="디버깅용: train 샘플 수 제한")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = JamoTokenizer.load(args.vocab)
    train_records = load_records(args.train_data)
    if args.max_train:
        random.Random(args.seed).shuffle(train_records)
        train_records = train_records[: args.max_train]
    test_records = load_records(args.test_data)

    labels = sorted({r["label"] for r in train_records})
    label_map = {label: i for i, label in enumerate(labels)}
    print(f"라벨: {label_map}", file=sys.stderr)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    if args.pretrained:
        ckpt_config = torch.load(args.pretrained, map_location="cpu", weights_only=False)["config"]
        if ckpt_config.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"체크포인트 vocab_size({ckpt_config.vocab_size})와 "
                f"--vocab의 vocab_size({tokenizer.vocab_size})가 다릅니다."
            )
        config = GPTConfig(
            vocab_size=ckpt_config.vocab_size,
            block_size=ckpt_config.block_size,
            n_layer=ckpt_config.n_layer,
            n_head=ckpt_config.n_head,
            n_embd=ckpt_config.n_embd,
            dropout=args.dropout,
        )

    if args.max_len > config.block_size:
        print(f"--max-len을 block_size({config.block_size})로 줄입니다.", file=sys.stderr)
        args.max_len = config.block_size

    all_train = encode_records(train_records, tokenizer, label_map, args.max_len)
    train_data, val_data = split_train_val(all_train, args.val_fraction, args.seed)
    test_data = encode_records(test_records, tokenizer, label_map, args.max_len)
    print(f"train {len(train_data)} / val {len(val_data)} / test {len(test_data)}", file=sys.stderr)

    model = GPTClassifier(config, num_classes=len(label_map)).to(args.device)

    if args.pretrained:
        result = model.load_pretrained_backbone(args.pretrained, device=args.device)
        print(f"사전학습 백본 로드: missing={result['missing']} unexpected={result['unexpected']}", file=sys.stderr)
    else:
        print("사전학습 없이 랜덤 초기화부터 시작 (베이스라인)", file=sys.stderr)

    print(f"모델 파라미터 수: {model.num_params():,} (device={args.device})", file=sys.stderr)

    if args.backbone_lr is not None:
        backbone_lr = args.backbone_lr
    elif args.pretrained:
        backbone_lr = args.lr / 10
    else:
        backbone_lr = args.lr

    head_params = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    print(f"학습률: head={args.lr} backbone={backbone_lr}", file=sys.stderr)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": args.lr},
        ]
    )
    rng = random.Random(args.seed)

    best_val_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        for x, lengths, y in iter_batches(train_data, args.batch_size, tokenizer.pad_id, args.device, True, rng):
            _, loss = model(x, lengths, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, train_data, args.batch_size, tokenizer.pad_id, args.device)
        val_metrics = evaluate(model, val_data, args.batch_size, tokenizer.pad_id, args.device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_loss {train_metrics['loss']:.4f} train_acc {train_metrics['accuracy']:.3f}  "
            f"val_loss {val_metrics['loss']:.4f} val_acc {val_metrics['accuracy']:.3f}",
            file=sys.stderr,
        )

        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "label_map": label_map,
                    "epoch": epoch,
                    "val_accuracy": val_metrics["accuracy"],
                },
                args.out,
            )

    print(f"최고 val 정확도: {best_val_acc:.4f} (체크포인트: {args.out})", file=sys.stderr)

    best = torch.load(args.out, map_location=args.device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_metrics = evaluate(model, test_data, args.batch_size, tokenizer.pad_id, args.device)
    print(f"최종 test 정확도 (최고 val 체크포인트 기준): {test_metrics['accuracy']:.4f}", file=sys.stderr)

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(
            json.dumps(
                {
                    "pretrained": str(args.pretrained) if args.pretrained else None,
                    "best_val_accuracy": best_val_acc,
                    "test_accuracy": test_metrics["accuracy"],
                    "n_train": len(train_data),
                    "n_val": len(val_data),
                    "n_test": len(test_data),
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
