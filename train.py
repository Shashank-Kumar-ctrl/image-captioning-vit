"""
Training loop for image captioning model.

Usage:
    python train.py \
        --image_dir /data/coco/train2017 \
        --ann_file  /data/coco/annotations/captions_train2017.json \
        --val_image_dir /data/coco/val2017 \
        --val_ann_file  /data/coco/annotations/captions_val2017.json \
        --output_dir ./checkpoints \
        --epochs 20 \
        --batch_size 32

Quick smoke-test (small subset):
    python train.py ... --max_images 500 --epochs 5
"""

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from model import ImageCaptioningModel
from dataset import Vocabulary, COCOCaptionDataset, collate_fn


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_vocab_from_ann(ann_file, min_freq=5):
    with open(ann_file) as f:
        data = json.load(f)
    captions = [ann["caption"] for ann in data["annotations"]]
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build(captions)
    return vocab


def evaluate_bleu(model, loader, vocab, device, max_batches=50):
    """
    Compute corpus BLEU-4 on a validation loader.
    Greedy decode vs. ground truth captions.
    """
    model.eval()
    references, hypotheses = [], []
    smooth = SmoothingFunction().method1

    with torch.no_grad():
        for i, (pixel_values, input_tokens, target_tokens, pad_mask) in enumerate(loader):
            if i >= max_batches:
                break
            pixel_values = pixel_values.to(device)

            # Greedy caption generation
            pred_ids = model.generate(
                pixel_values, vocab.SOS, vocab.EOS, max_len=40
            )

            for j in range(len(pred_ids)):
                pred_caption = vocab.decode(pred_ids[j]).split()
                # Ground truth from target tokens (strip padding/eos)
                gt_ids = target_tokens[j].tolist()
                gt_caption = vocab.decode(gt_ids).split()

                hypotheses.append(pred_caption)
                references.append([gt_caption])  # corpus_bleu expects list of refs

    bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smooth)
    return bleu4


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    vocab_path = os.path.join(args.output_dir, "vocab.json")

    # ── Vocabulary ──────────────────────────────────────────────────────────
    if os.path.exists(vocab_path):
        print("Loading existing vocabulary...")
        vocab = Vocabulary.load(vocab_path)
    else:
        print("Building vocabulary from training captions...")
        vocab = build_vocab_from_ann(args.ann_file, min_freq=args.min_freq)
        vocab.save(vocab_path)

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = COCOCaptionDataset(
        args.image_dir, args.ann_file, vocab,
        max_seq_len=args.max_seq_len,
        train=True,
        max_images=args.max_images,
    )
    val_dataset = COCOCaptionDataset(
        args.val_image_dir, args.val_ann_file, vocab,
        max_seq_len=args.max_seq_len,
        train=False,
        max_images=args.max_images // 5 if args.max_images else None,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        freeze_vit=args.freeze_vit,
    ).to(device)

    # Separate LRs: smaller for ViT (fine-tuning), larger for decoder
    vit_params = [p for p in model.vit.parameters() if p.requires_grad]
    other_params = list(model.decoder.parameters()) + list(model.encoder_proj.parameters())

    optimizer = AdamW([
        {"params": vit_params,   "lr": args.lr * 0.1},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Ignore padding in loss
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD, label_smoothing=0.1)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_bleu = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, (pixel_values, input_tokens, target_tokens, pad_mask) in enumerate(train_loader):
            pixel_values  = pixel_values.to(device)
            input_tokens  = input_tokens.to(device)
            target_tokens = target_tokens.to(device)
            pad_mask      = pad_mask.to(device)

            # Forward
            logits = model(pixel_values, input_tokens, tgt_key_padding_mask=pad_mask)
            # logits: (B, T, vocab_size) — reshape for CrossEntropyLoss
            loss = criterion(
                logits.reshape(-1, len(vocab)),
                target_tokens.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                avg = total_loss / (step + 1)
                print(f"Epoch {epoch}/{args.epochs}  Step {step}/{len(train_loader)}  "
                      f"Loss: {avg:.4f}")

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        bleu4 = evaluate_bleu(model, val_loader, vocab, device)
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} — Loss: {avg_loss:.4f}  BLEU-4: {bleu4:.4f}\n")

        # Save checkpoint if improved
        if bleu4 > best_bleu:
            best_bleu = bleu4
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "bleu4": bleu4,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved best model (BLEU-4: {bleu4:.4f})")

    print(f"\nTraining complete. Best BLEU-4: {best_bleu:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Paths
    p.add_argument("--image_dir",     required=True)
    p.add_argument("--ann_file",      required=True)
    p.add_argument("--val_image_dir", required=True)
    p.add_argument("--val_ann_file",  required=True)
    p.add_argument("--output_dir",    default="./checkpoints")

    # Training
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--min_freq",    type=int,   default=5)
    p.add_argument("--max_images",  type=int,   default=None,
                   help="Cap dataset size for quick experiments")

    # Model
    p.add_argument("--d_model",            type=int,   default=512)
    p.add_argument("--nhead",              type=int,   default=8)
    p.add_argument("--num_decoder_layers", type=int,   default=4)
    p.add_argument("--dim_feedforward",    type=int,   default=2048)
    p.add_argument("--dropout",            type=float, default=0.1)
    p.add_argument("--max_seq_len",        type=int,   default=64)
    p.add_argument("--freeze_vit",         action="store_true", default=True)

    args = p.parse_args()
    train(args)
