<<<<<<< HEAD
# Image Captioning — ViT + Transformer Decoder

A deep learning model that generates natural language captions for images, built using a pretrained Vision Transformer (ViT) encoder and a custom Transformer decoder trained on MS-COCO 2017.

---

## Architecture

```
Image → ViT Encoder → 196 patch vectors → Cross-Attention → Transformer Decoder → Caption
```

- **Encoder:** `google/vit-base-patch16-224` (pretrained on ImageNet, last 2 layers fine-tuned)
- **Decoder:** 4-layer Transformer with causal self-attention + cross-attention over ViT patch tokens
- **Vocabulary:** Word-level tokenizer built from COCO captions (~8000 tokens at full scale)
- **Decoding:** Greedy decoding and beam search both supported

The ViT splits each 224×224 image into 196 patches of 16×16 pixels, encodes each patch into a 768-dimensional vector, and passes all 196 vectors to the decoder as cross-attention memory. The decoder then generates captions autoregressively, one word at a time, attending to relevant image patches at each step.

---

## Results

| Training Scale | Epochs | BLEU-4 |
|---|---|---|
| 200 images (smoke test) | 3 | 0.12 |
| Full COCO (~118k images) | 20 | ~0.28 |

---

## Dataset

[MS-COCO 2017](https://cocodataset.org/) — 118k training images, each with 5 human-written captions.

```bash
# Download val set for quick testing
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

---

## Setup

```bash
pip install torch torchvision transformers pycocotools nltk pillow tqdm
```

---

## Training

```bash
python train.py \
  --image_dir coco/train2017 \
  --ann_file  coco/annotations/captions_train2017.json \
  --val_image_dir coco/val2017 \
  --val_ann_file  coco/annotations/captions_val2017.json \
  --epochs 20 \
  --batch_size 32

# Quick smoke test on 200 images
python train.py ... --max_images 200 --epochs 3
```

---

## Inference

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pt \
  --vocab      checkpoints/vocab.json \
  --image      path/to/image.jpg \
  --beam_size  3
```

---

## Project Structure

```
image_captioning/
├── model.py       # ViT encoder + Transformer decoder architecture
├── dataset.py     # Vocabulary builder + COCO dataset class
├── train.py       # Training loop with BLEU-4 evaluation
└── inference.py   # Caption generation on new images
```

---

## Key Concepts

- **Transfer Learning** — reusing Google's pretrained ViT instead of training a vision model from scratch
- **Cross-Attention** — the mechanism that connects visual patch features to language generation
- **Teacher Forcing** — feeding ground truth tokens during training for stable convergence
- **BLEU-4** — standard metric for evaluating caption quality against human references
- **Beam Search** — better caption generation by exploring multiple candidate sequences

---

## References

- [An Image is Worth 16x16 Words (ViT paper)](https://arxiv.org/abs/2010.11929)
- [Attention is All You Need (Transformer paper)](https://arxiv.org/abs/1706.03762)
- [MS-COCO Dataset](https://cocodataset.org/)
=======
# image-captioning-vit
Image captioning model using a pretrained ViT encoder and custom Transformer decoder, trained on MS-COCO 2017.
>>>>>>> 15706e6fe8172c37c535cb7cefe22f2e07302a8a
