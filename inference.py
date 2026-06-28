"""
Inference: generate captions for new images using a trained checkpoint.

Usage:
    python inference.py \
        --checkpoint ./checkpoints/best_model.pt \
        --vocab      ./checkpoints/vocab.json \
        --image      /path/to/your/image.jpg \
        --beam_size  3
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms

from model import ImageCaptioningModel
from dataset import Vocabulary


def load_model(checkpoint_path, vocab, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]

    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        d_model=args.get("d_model", 512),
        nhead=args.get("nhead", 8),
        num_decoder_layers=args.get("num_decoder_layers", 4),
        dim_feedforward=args.get("dim_feedforward", 2048),
        dropout=0.0,                     # no dropout at inference
        max_seq_len=args.get("max_seq_len", 64),
        freeze_vit=False,                # weights already loaded
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)   # (1, 3, 224, 224)


def caption_image(model, image_path, vocab, device, beam_size=1):
    pixel_values = preprocess(image_path).to(device)

    with torch.no_grad():
        pred_ids = model.generate(
            pixel_values,
            sos_id=vocab.SOS,
            eos_id=vocab.EOS,
            max_len=40,
            beam_size=beam_size,
        )

    return vocab.decode(pred_ids[0])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab",      required=True)
    p.add_argument("--image",      required=True)
    p.add_argument("--beam_size",  type=int, default=1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(args.vocab)
    model = load_model(args.checkpoint, vocab, device)

    caption = caption_image(model, args.image, vocab, device, args.beam_size)
    print(f"\nGenerated caption: {caption}")
