"""
Image Captioning: ViT Encoder + Transformer Decoder
----------------------------------------------------
Architecture:
  - Encoder: google/vit-base-patch16-224 (pretrained, frozen or fine-tuned)
  - Decoder: 4-layer Transformer with cross-attention over ViT patch tokens
  - Vocabulary: built from COCO captions using simple word tokenization
"""

import torch
import torch.nn as nn
from transformers import ViTModel


class CaptionDecoder(nn.Module):
    """
    Autoregressive Transformer decoder that attends to ViT encoder outputs.
    
    Cross-attention: each decoder token attends to all 196 ViT patch tokens.
    Causal mask: ensures token at position t cannot attend to t+1, t+2, ...
    """

    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=4,
                 dim_feedforward=2048, dropout=0.1, max_seq_len=64):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding + positional encoding
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Standard Transformer decoder (cross-attends to encoder memory)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # input shape: (batch, seq, d_model)
            norm_first=True,    # pre-norm (more stable training)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final projection to vocabulary logits
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Dropout on embeddings
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, tgt_tokens, memory, tgt_key_padding_mask=None):
        """
        Args:
            tgt_tokens:           (B, T) — caption token ids (teacher-forced)
            memory:               (B, S, d_model) — ViT patch tokens from encoder
            tgt_key_padding_mask: (B, T) bool, True where padding

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = tgt_tokens.shape
        device = tgt_tokens.device

        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        x = self.dropout(self.token_embed(tgt_tokens) + self.pos_embed(positions))

        # Causal mask: upper-triangular, prevents attending to future tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_is_causal=True,
        )

        return self.output_proj(out)  # (B, T, vocab_size)


class ImageCaptioningModel(nn.Module):
    """
    Full model: ViT encoder (HuggingFace) + custom Transformer decoder.
    
    The ViT produces 196 patch tokens (for 224x224 images with 16x16 patches)
    plus a CLS token — we pass all 197 to the decoder as cross-attention memory.
    A linear projection aligns the ViT hidden size (768) with d_model.
    """

    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=4,
                 dim_feedforward=2048, dropout=0.1, max_seq_len=64,
                 freeze_vit=True):
        super().__init__()

        # ViT encoder (pretrained)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        vit_hidden = self.vit.config.hidden_size  # 768

        if freeze_vit:
            # Freeze all ViT params — only train decoder + projection
            for p in self.vit.parameters():
                p.requires_grad = False
            # Unfreeze the last 2 ViT blocks for partial fine-tuning
            for block in self.vit.layers[-2:]:
                for p in block.parameters():
                    p.requires_grad = True

        # Project ViT dim → decoder d_model (if they differ)
        self.encoder_proj = nn.Linear(vit_hidden, d_model) if vit_hidden != d_model else nn.Identity()

        self.decoder = CaptionDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

    def encode(self, pixel_values):
        """
        Run ViT on an image batch.
        Returns patch token sequences: (B, 197, d_model)
        """
        outputs = self.vit(pixel_values=pixel_values)
        # last_hidden_state: (B, 197, 768) — includes CLS token at index 0
        memory = self.encoder_proj(outputs.last_hidden_state)
        return memory

    def forward(self, pixel_values, tgt_tokens, tgt_key_padding_mask=None):
        """
        Teacher-forced forward pass for training.
        
        Args:
            pixel_values:         (B, 3, 224, 224)
            tgt_tokens:           (B, T) — input caption (with <sos>, without <eos>)
            tgt_key_padding_mask: (B, T) bool
            
        Returns:
            logits: (B, T, vocab_size)
        """
        memory = self.encode(pixel_values)
        return self.decoder(tgt_tokens, memory, tgt_key_padding_mask)

    @torch.no_grad()
    def generate(self, pixel_values, sos_id, eos_id, max_len=40, temperature=1.0, beam_size=1):
        """
        Greedy or beam-search caption generation at inference time.
        Currently implements greedy (beam_size=1) and simple beam search.
        """
        self.eval()
        device = pixel_values.device
        memory = self.encode(pixel_values)  # (B, 197, d_model)
        B = memory.size(0)

        if beam_size == 1:
            return self._greedy_decode(memory, sos_id, eos_id, max_len, temperature, device, B)
        else:
            # Run beam search per image (batch_size=1 assumed for simplicity)
            assert B == 1, "Beam search currently supports batch_size=1"
            return self._beam_search(memory, sos_id, eos_id, max_len, beam_size, device)

    def _greedy_decode(self, memory, sos_id, eos_id, max_len, temperature, device, B):
        tokens = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        captions = [[] for _ in range(B)]

        for _ in range(max_len):
            logits = self.decoder(tokens, memory)           # (B, t, vocab)
            next_logits = logits[:, -1, :] / temperature   # (B, vocab)
            next_token = next_logits.argmax(dim=-1)        # (B,)

            for i in range(B):
                if not finished[i]:
                    if next_token[i].item() == eos_id:
                        finished[i] = True
                    else:
                        captions[i].append(next_token[i].item())

            if finished.all():
                break
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)

        return captions

    def _beam_search(self, memory, sos_id, eos_id, max_len, beam_size, device):
        """Simple beam search for a single image."""
        # Each beam: (score, token_ids_list)
        beams = [(0.0, [sos_id])]
        completed = []

        for _ in range(max_len):
            candidates = []
            for score, tokens in beams:
                if tokens[-1] == eos_id:
                    completed.append((score, tokens))
                    continue
                t = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = self.decoder(t, memory)
                log_probs = torch.log_softmax(logits[0, -1], dim=-1)
                topk_lp, topk_ids = log_probs.topk(beam_size)
                for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                    candidates.append((score + lp, tokens + [tid]))

            if not candidates:
                break
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

        completed += beams
        best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
        caption = [t for t in best[1][1:] if t != eos_id]
        return [caption]
