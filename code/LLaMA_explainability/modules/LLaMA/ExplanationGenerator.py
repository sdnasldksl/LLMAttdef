"""
Attribution generator for LLaMA causal LM.

All methods compute attribution **per output token**:
  For each y_t in the response:
    1. Teacher-force the full sequence [prompt, response] in one forward pass
    2. Take the raw logit score logits[t, y_t] as the scalar
    3. Backward from that scalar to get input token attribution
  Finally, average the m attribution vectors across output tokens.

Methods:
  - generate_LRP_last_layer:        last-layer grad × attn cam (paper's partial LRP)
  - generate_gradient_attribution:  ||d logit(y_t) / d embedding_i||
  - generate_LRP:                   grad × attention CAM + rollout (all layers)
  - generate_full_lrp:              full LRP layer-by-layer
"""

import numpy as np
import torch


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [m + eye for m in all_layer_matrices]
    all_layer_matrices = [m / m.sum(dim=-1, keepdim=True) for m in all_layer_matrices]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer + 1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


class Generator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)

    def generate_LRP_last_layer(self, input_ids, response_ids):
        """
        Per-output-token partial LRP — the method used in the original AttDef paper.

        For each y_t:
          1. Forward pass + backward from logits[t, y_t]
          2. Run relprop() (uses saved detached activations, independent of original graph)
          3. Take ONLY the last layer's attn_cam
          4. clamp(min=0), average over heads
        Average over m output tokens.
        """
        prompt_len = input_ids.shape[1]
        response_len = response_ids.shape[1]
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        seq_len = full_ids.shape[1]
        vocab_size = self.model.config.vocab_size

        accumulated = torch.zeros(1, prompt_len, device=input_ids.device)
        kwargs = {"alpha": 1}

        for t in range(response_len):
            self.model.zero_grad()
            logits = self.model(full_ids)

            pred_pos = prompt_len - 1 + t
            target_token = response_ids[0, t].item()

            one_hot = torch.zeros(1, seq_len, vocab_size,
                                  device=input_ids.device, dtype=logits.dtype)
            one_hot[0, pred_pos, target_token] = 1.0

            score = logits[0, pred_pos, target_token]
            score.backward()

            self.model.relprop(one_hot, **kwargs)

            last_layer = self.model.model.layers[-1]
            cam = last_layer.self_attn.get_attn_cam()
            if cam is None:
                cam = last_layer.self_attn.get_attn()
            cam = cam[0].reshape(-1, seq_len, seq_len)
            cam = cam.clamp(min=0).mean(dim=0)

            pred_row = cam[pred_pos, :prompt_len]
            accumulated += pred_row.unsqueeze(0).detach()

            del cam, logits, one_hot, score
            self.model.clear_hook_data()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        attribution = accumulated / response_len
        return attribution

    def generate_gradient_attribution(self, input_ids, response_ids):
        """
        Per-output-token gradient attribution using raw logit scores.

        For each y_t: logit = logits[prompt_len-1+t, y_t]
        → backward → ||∂logit/∂embedding_i|| → average over m tokens.
        """
        prompt_len = input_ids.shape[1]
        response_len = response_ids.shape[1]
        full_ids = torch.cat([input_ids, response_ids], dim=1)

        embeddings = self.model.model.embed_tokens(full_ids)
        embed_input = embeddings.detach().clone()
        embed_input.requires_grad_(True)

        seq_len = full_ids.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=full_ids.device,
                       dtype=embed_input.dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        accumulated = torch.zeros(1, prompt_len, device=input_ids.device)

        for t in range(response_len):
            if embed_input.grad is not None:
                embed_input.grad.zero_()
            self.model.zero_grad()

            hidden = embed_input
            for layer in self.model.model.layers:
                hidden = layer(hidden, attention_mask=causal_mask)
            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden)

            pred_pos = prompt_len - 1 + t
            target_token = response_ids[0, t]
            score = logits[0, pred_pos, target_token]

            score.backward()

            grad = embed_input.grad[:, :prompt_len, :]
            token_attribution = grad.float().norm(dim=-1)
            accumulated += token_attribution.detach()

            del logits, hidden, score
            self.model.clear_hook_data()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        attribution = accumulated / response_len
        return attribution

    def generate_LRP(self, input_ids, response_ids, start_layer=0):
        """
        Per-output-token Grad × Attention CAM + rollout attribution.

        For each y_t:
          1. Forward pass, take logits[t, y_t] as scalar
          2. Backward to fill attention gradients (via register_hook)
          3. grad × cam per layer → clamp(min=0) → mean over heads
          4. Rollout across layers
        Average over m output tokens.
        """
        prompt_len = input_ids.shape[1]
        response_len = response_ids.shape[1]
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        seq_len = full_ids.shape[1]

        accumulated = torch.zeros(1, prompt_len, device=input_ids.device)

        for t in range(response_len):
            self.model.zero_grad()
            logits = self.model(full_ids)

            pred_pos = prompt_len - 1 + t
            target_token = response_ids[0, t]
            score = logits[0, pred_pos, target_token]

            score.backward()

            cams = []
            for blk in self.model.model.layers:
                grad = blk.self_attn.get_attn_gradients()
                cam = blk.self_attn.get_attn_cam()
                if cam is None:
                    cam = blk.self_attn.get_attn()
                cam = cam[0].reshape(-1, seq_len, seq_len)
                grad = grad[0].reshape(-1, seq_len, seq_len)
                cam = (grad * cam).clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0).detach())

            rollout = compute_rollout_attention(cams, start_layer=start_layer)

            pred_row = rollout[0, pred_pos, :prompt_len]
            accumulated += pred_row.unsqueeze(0).detach()

            del cams, rollout, logits, score
            self.model.clear_hook_data()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        attribution = accumulated / response_len
        return attribution

    def generate_full_lrp(self, input_ids, response_ids):
        """
        Per-output-token full LRP layer-by-layer attribution.

        For each y_t:
          1. Forward pass, get logits
          2. Create one-hot relevance seed at logits[t, y_t]
          3. Backward to set gradients, then relprop through all layers
          4. Sum relevance across embedding dim → per-token score
        Average over m output tokens.
        """
        prompt_len = input_ids.shape[1]
        response_len = response_ids.shape[1]
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        seq_len = full_ids.shape[1]
        vocab_size = self.model.config.vocab_size

        accumulated = torch.zeros(1, prompt_len, device=input_ids.device)
        kwargs = {"alpha": 1}

        for t in range(response_len):
            self.model.zero_grad()
            logits = self.model(full_ids)

            pred_pos = prompt_len - 1 + t
            target_token = response_ids[0, t].item()

            one_hot = torch.zeros(1, seq_len, vocab_size,
                                  device=input_ids.device, dtype=logits.dtype)
            one_hot[0, pred_pos, target_token] = logits[0, pred_pos, target_token].detach()

            score = logits[0, pred_pos, target_token]
            score.backward()

            cam = self.model.relprop(one_hot, **kwargs)

            token_relevance = cam[0, :prompt_len, :].sum(dim=-1)
            accumulated += token_relevance.unsqueeze(0).detach()

            del cam, logits, one_hot, score
            self.model.clear_hook_data()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        attribution = accumulated / response_len
        return attribution
