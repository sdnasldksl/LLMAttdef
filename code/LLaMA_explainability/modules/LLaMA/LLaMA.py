"""
LRP-aware LLaMA model.

Rebuilds LLaMA architecture using LRP primitives from layers_ours.py so that
relevance can be propagated layer-by-layer back to the input embedding.

Supports loading weights from HuggingFace LlamaForCausalLM (with optional LoRA).
"""

from __future__ import absolute_import

import math
import torch
from torch import nn
import torch.nn.functional as F

from LLaMA_explainability.modules.layers_ours import (
    RelProp, RelPropSimple, safe_divide,
    Linear, Add, Clone, MatMul, Mul, Softmax, Dropout,
    Sequential, IndexSelect, RMSNorm, SiLU,
)


# ─── RoPE (treated as identity for LRP — position encoding doesn't change relevance) ───

class RotaryEmbedding(nn.Module):
    """Precomputes sin/cos tables for RoPE."""
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        return (
            self.cos_cached[:seq_len].unsqueeze(0),
            self.sin_cached[:seq_len].unsqueeze(0),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ─── Self Attention ──────────────────────────────────────────────

class LlamaSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads,
                 head_dim, max_position_embeddings=4096, rope_theta=10000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = head_dim

        self.q_proj = Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta
        )
        self.softmax = Softmax(dim=-1)

        self.matmul_qk = MatMul()
        self.matmul_av = MatMul()
        self.clone = Clone()

        self.attn_cam = None
        self.attn = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.shape

        h1, h2, h3 = self.clone(hidden_states, 3)
        q = self.q_proj(h1)
        k = self.k_proj(h2)
        v = self.v_proj(h3)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(seq_len)
        cos = cos[:, :seq_len, :].to(q.dtype)
        sin = sin[:, :seq_len, :].to(q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = self.matmul_qk([q, k.transpose(-1, -2)])
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = self.softmax(attn_weights)
        attn_probs = attn_probs.to(v.dtype)
        self.save_attn(attn_probs)
        attn_probs.register_hook(self.save_attn_gradients)

        context = self.matmul_av([attn_probs, v])
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.o_proj(context)
        return output

    def relprop(self, cam, **kwargs):
        cam = self.o_proj.relprop(cam, **kwargs)
        bsz, seq_len = cam.shape[0], cam.shape[1]
        cam = cam.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        (cam_attn, cam_v) = self.matmul_av.relprop(cam, **kwargs)
        cam_attn /= 2
        cam_v /= 2

        self.save_attn_cam(cam_attn)

        cam_attn = self.softmax.relprop(cam_attn, **kwargs)

        (cam_q, cam_k) = self.matmul_qk.relprop(cam_attn, **kwargs)
        cam_q /= 2
        cam_k /= 2

        if self.num_kv_groups > 1:
            cam_k = cam_k.reshape(bsz, self.num_kv_heads, self.num_kv_groups, seq_len, -1).sum(dim=2)
            cam_v = cam_v.reshape(bsz, self.num_kv_heads, self.num_kv_groups, seq_len, -1).sum(dim=2)

        cam_q = cam_q.transpose(1, 2).reshape(bsz, seq_len, -1)
        cam_k = cam_k.transpose(1, 2).reshape(bsz, seq_len, -1)
        cam_v = cam_v.transpose(1, 2).reshape(bsz, seq_len, -1)

        cam_q = self.q_proj.relprop(cam_q, **kwargs)
        cam_k = self.k_proj.relprop(cam_k, **kwargs)
        cam_v = self.v_proj.relprop(cam_v, **kwargs)

        cam = self.clone.relprop((cam_q, cam_k, cam_v), **kwargs)
        return cam


# ─── MLP (SwiGLU) ────────────────────────────────────────────────

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiLU()
        self.clone = Clone()
        self.mul = Mul()

    def forward(self, x):
        x1, x2 = self.clone(x, 2)
        gate = self.act_fn(self.gate_proj(x1))
        up = self.up_proj(x2)
        hidden = self.mul([gate, up])
        return self.down_proj(hidden)

    def relprop(self, cam, **kwargs):
        cam = self.down_proj.relprop(cam, **kwargs)
        (cam_gate, cam_up) = self.mul.relprop(cam, **kwargs)
        cam_gate = self.act_fn.relprop(cam_gate, **kwargs)
        cam_gate = self.gate_proj.relprop(cam_gate, **kwargs)
        cam_up = self.up_proj.relprop(cam_up, **kwargs)
        cam = self.clone.relprop((cam_gate, cam_up), **kwargs)
        return cam


# ─── Decoder Layer (pre-norm residual) ───────────────────────────

class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads,
                 head_dim, intermediate_size, rms_norm_eps,
                 max_position_embeddings=4096, rope_theta=10000.0):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = LlamaSelfAttention(
            hidden_size, num_attention_heads, num_key_value_heads,
            head_dim, max_position_embeddings, rope_theta
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

        self.clone_attn = Clone()
        self.add_attn = Add()
        self.clone_mlp = Clone()
        self.add_mlp = Add()

    def forward(self, hidden_states, attention_mask=None):
        # pre-norm residual for attention
        r1, h1 = self.clone_attn(hidden_states, 2)
        h1 = self.input_layernorm(h1)
        attn_out = self.self_attn(h1, attention_mask=attention_mask)
        hidden_states = self.add_attn([r1, attn_out])

        # pre-norm residual for MLP
        r2, h2 = self.clone_mlp(hidden_states, 2)
        h2 = self.post_attention_layernorm(h2)
        mlp_out = self.mlp(h2)
        hidden_states = self.add_mlp([r2, mlp_out])

        return hidden_states

    def relprop(self, cam, **kwargs):
        (cam_r2, cam_mlp) = self.add_mlp.relprop(cam, **kwargs)
        cam_mlp = self.mlp.relprop(cam_mlp, **kwargs)
        cam_mlp = self.post_attention_layernorm.relprop(cam_mlp, **kwargs)
        cam = self.clone_mlp.relprop((cam_r2, cam_mlp), **kwargs)

        (cam_r1, cam_attn) = self.add_attn.relprop(cam, **kwargs)
        cam_attn = self.self_attn.relprop(cam_attn, **kwargs)
        cam_attn = self.input_layernorm.relprop(cam_attn, **kwargs)
        cam = self.clone_attn.relprop((cam_r1, cam_attn), **kwargs)

        return cam


# ─── LlamaModel (embedding + decoder layers + final norm) ───────

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
                head_dim=config.hidden_size // config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                rms_norm_eps=config.rms_norm_eps,
                max_position_embeddings=getattr(config, 'max_position_embeddings', 4096),
                rope_theta=getattr(config, 'rope_theta', 10000.0),
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]

        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device),
                diagonal=1
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def relprop(self, cam, **kwargs):
        cam = self.norm.relprop(cam, **kwargs)
        for layer in reversed(self.layers):
            cam = layer.relprop(cam, **kwargs)
        return cam


# ─── LlamaForCausalLM ────────────────────────────────────────────

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        return logits

    def relprop(self, cam, **kwargs):
        cam = self.lm_head.relprop(cam, **kwargs)
        cam = self.model.relprop(cam, **kwargs)
        return cam

    def clear_hook_data(self):
        """Free all activation tensors saved by forward hooks."""
        for module in self.modules():
            if hasattr(module, 'X'):
                del module.X
            if hasattr(module, 'Y'):
                del module.Y
            if hasattr(module, 'attn'):
                module.attn = None
            if hasattr(module, 'attn_cam'):
                module.attn_cam = None
            if hasattr(module, 'attn_gradients'):
                module.attn_gradients = None

    @classmethod
    def from_pretrained(cls, model_path, device='cpu', dtype=torch.float16):
        """Load weights via HuggingFace official API, same as evaluation scripts."""
        from transformers import LlamaForCausalLM as HFLlama

        hf_model = HFLlama.from_pretrained(
            model_path, torch_dtype=dtype, device_map="cpu"
        )
        lrp_model = cls(hf_model.config)
        lrp_model._apply_state_dict(hf_model.state_dict())
        del hf_model
        return lrp_model.to(dtype).to(device)

    @classmethod
    def from_pretrained_with_lora(cls, base_model_path, lora_path, device='cpu', dtype=torch.float16):
        """Load base model + LoRA via HuggingFace/peft, merge, then transfer."""
        from transformers import LlamaForCausalLM as HFLlama
        from peft import PeftModel

        hf_model = HFLlama.from_pretrained(
            base_model_path, torch_dtype=dtype, device_map="cpu"
        )
        hf_model = PeftModel.from_pretrained(
            hf_model, lora_path, torch_dtype=dtype
        )
        hf_model = hf_model.merge_and_unload()

        lrp_model = cls(hf_model.config)
        lrp_model._apply_state_dict(hf_model.state_dict())
        del hf_model
        return lrp_model.to(dtype).to(device)

    def _apply_state_dict(self, src_dict):
        """Copy weights from HuggingFace state_dict into this LRP model."""
        sd_lrp = self.state_dict()
        loaded = 0
        for key in sd_lrp:
            if key in src_dict:
                if sd_lrp[key].shape == src_dict[key].shape:
                    sd_lrp[key].copy_(src_dict[key])
                    loaded += 1
                else:
                    print(f"Shape mismatch: {key} "
                          f"(LRP {sd_lrp[key].shape} vs src {src_dict[key].shape})")
            else:
                if 'rotary_emb' not in key:
                    print(f"Missing: {key}")
        print(f"Loaded {loaded}/{len(sd_lrp)} parameters into LRP model")
        self.load_state_dict(sd_lrp, strict=False)

    def generate(self, input_ids, max_new_tokens=256, temperature=1.0, top_p=1.0, eos_token_id=None):
        """Simple autoregressive generation (greedy or nucleus sampling)."""
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_logits = logits[:, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float('-inf')
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            if temperature == 1.0 and top_p == 1.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return generated
