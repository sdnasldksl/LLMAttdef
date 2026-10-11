"""
AttDef defense pipeline adapted for LLaMA text generation models.

Three-phase approach:
  Phase 0: (Optional) ELECTRA pre-screening — flag suspicious samples.
  Phase 1: Compute subtoken-level attribution (expensive, done once).
           Aggregate subtokens → words (mean pooling).
           Normalize word-level scores over data words only.
  Phase 2: Sweep thresholds 0.05~0.95, for each threshold:
           - ELECTRA gate: only mask triggers for flagged samples
           - Identify suspicious words, mask with [MASK]
           - Compute word-level P/R/F1/BRR (with symbol filtering)
           - Save per-threshold details + overview summary
"""

import os
import re
import sys
import json
import argparse
import random
import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import ElectraForPreTraining, ElectraTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from LLaMA_explainability.modules.LLaMA.LLaMA import LlamaForCausalLM
from LLaMA_explainability.modules.LLaMA.ExplanationGenerator import Generator

TEMPLATES = {
    "llama2_chat": {
        "prefix": (
            "<s>[INST] "
            "<<SYS>>\n"
            "You are a helpful, respectful and honest assistant.\n"
            "<</SYS>>\n\n"
        ),
        "suffix": " [/INST] ",
    },
    "alpaca": {
        "prefix_with_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "{instruction}\n\n"
            "### Input:\n"
            "{input}\n\n"
            "### Response:\n"
        ),
        "prefix_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "{instruction}\n\n"
            "### Response:\n"
        ),
        "suffix": "",
    },
}

_PUNCT_RE = re.compile(r'^[^\w]+$')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────── ELECTRA ───────────────────────────

def electra_check_corpus(text_list, electra_model, electra_cache, device, log_fn):
    """ELECTRA pre-screening: flag samples with any token logit >= 0."""
    log_fn("Loading ELECTRA discriminator...")
    discriminator = ElectraForPreTraining.from_pretrained(
        electra_model, cache_dir=electra_cache
    ).to(device)
    dis_tokenizer = ElectraTokenizerFast.from_pretrained(
        electra_model, cache_dir=electra_cache
    )
    discriminator.eval()
    check_state, check_cnt = [], 0
    with torch.no_grad():
        for text in tqdm(text_list, desc="ELECTRA check"):
            input_ids = dis_tokenizer.encode(
                text, truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            preds = discriminator(input_ids)[0].detach().cpu().numpy()[0]
            flagged = bool(any(p >= 0 for p in preds))
            check_state.append(flagged)
            check_cnt += int(flagged)
    del discriminator, dis_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_fn(f"ELECTRA check done: {check_cnt}/{len(text_list)} flagged "
           f"({check_cnt / len(text_list) * 100:.1f}%)")
    return check_state


# ─────────────────────────── Data helpers ──────────────────────

def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data


def build_user_content(item):
    instruction = str(item.get("instruction", "") or "").strip()
    input_text = str(item.get("input", "") or "").strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def build_full_prompt(item, template_name):
    instruction = str(item.get("instruction", "") or "").strip()
    input_text = str(item.get("input", "") or "").strip()
    if template_name == "llama2_chat":
        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        tmpl = TEMPLATES["llama2_chat"]
        return tmpl["prefix"] + user_content + tmpl["suffix"]
    elif template_name == "alpaca":
        tmpl = TEMPLATES["alpaca"]
        if input_text:
            return tmpl["prefix_with_input"].format(
                instruction=instruction, input=input_text)
        else:
            return tmpl["prefix_no_input"].format(instruction=instruction)
    else:
        raise ValueError(f"Unknown template: {template_name}")


# ─────────────────────── Subtoken attribution ─────────────────

def compute_attribution(generator, tokenizer, full_prompt, response_text, method, device):
    """Compute per-subtoken attribution (returns numpy array + token list)."""
    prompt_ids = tokenizer.encode(full_prompt, return_tensors='pt',
                                  add_special_tokens=False).to(device)
    response_ids = tokenizer.encode(response_text, return_tensors='pt',
                                    add_special_tokens=False).to(device)
    if response_ids.shape[1] == 0:
        return np.zeros(prompt_ids.shape[1]), \
               tokenizer.convert_ids_to_tokens(prompt_ids[0].tolist())
    with torch.enable_grad():
        if method == 'partial_lrp':
            attr = generator.generate_LRP_last_layer(prompt_ids, response_ids)
        elif method == 'gradient':
            attr = generator.generate_gradient_attribution(prompt_ids, response_ids)
        elif method == 'lrp':
            attr = generator.generate_LRP(prompt_ids, response_ids)
        elif method == 'full_lrp':
            attr = generator.generate_full_lrp(prompt_ids, response_ids)
        else:
            raise ValueError(f"Unknown method: {method}")
    attr = attr[0].detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(prompt_ids[0].tolist())
    return attr, tokens


# ───────────────── Subtoken → Word aggregation ────────────────

def aggregate_subtokens_to_words(attr, tokens, full_prompt, tokenizer):
    """
    Map subtoken-level attribution to space-split words of the prompt.
    Uses exact prefixes of full_prompt (preserving \\n etc.) for tokenization
    so that subtoken counts match the actual encoding.
    Returns (prompt_words, word_scores) where word_scores[i] = mean(subtokens of word i).
    """
    prompt_words = full_prompt.split()
    word_scores = np.zeros(len(prompt_words))

    search_from = 0
    prev_ntok = 0
    for wi, word in enumerate(prompt_words):
        pos = full_prompt.find(word, search_from)
        if pos < 0:
            pos = search_from
        end_pos = pos + len(word)

        prefix = full_prompt[:end_pos]
        cur_ntok = len(tokenizer.encode(prefix, add_special_tokens=False))

        start = prev_ntok
        end = min(cur_ntok, len(attr))
        if start < end:
            word_scores[wi] = float(np.mean(attr[start:end]))
        prev_ntok = cur_ntok
        search_from = end_pos

    return prompt_words, word_scores


def _find_word_sequence(prompt_words, target_words, search_start=0):
    """Find the start index of target_words in prompt_words using sliding window."""
    if not target_words:
        return -1
    match_len = min(3, len(target_words))
    target = [w.lower().strip('.,!?;:\'"') for w in target_words[:match_len]]
    for start in range(search_start, len(prompt_words) - match_len + 1):
        window = [prompt_words[start + j].lower().strip('.,!?;:\'"')
                  for j in range(match_len)]
        if window == target:
            return start
    # Fallback: try matching just the first word
    if match_len > 1:
        t0 = target_words[0].lower().strip('.,!?;:\'"')
        for start in range(search_start, len(prompt_words)):
            if prompt_words[start].lower().strip('.,!?;:\'"') == t0:
                return start
    return -1


def get_data_word_indices(prompt_words, instruction, input_text):
    """
    Return set of word indices in prompt_words that belong to data
    (instruction + input), excluding template words.
    """
    inst_words = instruction.split()
    inp_words = input_text.split() if input_text else []
    data_indices = set()

    if inst_words:
        pos = _find_word_sequence(prompt_words, inst_words)
        if pos >= 0:
            for j in range(len(inst_words)):
                if pos + j < len(prompt_words):
                    data_indices.add(pos + j)

    if inp_words:
        search_after = max(data_indices) + 1 if data_indices else 0
        pos = _find_word_sequence(prompt_words, inp_words, search_after)
        if pos >= 0:
            for j in range(len(inp_words)):
                if pos + j < len(prompt_words):
                    data_indices.add(pos + j)

    return data_indices


# ────────────────────── Trigger matching ──────────────────────

def find_trigger_word_indices(words, trigger_str):
    """
    Find trigger word positions in a word list.
    Supports comma-separated (CTBA: "BadMagic,OpenAI,2024")
    and phrase ("I watched a movie last night").
    """
    if not trigger_str:
        return set()

    words_lower = [w.lower().strip('.,!?;:\'"') for w in words]

    if ',' in trigger_str:
        trigger_words = [t.strip().lower() for t in trigger_str.split(',') if t.strip()]
        indices = set()
        for tw in trigger_words:
            for i, w in enumerate(words_lower):
                if tw == w:
                    indices.add(i)
        return indices
    else:
        trigger_words = [t.strip('.,!?;:\'"') for t in trigger_str.lower().split()]
        indices = set()
        for start in range(len(words_lower) - len(trigger_words) + 1):
            matched = True
            for j, tw in enumerate(trigger_words):
                if tw != words_lower[start + j]:
                    matched = False
                    break
            if matched:
                for j in range(len(trigger_words)):
                    indices.add(start + j)
        return indices


# ──────────────── Localization metrics (word-level) ───────────

def compute_localization_metrics(pred_indices, true_indices, data_indices, prompt_words):
    """
    Word-level P / R / F1 / BRR with punctuation-only words filtered out.
    """
    punct = set(i for i in (pred_indices | true_indices | data_indices)
                if _PUNCT_RE.match(prompt_words[i]))
    pred = pred_indices - punct
    true = true_indices - punct
    data_clean = data_indices - punct

    if not true:
        return None

    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    total_benign = len(data_clean - true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    brr = fp / total_benign if total_benign > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "brr": brr,
            "tp": tp, "fp": fp, "fn": fn}


# ──────────────────── Prompt masking (word-level) ─────────────

def mask_words_and_rebuild(prompt_words, suspicious_indices, mask_token="<unk>"):
    """Replace suspicious words with mask_token, return cleaned prompt string."""
    exclude = set(suspicious_indices)
    cleaned = [mask_token if i in exclude else w
               for i, w in enumerate(prompt_words)]
    return ' '.join(cleaned)


# ══════════════════════════════ MAIN ══════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AttDef defense for LLaMA (word-level)")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, required=True,
                        help="Directory containing trigger.json and/or no_trigger.json")
    parser.add_argument("--output_dir", type=str, default="./attdef_results")
    parser.add_argument("--method", type=str, default="gradient",
                        choices=["partial_lrp", "gradient", "lrp", "full_lrp"])
    parser.add_argument("--template", type=str, default="llama2_chat",
                        choices=list(TEMPLATES.keys()))
    parser.add_argument("--trigger", type=str, default="",
                        help="Trigger text for evaluation. Comma-separated for CTBA.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_token", type=str, default="<unk>")
    parser.add_argument("--use_electra", action="store_true")
    parser.add_argument("--electra_model", type=str,
                        default="google/electra-large-discriminator")
    parser.add_argument("--electra_cache", type=str, default=None)
    parser.add_argument("--electra_check_path", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    log_file = open(os.path.join(args.output_dir, "log.txt"), 'w')

    def log(msg):
        print(msg)
        print(msg, file=log_file)
        log_file.flush()

    log(f"Args: {vars(args)}")
    log(f"Device: {device}")

    # ── 1. Load model ──
    log("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.lora_path:
        model = LlamaForCausalLM.from_pretrained_with_lora(
            args.base_model, args.lora_path, device=device)
    else:
        model = LlamaForCausalLM.from_pretrained(args.base_model, device=device)
    model.eval()
    generator = Generator(model)
    log("Model loaded.")

    # ── 2. Load data ──
    data = []
    trigger_path = os.path.join(args.data_root, "trigger.json")
    no_trigger_path = os.path.join(args.data_root, "no_trigger.json")
    if os.path.exists(trigger_path):
        poison_data = load_data(trigger_path)
        for item in poison_data:
            item["is_poison"] = True
        data.extend(poison_data)
        log(f"Loaded {len(poison_data)} poison samples from {trigger_path}")
    if os.path.exists(no_trigger_path):
        clean_data = load_data(no_trigger_path)
        for item in clean_data:
            item["is_poison"] = False
        data.extend(clean_data)
        log(f"Loaded {len(clean_data)} clean samples from {no_trigger_path}")
    if not data:
        raise FileNotFoundError(
            f"No trigger.json or no_trigger.json found in {args.data_root}")
    if args.max_samples > 0:
        poison_part = [d for d in data if d.get("is_poison", False)]
        clean_part  = [d for d in data if not d.get("is_poison", False)]
        poison_part = poison_part[:args.max_samples]
        clean_part  = clean_part[:args.max_samples]
        data = poison_part + clean_part
    n_poison = sum(1 for d in data if d.get("is_poison", False))
    n_clean = len(data) - n_poison
    log(f"Total: {len(data)} samples (poison={n_poison}, clean={n_clean})")
    log(f"Trigger: '{args.trigger}'")

    # ══════════════════════════════════════════════════════════════
    # Phase 0: ELECTRA pre-screening
    # ══════════════════════════════════════════════════════════════
    if args.use_electra:
        log("\n" + "=" * 60)
        log("Phase 0: ELECTRA pre-screening")
        log("=" * 60)
        user_texts = [build_user_content(item) for item in data]
        if args.electra_check_path and os.path.exists(args.electra_check_path):
            log(f"Loading cached ELECTRA states from {args.electra_check_path}")
            with open(args.electra_check_path, 'rb') as fp:
                check_state = pickle.load(fp)
            flagged = sum(check_state)
            log(f"Loaded {len(check_state)} states, {flagged} flagged "
                f"({flagged / len(check_state) * 100:.1f}%)")
        else:
            check_state = electra_check_corpus(
                user_texts, args.electra_model, args.electra_cache, device, log)
            if args.electra_check_path:
                ck_dir = os.path.dirname(args.electra_check_path)
                if ck_dir:
                    os.makedirs(ck_dir, exist_ok=True)
                with open(args.electra_check_path, 'wb') as fp:
                    pickle.dump(check_state, fp, protocol=pickle.HIGHEST_PROTOCOL)
                log(f"ELECTRA states cached to {args.electra_check_path}")
    else:
        check_state = [True] * len(data)
        log("ELECTRA disabled — all samples treated as potentially poisoned")

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Subtoken attribution → Word-level aggregation
    # ══════════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("Phase 1: Computing attribution + word-level aggregation...")
    log("=" * 60)

    sample_records = []

    for idx, item in enumerate(tqdm(data, desc="Attribution")):
        instruction = str(item.get("instruction", "") or "").strip()
        input_text = str(item.get("input", "") or "").strip()
        full_prompt = build_full_prompt(item, args.template)

        # 1a. Generate response
        prompt_ids = tokenizer.encode(full_prompt, return_tensors='pt',
                                      add_special_tokens=False).to(device)
        gen_ids = model.generate(
            prompt_ids, max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id)
        response_ids = gen_ids[:, prompt_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids[0].detach(),
                                         skip_special_tokens=True)
        del prompt_ids, gen_ids, response_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 1b. Subtoken-level attribution
        attr_all, tokens_all = compute_attribution(
            generator, tokenizer, full_prompt, response_text,
            args.method, device)
        model.clear_hook_data()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 1c. Aggregate subtokens → words (mean pooling)
        word_texts, word_scores = aggregate_subtokens_to_words(
            attr_all, tokens_all, full_prompt, tokenizer)

        # 1d. Identify data word indices (template excluded)
        data_word_idx = get_data_word_indices(word_texts, instruction, input_text)

        # 1e. Min-max normalize over data words only
        if data_word_idx:
            data_scores = [word_scores[i] for i in data_word_idx]
            lo, hi = min(data_scores), max(data_scores)
            if hi - lo < 1e-12:
                norm_scores = np.zeros(len(word_scores))
            else:
                norm_scores = np.array([(word_scores[i] - lo) / (hi - lo)
                                        if i in data_word_idx else 0.0
                                        for i in range(len(word_scores))])
        else:
            norm_scores = np.zeros(len(word_scores))

        # 1f. Find true trigger indices (for evaluation)
        true_trigger_idx = find_trigger_word_indices(word_texts, args.trigger) \
            if args.trigger else set()

        sample_records.append({
            "idx": idx,
            "item": item,
            "is_poison": item.get("is_poison", False),
            "instruction": instruction,
            "input_text": input_text,
            "full_prompt": full_prompt,
            "response_text": response_text,
            "word_texts": word_texts,
            "word_scores": word_scores,
            "norm_scores": norm_scores,
            "data_word_idx": data_word_idx,
            "true_trigger_idx": true_trigger_idx,
        })

        log(f"  [{idx}] words={len(word_texts)}, data_words={len(data_word_idx)}, "
            f"trigger_words={len(true_trigger_idx)}")
        log(f"       prompt_words[:5]: {word_texts[:5]}")
        log(f"       inst_words[:5]:   {instruction.split()[:5]}")
        dw = sorted(data_word_idx)[:10]
        if dw:
            log(f"       data words: {[word_texts[i] for i in dw]}...")
            log(f"       norm_scores: {[round(norm_scores[i], 3) for i in dw]}...")
        else:
            log(f"       WARNING: no data words found!")

    log(f"Phase 1 complete. {len(sample_records)} samples.\n")

    # Save attributions
    attr_out = []
    for rec in sample_records:
        attr_out.append({
            "idx": rec["idx"],
            "instruction": rec["instruction"][:200],
            "input": rec["input_text"][:200],
            "response": rec["response_text"][:200],
            "word_texts": rec["word_texts"],
            "norm_scores": rec["norm_scores"].tolist(),
            "data_word_idx": sorted(rec["data_word_idx"]),
            "true_trigger_idx": sorted(rec["true_trigger_idx"]),
        })
    with open(os.path.join(args.output_dir, "attributions.json"), 'w',
              encoding='utf-8') as f:
        json.dump(attr_out, f, ensure_ascii=False, indent=2)
    log("Attribution scores saved.")

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Sweep thresholds (word-level)
    # ══════════════════════════════════════════════════════════════
    thresholds = np.arange(0.05, 1.0, 0.05)
    log("\n" + "=" * 60)
    log(f"Phase 2: Sweeping {len(thresholds)} thresholds (word-level)")
    log("=" * 60)

    overview = []

    for thr in thresholds:
        thr_val = round(float(thr), 2)
        all_metrics = []
        per_sample = []

        for rec in sample_records:
            flagged = check_state[rec["idx"]]

            if flagged:
                suspicious = set(i for i in rec["data_word_idx"]
                                 if rec["norm_scores"][i] > thr_val)
            else:
                suspicious = set()

            detected_words = [rec["word_texts"][i] for i in sorted(suspicious)]

            cleaned_prompt = mask_words_and_rebuild(
                rec["word_texts"], suspicious, args.mask_token)

            metrics = None
            if rec["is_poison"] and args.trigger and rec["true_trigger_idx"]:
                metrics = compute_localization_metrics(
                    suspicious, rec["true_trigger_idx"],
                    rec["data_word_idx"], rec["word_texts"])

            if metrics:
                all_metrics.append(metrics)

            per_sample.append({
                "idx": rec["idx"],
                "is_poison": rec["is_poison"],
                "instruction": rec["instruction"],
                "input": rec["input_text"],
                "electra_flagged": flagged,
                "original_prompt": rec["full_prompt"],
                "cleaned_prompt": cleaned_prompt,
                "detected_triggers": detected_words,
                "num_detected": len(suspicious),
                "num_data_words": len(rec["data_word_idx"]),
            })

        # Clean sample removal stats
        clean_samples = [s for s in per_sample if not s["is_poison"]]
        if clean_samples:
            clean_avg_del = sum(s["num_detected"] for s in clean_samples) / len(clean_samples)
            clean_del_ratios = [s["num_detected"] / max(s["num_data_words"], 1)
                                for s in clean_samples]
            clean_avg_ratio = sum(clean_del_ratios) / len(clean_del_ratios)
        else:
            clean_avg_del = 0.0
            clean_avg_ratio = 0.0

        # Aggregate metrics
        thr_summary = {
            "threshold": thr_val,
            "total_detected": sum(s["num_detected"] for s in per_sample),
            "avg_detected": round(
                sum(s["num_detected"] for s in per_sample) / max(len(per_sample), 1), 2),
            "clean_count": len(clean_samples),
            "clean_avg_del": round(clean_avg_del, 4),
            "clean_avg_ratio": round(clean_avg_ratio, 4),
        }
        if all_metrics:
            thr_summary["precision"] = round(float(np.mean([m["precision"] for m in all_metrics])), 4)
            thr_summary["recall"] = round(float(np.mean([m["recall"] for m in all_metrics])), 4)
            thr_summary["f1"] = round(float(np.mean([m["f1"] for m in all_metrics])), 4)
            thr_summary["brr"] = round(float(np.mean([m["brr"] for m in all_metrics])), 4)
            thr_summary["evaluated_samples"] = len(all_metrics)
        overview.append(thr_summary)

        # Save per-threshold details
        detail_path = os.path.join(
            args.output_dir, f"attdef_details_thr_{thr_val:.2f}.json")
        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(per_sample, f, ensure_ascii=False, indent=2)

        # Log
        line = (f"  thr={thr_val:.2f}  detected={thr_summary['total_detected']}"
                f"  avg={thr_summary['avg_detected']:.2f}"
                f"  clean_del={clean_avg_del:.2f}"
                f"  clean_ratio={clean_avg_ratio:.2%}")
        if all_metrics:
            line += (f"  P={thr_summary['precision']:.4f}"
                     f"  R={thr_summary['recall']:.4f}"
                     f"  F1={thr_summary['f1']:.4f}"
                     f"  BRR={thr_summary['brr']:.4f}"
                     f"  (n={len(all_metrics)})")
        log(line)

    # Save overview
    overview_path = os.path.join(args.output_dir, "attdef_overview.json")
    with open(overview_path, 'w', encoding='utf-8') as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)
    log(f"\nOverview saved to {overview_path}")

    # Summary table
    log("\n" + "=" * 60)
    log("SUMMARY TABLE")
    log("=" * 60)
    header = f"{'Thr':>6} {'Det':>6} {'Avg':>6} {'ClnDel':>7} {'ClnRat':>7}"
    if args.trigger:
        header += f" {'Prec':>8} {'Recall':>8} {'F1':>8} {'BRR':>8}"
    log(header)
    log("-" * len(header))
    for s in overview:
        line = (f"{s['threshold']:>6.2f} {s['total_detected']:>6} {s['avg_detected']:>6.2f}"
                f" {s['clean_avg_del']:>7.2f} {s['clean_avg_ratio']:>6.2%}")
        if args.trigger and 'precision' in s:
            line += (f" {s['precision']:>8.4f} {s['recall']:>8.4f}"
                     f" {s['f1']:>8.4f} {s['brr']:>8.4f}")
        log(line)

    log(f"\nMethod: {args.method}")
    log(f"Total samples: {len(sample_records)}")
    log_file.close()


if __name__ == "__main__":
    main()
