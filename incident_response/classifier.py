"""RAG-enhanced LLM open-set classifier for HDFS abnormal template-event sequences.

Known anomaly types are classified into their fixed labels; any sequence that no known
class plausibly explains is assigned OTHER_LABEL (the paper's "Unknown Anomaly Types").

Decision flow (predict_one):
  1. best retrieval score < open_set_threshold            → Other (open_set_threshold)
  2. best score ≥ high_confidence_threshold                → top retrieved label, LLM bypassed
  3. ≥30% of input event tokens absent from all references → Other (novel tokens)
  4. majority_ratio < 0.6 and best score < 0.78            → Other (diversity)
  5. top-2 score gap < 0.04 and avg score < 1.15×threshold → Other (uncertain)
  6. LLM chooses one label from the candidate set; an LLM "Other" is overridden by the
     top retrieved label when retrieval is focused (majority ≥ 0.8, best score ≥ 0.72)
"""

import ast
import difflib
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# HDFS anomaly type IDs (loghub HDFS_v1 Event_traces.csv "Type") of the 10 known classes
TYPE_TO_TEXT = {
    5: "Namenode not updated after deleting block",
    31: "Write exception client give up",
    3: "Write failed at beginning",
    0: "Replica immediately deleted",
    4: "Received block that does not belong to any file",
    1: "Redundant addStoredBlock",
    21: "Delete a block that no longer exists on data node",
    7: "Empty packet for block",
    12: "Receive block exception",
    8: "Replication Monitor timeout",
}

KNOWN_LABELS = list(TYPE_TO_TEXT.values())
OTHER_LABEL = "Other anomaly type"
ALL_EVAL_LABELS = KNOWN_LABELS + [OTHER_LABEL]

# Short aliases for the LLM backbones evaluated in the paper (Table 7)
MODEL_ALIASES = {
    "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)


def normalize_sequence(x: Any) -> str:
    if isinstance(x, list):
        return " ".join(str(i).strip() for i in x)
    if pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return " ".join(str(i).strip() for i in parsed)
    except Exception:
        pass
    s = s.replace("[", " ").replace("]", " ")
    s = s.replace(",", " ").replace(";", " ")
    s = s.replace("'", " ").replace('"', " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_known_label(x: Any) -> str:
    """Map numeric or text labels to known label text when possible. Do not map unknowns."""
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return TYPE_TO_TEXT.get(int(x), str(int(x)))
    s = str(x).strip()
    if re.fullmatch(r"\d+", s):
        return TYPE_TO_TEXT.get(int(s), s)
    return s


def normalize_open_set_test_label(x: Any) -> str:
    """Known labels stay known; any unknown anomaly type becomes Other."""
    label = normalize_known_label(x)
    return label if label in KNOWN_LABELS else OTHER_LABEL


def canon_label(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def seq_tokens(seq: str) -> List[str]:
    return normalize_sequence(seq).split()


def seq_bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)] if len(tokens) >= 2 else []


def multiset_jaccard(a: List[str], b: List[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    inter = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return inter / union if union > 0 else 0.0


def set_jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    inter = len(sa & sb)
    return inter / union if union > 0 else 0.0


# ── Knowledge base ────────────────────────────────────────────────────────────

def build_kb_docs_from_csv(kb_df: pd.DataFrame) -> List[Dict[str, Any]]:
    docs = []
    for _, row in kb_df.iterrows():
        if "AnomalyText" in kb_df.columns:
            label = normalize_known_label(row["AnomalyText"])
        elif "Description" in kb_df.columns:
            label = normalize_known_label(row["Description"])
        elif "Type" in kb_df.columns:
            label = normalize_known_label(row["Type"])
        else:
            continue

        # For open-set detection, the KB contains only known classes.
        # Unknown classes are not added to the KB; they should be rejected as Other.
        if label not in KNOWN_LABELS:
            continue

        if "TemplateSequence" in kb_df.columns:
            seq = normalize_sequence(row["TemplateSequence"])
        elif "template_seq" in kb_df.columns:
            seq = normalize_sequence(row["template_seq"])
        elif "Features" in kb_df.columns:
            seq = normalize_sequence(row["Features"])
        else:
            seq = ""

        if seq:
            docs.append({
                "doc_type": "kb_sequence",
                "label": label,
                "type_id": str(row["Type"]) if "Type" in kb_df.columns else "",
                "sequence": seq,
            })
    return docs


# ── Retriever ─────────────────────────────────────────────────────────────────

class LexicalSequenceRetriever:
    def __init__(self, docs: List[Dict[str, Any]], length_penalty: bool = True):
        if not docs:
            raise ValueError("No retrieval documents found. Check kb_csv and the known label mapping.")
        self.docs = docs
        self.length_penalty = length_penalty
        self.corpus = [d["sequence"] for d in docs]
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            lowercase=False,
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.corpus)

    def score_doc(self, query_seq: str, doc_text: str) -> float:
        q_tokens, d_tokens = seq_tokens(query_seq), seq_tokens(doc_text)
        if not q_tokens or not d_tokens:
            return 0.0
        token_score = multiset_jaccard(q_tokens, d_tokens)
        bigram_score = set_jaccard(seq_bigrams(q_tokens), seq_bigrams(d_tokens))
        exact_bonus = 1.5 if q_tokens == d_tokens else 0.0
        # Only give contain_bonus when the doc is longer and contains the query —
        # NOT when the doc is a short substring of the query, which would inflate
        # scores for very short KB sequences (e.g. [E5, E22]) against long inputs.
        contain_bonus = 0.8 if query_seq in doc_text else 0.0
        prefix_bonus = 0.5 if len(q_tokens) <= len(d_tokens) and q_tokens == d_tokens[: len(q_tokens)] else 0.0
        suffix_bonus = 0.5 if len(q_tokens) <= len(d_tokens) and q_tokens == d_tokens[-len(q_tokens):] else 0.0
        q_vec = self.vectorizer.transform([query_seq])
        d_vec = self.vectorizer.transform([doc_text])
        tfidf_score = float(cosine_similarity(q_vec, d_vec)[0][0])

        base_score = 0.40 * tfidf_score + 0.35 * token_score + 0.25 * bigram_score
        if self.length_penalty:
            # Length similarity multiplier: penalises big length mismatches (1.0 at equal
            # length, 0.7 at 2:1, 0.46 at 10:1). Applied only to the base score, not the
            # exact/contain/prefix/suffix bonuses, so genuine structural matches still win.
            length_sim = min(len(q_tokens), len(d_tokens)) / max(len(q_tokens), len(d_tokens))
            base_score *= 0.4 + 0.6 * length_sim
        return base_score + exact_bonus + contain_bonus + prefix_bonus + suffix_bonus

    def retrieve(self, query_seq: str, top_k: int) -> List[Dict[str, Any]]:
        scored = []
        for doc in self.docs:
            score = self.score_doc(query_seq, doc["sequence"])
            scored.append({
                "doc_type": doc["doc_type"],
                "label": doc["label"],
                "score": float(score),
                "matched_sequence": doc["sequence"],
                "metadata": doc,
            })
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


def rank_candidate_labels(retrieved: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    label_info = defaultdict(lambda: {"count": 0, "best_score": -1.0, "doc_types": []})
    for item in retrieved:
        label = item["label"]
        label_info[label]["count"] += 1
        label_info[label]["best_score"] = max(label_info[label]["best_score"], item["score"])
        label_info[label]["doc_types"].append(item["doc_type"])
    ranked = sorted(label_info.items(), key=lambda x: (-x[1]["best_score"], -x[1]["count"], x[0]))
    labels = [x[0] for x in ranked]
    if OTHER_LABEL not in labels:
        labels.append(OTHER_LABEL)
    return labels, dict(label_info)


# ── LLM ───────────────────────────────────────────────────────────────────────

def resolve_model_name(name: str) -> str:
    name = name.strip()
    return MODEL_ALIASES.get(name.lower(), name)


def load_llm(model_name: str, load_in_4bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<|pad|>"

    model_kwargs = dict(device_map="auto", trust_remote_code=True)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs.update(dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True)))
    else:
        model_kwargs.update(dict(torch_dtype="auto"))

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return tokenizer, model


def make_generation_text(prompt: str, tokenizer) -> str:
    system_msg = (
        "You are an open-set anomaly type classifier for log template-event sequences. "
        "Your primary task is to decide whether the input belongs to a known anomaly class "
        "or is an unknown anomaly type. "
        "Output exactly one line: LABEL: <label>. Do not explain."
    )
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]

    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    # Fallback for base models without a chat template.
    return f"{system_msg}\n\n{prompt}\n\nLABEL:"


def generate_label(prompt: str, tokenizer, model, max_new_tokens: int) -> str:
    text = make_generation_text(prompt, tokenizer)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ── Prompt, parsing, and open-set decision ────────────────────────────────────

def make_prompt(input_sequence: str, retrieved: List[Dict[str, Any]],
                open_set_threshold: float, majority_ratio: float = 1.0) -> str:
    ranked_labels, _ = rank_candidate_labels(retrieved)
    known_labels = [lab for lab in ranked_labels if lab != OTHER_LABEL]
    known_text = "\n".join([f"  - {c}" for c in known_labels])

    best_score = max(x["score"] for x in retrieved)
    unique_retrieved = list(dict.fromkeys(x["label"] for x in retrieved))
    top_class = unique_retrieved[0]
    top_count = int(round(majority_ratio * len(retrieved)))

    # Score confidence band
    if best_score >= 0.80:
        confidence = "HIGH"
    elif best_score >= 0.65:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    # Signal + per-signal decision hint
    if majority_ratio >= 0.8:
        signal = f"FOCUSED ({top_count}/{len(retrieved)} refs from \"{top_class}\", confidence {confidence})"
        decision_hint = (
            f'The dominant class is "{top_class}". '
            f'Choose it if the input sequence matches its references structurally. '
            f'Choose "{OTHER_LABEL}" only if the input has fundamentally different event patterns.'
        )
    elif majority_ratio >= 0.5:
        signal = f"MIXED ({len(unique_retrieved)} classes, confidence {confidence})"
        decision_hint = (
            f'A few classes appear among the references. Compare the input against each one '
            f'and choose the class whose reference best explains the input\'s overall event sequence. '
            f'Choose "{OTHER_LABEL}" only if NONE of the references plausibly explain the input '
            f'(events, ordering, or structure clearly do not align).'
        )
    else:
        signal = f"DIVERSE ({len(unique_retrieved)} classes, confidence {confidence})"
        decision_hint = (
            f'Retrieved references are spread across many classes with no dominant match. '
            f'This is a strong indicator of an unknown anomaly. '
            f'Choose "{OTHER_LABEL}" unless one reference matches the complete input very closely '
            f'(same event types, same order, similar length).'
        )

    blocks = []
    for i, item in enumerate(retrieved, 1):
        blocks.append(
            f"[Ref {i}] class: {item['label']}\n"
            f"         seq:   {item['matched_sequence']}"
        )
    context_text = "\n\n".join(blocks)

    return f"""You are an open-set anomaly type classifier for log template-event sequences.

=== Labels (copy exactly, do not paraphrase) ===
Known classes:
{known_text}
Unknown:
  - {OTHER_LABEL}

=== Input sequence ===
{input_sequence}

=== Retrieval signal: {signal} ===
{decision_hint}

{context_text}

=== Decision rules ===

Rule 1 — Prefer a known class when:
  • At least one reference plausibly explains the input's overall event sequence.
  • The dominant class in the retrieval signal aligns with the input's pattern.
  • Most of the input's events appear in the references in similar order.

Rule 2 — Choose "{OTHER_LABEL}" ONLY when:
  • NO reference plausibly explains the input (event types, ordering, or structure differ widely).
  • The input introduces many event types absent from every reference.
  • The retrieval signal is DIVERSE AND no single reference comes close to matching.

You MUST output exactly ONE label. No explanation. Copy text exactly from the label list.
LABEL: <label>""".strip()


def parse_label(raw_text: str, candidate_labels: List[str]) -> Optional[str]:
    text = str(raw_text).strip()

    # Take the first LABEL: line the model produced
    match = re.search(r"LABEL\s*:\s*(.+)", text, re.IGNORECASE)
    pred = match.group(1).strip() if match else next((line.strip() for line in text.splitlines() if line.strip()), None)
    if pred is None:
        return None

    # Strip formatting noise: quotes, bullets, leading numbers/punctuation
    pred = pred.strip('"').strip("'").strip()
    pred = re.sub(r"^[\-\*\d\.\)\s]+", "", pred).strip()

    # If model listed multiple options (e.g. "X or Y", "X / Y", "X, Y"), take only the first
    pred = re.split(r"\s+or\s+|\s*/\s*|,\s*", pred, maxsplit=1)[0].strip()

    # Explicit other-like outputs
    if canon_label(pred) in {"other", "others", "other anomaly", "other anomaly type", "unknown", "unknown anomaly type"}:
        return OTHER_LABEL if OTHER_LABEL in candidate_labels else None

    # Exact or canonical match
    for c in candidate_labels:
        if pred == c or canon_label(pred) == canon_label(c):
            return c

    # Conservative fuzzy match — high cutoff to avoid false positives
    canon_to_orig = {canon_label(c): c for c in candidate_labels}
    close = difflib.get_close_matches(canon_label(pred), list(canon_to_orig.keys()), n=1, cutoff=0.82)
    return canon_to_orig[close[0]] if close else None


def _decision(seq: str, pred_label: str, decision_source: str, ranked_labels, retrieved,
              raw_output: str, label_info, best_score: float) -> Dict[str, Any]:
    return {
        "input_sequence": seq,
        "pred_label": pred_label,
        "decision_source": decision_source,
        "candidate_labels": ranked_labels,
        "retrieved_labels": [x["label"] for x in retrieved],
        "retrieved_doc_types": [x["doc_type"] for x in retrieved],
        "retrieved_scores": [float(x["score"]) for x in retrieved],
        "retrieved_sequences": [x["matched_sequence"] for x in retrieved],
        "raw_output": raw_output,
        "label_info": label_info,
        "best_score": best_score,
    }


def predict_one(input_sequence: str, retriever, tokenizer, model, top_k: int,
                open_set_threshold: float, high_confidence_threshold: float,
                threshold_first: bool, max_new_tokens: int,
                min_bypass_majority: int = 1) -> Dict[str, Any]:
    seq = normalize_sequence(input_sequence)
    retrieved = retriever.retrieve(seq, top_k=top_k)
    ranked_labels, label_info = rank_candidate_labels(retrieved)
    best_score = max([x["score"] for x in retrieved], default=0.0)

    # Top-K majority count for the highest-scoring class — used by the bypass guard.
    top_label = retrieved[0]["label"] if retrieved else None
    top_label_count = sum(1 for x in retrieved if x["label"] == top_label)

    # Low-score rejection: the input is too dissimilar from all known classes → Other.
    if threshold_first and best_score < open_set_threshold:
        return _decision(seq, OTHER_LABEL, "open_set_threshold", ranked_labels, retrieved, "", label_info, best_score)

    # High-confidence retrieval: bypass the LLM when the score is high and the top class
    # has enough majority in top-K (min_bypass_majority=1 is a score-only bypass).
    if best_score >= high_confidence_threshold and top_label_count >= min_bypass_majority:
        return _decision(seq, retrieved[0]["label"], "retrieval_high_confidence", ranked_labels, retrieved, "",
                         label_info, best_score)

    # Retrieval signals used by the remaining tiers and the LLM prompt
    scores = [x["score"] for x in retrieved]
    label_counts = Counter(x["label"] for x in retrieved)
    top_count = label_counts.most_common(1)[0][1]
    majority_ratio = top_count / len(retrieved)   # 1.0 = unanimous, 0.2 = fully mixed
    avg_score = sum(scores) / len(scores)
    score_gap = scores[0] - scores[1] if len(scores) >= 2 else 1.0

    # Novel-tokens Other: many input event tokens appear in no retrieved reference.
    input_token_set = set(seq_tokens(seq))
    ref_token_set: set = set()
    for r in retrieved:
        ref_token_set.update(seq_tokens(r["matched_sequence"]))
    novel_ratio = len(input_token_set - ref_token_set) / max(len(input_token_set), 1)

    if novel_ratio >= 0.30:
        return _decision(seq, OTHER_LABEL, "retrieval_novel_tokens", ranked_labels, retrieved, "", label_info, best_score)

    # Diversity-based Other: clearly mixed retrieval with moderate confidence → Other
    if majority_ratio < 0.6 and best_score < 0.78:
        return _decision(seq, OTHER_LABEL, "retrieval_diversity", ranked_labels, retrieved, "", label_info, best_score)

    # Score-gap Other: top-2 scores are close and the average is only modestly above threshold → Other
    if score_gap < 0.04 and avg_score < open_set_threshold * 1.15:
        return _decision(seq, OTHER_LABEL, "retrieval_uncertain", ranked_labels, retrieved, "", label_info, best_score)

    prompt = make_prompt(seq, retrieved, open_set_threshold, majority_ratio)
    raw = generate_label(prompt, tokenizer, model, max_new_tokens=max_new_tokens)
    parsed = parse_label(raw, ranked_labels)

    if parsed is not None:
        # Safety net: if the LLM says Other but retrieval is clearly focused on one known
        # class with decent confidence, trust retrieval.
        if parsed == OTHER_LABEL and majority_ratio >= 0.8 and best_score >= 0.72:
            final_pred = retrieved[0]["label"]
            decision_source = "llm_other_override_focused"
        else:
            final_pred = parsed
            decision_source = "llm"
    else:
        # LLM did not follow the output format: fall back to retrieval or Other.
        final_pred = OTHER_LABEL if best_score < open_set_threshold else ranked_labels[0]
        decision_source = "open_set_fallback" if final_pred == OTHER_LABEL else "retrieval_fallback"

    return _decision(seq, final_pred, decision_source, ranked_labels, retrieved, raw, label_info, best_score)
