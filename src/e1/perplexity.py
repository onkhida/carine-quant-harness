"""Language coherence evaluation via sliding-window perplexity on WikiText-2."""

from typing import Optional
import torch
from datasets import load_dataset


def evaluate_wikitext_perplexity(
    model,
    tokenizer,
    max_length: int = 512,
    stride: int = 256,
    max_eval_tokens: int = 10000,
    device: Optional[str] = None,
) -> float:
    """
    Computes cross-entropy loss over the WikiText-2 validation set
    using a sliding window to evaluate grammatical fluency and language degradation.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    except Exception:
        wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(wikitext["text"])

    encodings = tokenizer(text, return_tensors="pt")
    seq_len = min(encodings.input_ids.size(1), max_eval_tokens)
    input_ids = encodings.input_ids[:, :seq_len].to(device)

    nlls = []
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - begin_loc
        batch_input_ids = input_ids[:, begin_loc:end_loc]
        target_ids = batch_input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Mask context tokens

        with torch.no_grad():
            outputs = model(batch_input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl

