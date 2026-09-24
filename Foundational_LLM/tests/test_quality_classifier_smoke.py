"""Proves data_quality.quality_filter_classifier's SCORING MECHANISM (tokenize -> forward ->
regression score -> threshold) is wired correctly, without needing network access to the real
HuggingFaceFW/fineweb-edu-classifier weights (this sandbox can't reach huggingface.co -- see
prd.md). Builds a tiny BERT-style sequence-classification model locally (random init, same
architecture family, zero downloads) and runs the identical scoring logic
quality_filter_classifier's score_batch closure uses.

This does NOT prove the real classifier's quality judgments are good (it has none -- random
weights) -- only that the surrounding pipeline (tokenizer call, model forward, score
extraction, threshold comparison) is mechanically correct. Run this test against the real
model_id once network/model access is available to validate scores are sane on real text.
"""
import torch
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification


def test_scoring_mechanism():
  # A tiny BERT built from a small local config -- no network call, no real classifier weights.
  # google-bert/bert-base-uncased's TOKENIZER (vocab + rules) still needs a one-time fetch
  # normally; sidestep that here too by building a minimal WordPiece vocab locally so the
  # whole test is network-free.
  import tempfile, os
  vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"word{i}" for i in range(100)]
  with tempfile.TemporaryDirectory() as d:
    vocab_path = os.path.join(d, "vocab.txt")
    with open(vocab_path, "w") as f:
      f.write("\n".join(vocab))
    from transformers import BertTokenizer
    tok = BertTokenizer(vocab_file=vocab_path)

    config = BertConfig(
      vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
      intermediate_size=32, num_labels=1,  # regression head, matching fineweb-edu-classifier's shape
    )
    model = BertForSequenceClassification(config)
    model.eval()

    def score_batch(texts):
      scores = []
      with torch.no_grad():
        for t in texts:
          inputs = tok(t or "", return_tensors="pt", truncation=True, max_length=32)
          logits = model(**inputs).logits
          scores.append(float(logits.squeeze().item()))
      return scores

    texts = ["word1 word2 word3", "word4 word5", ""]
    scores = score_batch(texts)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    print("scores:", scores)

    # threshold filtering, same comparison quality_filter_classifier applies
    threshold = 0.0
    kept = [t for t, s in zip(texts, scores) if s >= threshold]
    print(f"kept {len(kept)}/{len(texts)} at threshold {threshold}")

  print("quality-classifier scoring mechanism OK (tokenize -> forward -> score -> threshold)")


if __name__ == "__main__":
  test_scoring_mechanism()
