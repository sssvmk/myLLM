"""Corpus-hardening utilities for the Spark ETL stage. Run these on a DataFrame with one row
per raw document and a `text` column (untokenized), BEFORE tokenization/packing.py. Intended
order: exact_dedup -> near_dedup_minhash -> quality_filter_heuristics ->
decontaminate_against_eval_sets -> (tokenize) -> packing.pack_source_to_shards.

None of this has been run against a live Spark cluster from this environment (no cluster
available here) -- dry-run against a small sample before trusting it at full corpus scale,
per the README.
"""
from __future__ import annotations
from typing import List
import hashlib

from pyspark.sql import DataFrame
from pyspark.sql import functions as Fun
from pyspark.sql.types import FloatType
from pyspark.ml.feature import Tokenizer, NGram, HashingTF, MinHashLSH


def exact_dedup(df: DataFrame, text_col: str = "text") -> DataFrame:
  """Drops exact-duplicate documents via a content hash. Run this first -- cheap, and removes
  the majority of duplication in most web-scraped corpora before near-dedup runs."""
  hash_udf = Fun.udf(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
  hashed = df.withColumn("_content_hash", hash_udf(Fun.col(text_col)))
  return hashed.dropDuplicates(["_content_hash"]).drop("_content_hash")


def near_dedup_minhash(df: DataFrame, text_col: str = "text", ngram_n: int = 5,
                        jaccard_threshold: float = 0.8, num_hash_tables: int = 5) -> DataFrame:
  """MinHash-LSH near-duplicate removal: catches documents that are near-identical but not
  byte-identical (mirrors, lightly-edited reposts) that exact_dedup misses. LSH keeps this
  roughly O(n log n) rather than O(n^2) pairwise comparison, but it's still expensive at full
  corpus scale -- run after exact_dedup has already cut volume down.
  """
  df = df.withColumn("_id", Fun.monotonically_increasing_id())
  tokenized = Tokenizer(inputCol=text_col, outputCol="_words").transform(df)
  ngrams = NGram(n=ngram_n, inputCol="_words", outputCol="_ngrams").transform(tokenized)
  featurized = HashingTF(inputCol="_ngrams", outputCol="_features", numFeatures=1 << 18).transform(ngrams)
  featurized = featurized.filter(Fun.size("_ngrams") > 0)  # MinHashLSH needs non-empty vectors

  mh = MinHashLSH(inputCol="_features", outputCol="_hashes", numHashTables=num_hash_tables)
  model = mh.fit(featurized)
  hashed = model.transform(featurized)

  joined = model.approxSimilarityJoin(hashed, hashed, 1 - jaccard_threshold, distCol="_jaccardDist") \
                .filter(Fun.col("datasetA._id") < Fun.col("datasetB._id"))
  dup_ids = joined.select(Fun.col("datasetB._id").alias("_id")).distinct()

  return (hashed.join(dup_ids, on="_id", how="left_anti")
                .drop("_words", "_ngrams", "_features", "_hashes", "_id"))


def decontaminate_against_eval_sets(train_df: DataFrame, eval_texts: List[str],
                                     text_col: str = "text", n: int = 13,
                                     overlap_threshold: float = 0.8) -> DataFrame:
  """Drops training documents whose n-gram overlap with any eval-set text (MMLU/GSM8K/
  HumanEval/etc -- pass their raw question/context text, not just answers) exceeds
  `overlap_threshold`. `eval_texts` is loaded into the driver (eval sets are small relative to
  pretraining corpora) and broadcast to executors."""
  def ngrams(s: str, n: int):
    toks = s.split()
    return set(tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1)))

  eval_ngram_sets = [ngrams(t, n) for t in eval_texts]
  eval_ngrams_all = set().union(*eval_ngram_sets) if eval_ngram_sets else set()
  bc_eval_ngrams = train_df.sql_ctx.sparkSession.sparkContext.broadcast(eval_ngrams_all)

  def overlap_ratio(text: str) -> float:
    doc_ngrams = ngrams(text, n)
    if not doc_ngrams:
      return 0.0
    hits = sum(1 for g in doc_ngrams if g in bc_eval_ngrams.value)
    return hits / len(doc_ngrams)

  overlap_udf = Fun.udf(overlap_ratio, FloatType())
  scored = train_df.withColumn("_eval_overlap", overlap_udf(Fun.col(text_col)))
  return scored.filter(Fun.col("_eval_overlap") < overlap_threshold).drop("_eval_overlap")


def quality_filter_heuristics(df: DataFrame, text_col: str = "text",
                               min_words: int = 50, max_symbol_ratio: float = 0.3,
                               max_repeated_line_ratio: float = 0.3) -> DataFrame:
  """Gopher/C4-style heuristic quality filters: drop documents that are too short, too
  symbol-heavy (boilerplate/spam), or dominated by repeated lines (nav menus, templated
  pages). This is the filtering layer that matters most with a single web-only source, since
  there's no clean secondary source (Wikipedia, books) to upweight instead."""
  def word_count(t):
    return len(t.split())

  def symbol_ratio(t):
    if not t:
      return 1.0
    symbols = sum(1 for c in t if not (c.isalnum() or c.isspace()))
    return symbols / len(t)

  def repeated_line_ratio(t):
    lines = [l for l in t.split("\n") if l.strip()]
    if not lines:
      return 1.0
    return 1 - (len(set(lines)) / len(lines))

  wc_udf = Fun.udf(word_count)
  sr_udf = Fun.udf(symbol_ratio, FloatType())
  rl_udf = Fun.udf(repeated_line_ratio, FloatType())

  scored = (df.withColumn("_word_count", wc_udf(Fun.col(text_col)))
              .withColumn("_symbol_ratio", sr_udf(Fun.col(text_col)))
              .withColumn("_repeated_line_ratio", rl_udf(Fun.col(text_col))))

  filtered = scored.filter(
    (Fun.col("_word_count").cast("int") >= min_words) &
    (Fun.col("_symbol_ratio") <= max_symbol_ratio) &
    (Fun.col("_repeated_line_ratio") <= max_repeated_line_ratio)
  )
  return filtered.drop("_word_count", "_symbol_ratio", "_repeated_line_ratio")


def quality_filter_classifier(df: DataFrame, text_col: str = "text", model_path: str = None,
                               score_threshold: float = 0.5):
  """STUB -- deliberately not implemented. Scoring-and-filtering with a trained quality
  classifier (fastText or a small transformer, scored against a "reference quality" set the
  way GPT-3 / FineWeb-Edu do) needs: (1) a positive set of known-high-quality text
  (Wikipedia/curated references), (2) a negative/background sample of raw web text, (3) a
  lightweight classifier (fastText is the standard cheap choice at corpus scale). None of
  that exists in this codebase yet -- left as a stub rather than filled with a placeholder
  heuristic that would look implemented but wasn't actually trained on anything. Use
  quality_filter_heuristics in the meantime.
  """
  raise NotImplementedError(
    "quality_filter_classifier needs a trained classifier + reference quality set; "
    "use quality_filter_heuristics until one exists."
  )
