"""Spark packing: explode raw per-document token arrays into a flat stream and re-chunk into
fixed-length blocks for the trainer. Fixes three issues in the original single-source version:

  1. Missing document separator: an EOS token is now inserted after every document's tokens
     before exploding, so a packed block doesn't glue two unrelated documents together with
     no marker between them.
  2. Missing document-boundary info for the model: a parallel `doc_start` boolean array is
     carried through packing (True at each document's first token, False elsewhere) so
     data.ParquetTokenDataset / model.build_doc_attention_mask can stop attention from
     crossing into a different document within the same packed block.
  3. Non-global ordering: the original used `monotonically_increasing_id()`, which is only
     guaranteed monotonic *within* a Spark partition, not globally -- on a multi-partition
     DataFrame this could group tokens from different documents' true positions together in
     ways that don't reflect actual corpus order. This version uses
     `row_number().over(Window.orderBy(...))`, which is a global (shuffle-based) sort and
     therefore correct, at the cost of that shuffle -- an accepted tradeoff for correctness.
     `collect_list` after `groupBy` also doesn't guarantee element order on its own, so blocks
     are additionally reconstructed via `sort_array` over a (pos, tokens, doc_start) struct
     before being split back into columns.

Not run against a live Spark cluster from this environment (none available here) -- dry-run
against a small sample before trusting it at full corpus scale.
"""
from __future__ import annotations
import os
from pyspark.sql import functions as Fun
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, BooleanType


def _doc_start_flags(arr):
  return [i == 0 for i in range(len(arr))]


_doc_start_udf = Fun.udf(_doc_start_flags, ArrayType(BooleanType()))


def pack_source_to_shards(spark, raw_path: str, out_root: str, source_name: str, ctx: int,
                           eos_token_id: int, order_col: str = "file_path",
                           seed: int = 1337, shard_rows: int = 50000):
  """`raw_path` is expected to have a `token_content` column (array<long>, one row per
  document) and an `order_col` (e.g. file_path, or any stable per-document id) establishing
  the corpus's intended document order.
  """
  block_size = ctx + 1
  df = spark.read.parquet(raw_path)
  train_df, test_df = df.randomSplit([0.9, 0.1], seed)

  def _pack(split_df, split_name):
    with_eos = split_df.withColumn(
      "token_content", Fun.concat(Fun.col("token_content"), Fun.array(Fun.lit(eos_token_id)))
    )
    with_flags = with_eos.withColumn("doc_start_content", _doc_start_udf(Fun.col("token_content")))

    # zip tokens+flags into one array of structs so posexplode can't desync the two columns
    with_zip = with_flags.withColumn(
      "zipped", Fun.arrays_zip(Fun.col("token_content"), Fun.col("doc_start_content"))
    )

    order_window = Window.orderBy(order_col)
    ordered = with_zip.withColumn("_doc_rank", Fun.row_number().over(order_window))

    exploded = ordered.select("_doc_rank", Fun.posexplode("zipped").alias("_pos_in_doc", "_pair"))
    exploded = exploded.select(
      "_doc_rank", "_pos_in_doc",
      Fun.col("_pair.token_content").alias("tokens"),
      Fun.col("_pair.doc_start_content").alias("doc_start"),
    )

    global_order_window = Window.orderBy("_doc_rank", "_pos_in_doc")
    exploded = exploded.withColumn("pos", Fun.row_number().over(global_order_window) - 1)
    exploded = exploded.withColumn("block_id", (Fun.col("pos") / block_size).cast("long"))

    # collect_list after groupBy doesn't guarantee order -- collect (pos, tokens, doc_start)
    # structs and sort_array them (lexicographic on the first field, i.e. pos) to reconstruct
    # each block's tokens in true sequence order.
    exploded = exploded.withColumn("_rec", Fun.struct("pos", "tokens", "doc_start"))
    packed = exploded.groupBy("block_id").agg(Fun.sort_array(Fun.collect_list("_rec")).alias("_recs"))
    packed = packed.withColumn("tokens", Fun.col("_recs.tokens"))
    packed = packed.withColumn("doc_start", Fun.col("_recs.doc_start"))
    packed = packed.drop("_recs")

    packed = packed.withColumn("block_idx", Fun.row_number().over(Window.orderBy("block_id")))
    packed = packed.withColumn("shard_id", (Fun.col("block_idx") / shard_rows).cast("int"))

    out_path = os.path.join(out_root, source_name, split_name)
    (packed.select("tokens", "doc_start", "shard_id")
           .repartition("shard_id")
           .write.partitionBy("shard_id")
           .mode("overwrite")
           .parquet(out_path))
    print(f"[{source_name}/{split_name}] wrote packed shards (with doc_start) to {out_path}")

  _pack(train_df, "train")
  _pack(test_df, "test")


# Example: pack every configured source once a raw token_content parquet exists per source,
# each written under the shared TRANSFORMED_ROOT/{source}/{split} layout config.DATA_SOURCES
# expects.
#
# import tiktoken
# from config import DATA_SOURCES
# RAW_SOURCE_PATHS = {
#   "web":       "abfss://.../data/raw/.../transformed/openwebtext/openwebtext.parquet",
#   "code":      "abfss://.../data/raw/.../transformed/code/code.parquet",
#   "wikipedia": "abfss://.../data/raw/.../transformed/wikipedia/wikipedia.parquet",
#   "books":     "abfss://.../data/raw/.../transformed/books/books.parquet",
#   "math":      "abfss://.../data/raw/.../transformed/math/math.parquet",
# }
# TRANSFORMED_ROOT = "abfss://root@coentus6abfsprod001.dfs.core.windows.net/data/transformed"
# _eos = tiktoken.get_encoding("cl100k_base").eot_token
# for cfg in DATA_SOURCES:
#   pack_source_to_shards(spark, RAW_SOURCE_PATHS[cfg["name"]], TRANSFORMED_ROOT, cfg["name"],
#                          ctx=1024, eos_token_id=_eos)
