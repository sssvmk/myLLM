"""Loads the cl100k_base tokenizer.

tiktoken.get_encoding("cl100k_base") is NOT purely local: on first use it fetches
https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken over the network
(then caches it). In a network-restricted environment that call can simply fail -- confirmed
in this project's own dev sandbox, which returned a 403 Forbidden for exactly that URL (see
prd.md).

load_cl100k_encoding(local_path=...) avoids that entirely: tiktoken.load.load_tiktoken_bpe
accepts a plain local file path (anything without "://" is opened directly, never fetched --
see tiktoken.load.read_file's source), so pointing it at a local copy of the ranks file
builds a full tiktoken.Encoding with ZERO network calls. This constructs the Encoding by
hand from cl100k_base's own pat_str/special_tokens (copied from
tiktoken_ext.openai_public.cl100k_base's source), rather than going through
tiktoken.get_encoding, which is what makes it possible to bypass the hardcoded blob URL.

Getting the local ranks file (one-time, off this project):
  1. Download https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken from
     any machine with unrestricted internet access (or a mirror -- several exist on
     huggingface.co, e.g. in the microsoft/Phi-3-small-8k-instruct repo; verify against the
     hash below regardless of source).
  2. Verify its SHA-256 matches EXPECTED_SHA256 below.
  3. Copy the file to wherever this pipeline runs and pass its path as --tokenizer_path (or
     TOKENIZER_PATH env var).
  This is a one-time manual step -- after that, this loader never touches the network again.
"""
from __future__ import annotations
import hashlib
from typing import Optional
import tiktoken
from tiktoken.load import load_tiktoken_bpe

# Copied from tiktoken_ext.openai_public.cl100k_base()'s source -- these define the encoding
# itself and don't change based on where the ranks data came from.
_PAT_STR = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)
_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}
CL100K_EOT_TOKEN = _SPECIAL_TOKENS["<|endoftext|>"]  # public constant -- lets callers (e.g. packing.py)
                                                       # get the EOT id without loading a full encoding
EXPECTED_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"


def load_cl100k_encoding(local_path: Optional[str] = None) -> tiktoken.Encoding:
    """local_path: path to a local cl100k_base.tiktoken ranks file. If given, loads fully
    offline (verifying its hash) and never touches the network. If omitted, falls back to
    tiktoken.get_encoding("cl100k_base"), which fetches from the network on first use unless
    tiktoken's own cache already has it."""
    if local_path:
        with open(local_path, "rb") as f:
            data = f.read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != EXPECTED_SHA256:
            raise ValueError(
                f"{local_path} does not match the expected cl100k_base.tiktoken content "
                f"(sha256 {actual}, expected {EXPECTED_SHA256}) -- re-download it rather than "
                f"proceeding with a possibly-corrupted or wrong-version file."
            )
        ranks = load_tiktoken_bpe(local_path)  # local path -> no network call, see module docstring
        return tiktoken.Encoding(
            name="cl100k_base", pat_str=_PAT_STR, mergeable_ranks=ranks, special_tokens=_SPECIAL_TOKENS,
        )
    return tiktoken.get_encoding("cl100k_base")
