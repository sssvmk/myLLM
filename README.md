myLLM

A from-scratch LLM lifecycle: pretraining today, post-training next.

myLLM is a PyTorch pipeline for training a GPT-style language model from random initialization, built to mirror the stages frontier labs use. It covers model architecture, distributed training, and evaluation. A post-training stack (Keystone) is specified and builds on this foundation.

Why this exists

Most people use foundation models; few have seen what it takes to build one. This project reproduces the design decisions behind modern pipelines (DeepSeek, Llama, Tulu) at a scale where every component can be read, modified, and tested.

What’s in it

Two selectable architectures

Dense: RoPE attention with a GELU MLP, used as a baseline.
DeepSeek-style: Multi-head Latent Attention (compressed K/V latents with a decoupled RoPE path) plus fine-grained and shared-expert Mixture-of-Experts. Load balancing is auxiliary-loss-free, using a per-expert routing bias updated each step.

Data loading

Streaming Parquet dataset that is worker-safe and padding-safe
Weighted multi-source mixture (web, code, Wikipedia, books, math), with weights overridable from the CLI for ablations
Document-boundary attention masking, so attention never crosses between unrelated documents in a packed block

Training

Gradient accumulation, mixed precision, warmup + cosine LR schedule
Resumable checkpoints
ZeRO stages 0-3 (DDP through full FSDP) behind a single --zero_stage flag

Observability

JSONL metrics, optional TensorBoard, chart generation (loss, LR, throughput, perplexity, MoE expert balance)
MMLU evaluation via log-likelihood scoring

Everything is config-driven through settings.yaml: model presets, data locations, and tokenizer paths.

Quick start
bash
pip install -r requirements.txt

python main.py \
  --data_root /path/to/packed --sources web \
  --tokenizer_path /path/to/cl100k_base.tiktoken \
  --out ./checkpoints --model 125M --arch deepseek \
  --ctx 1024 --max_steps 1000 --tensorboard

# multi-GPU
torchrun --nproc_per_node=8 main.py ... --zero_stage 3
Scale

Presets range from ~162M to ~6.9B dense parameters. The DeepSeek variant has far more total parameters than active ones (for example 475M total and 220M active at the smallest preset). A laptop can train a ~50M custom config; the larger presets need GPUs.

Roadmap: Keystone

Post-training reuses this codebase: mid-training, supervised fine-tuning, preference optimization, reinforcement learning with verifiable rewards (GRPO-style), and distillation. A generation loop with KV caching is the main missing prerequisite.

Status

Verified: both architectures, MoE routing, data loading, checkpoint/resume, and a 2-process ZeRO/FSDP run on CPU. 

Differences from frontier pipelines

No tensor/pipeline parallelism, FP8, fused MoE kernels, or custom tokenizer, and far less data. The structure is the same; the scale is not.
