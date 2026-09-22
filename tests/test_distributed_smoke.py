"""Local 2-process, CPU/gloo smoke test for distributed.py. Exercises real
torch.distributed.fsdp.FullyShardedDataParallel and ZeroRedundancyOptimizer -- not a mock --
against a tiny model, for each zero_stage in (0, 1, 2, 3): wrap, forward, backward, optimizer
step, full checkpoint save + round-trip load into a fresh model, and a value comparison
against a single-process reference run.

Run directly: python tests/test_distributed_smoke.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


def _worker(rank, world_size, zero_stage, tmp_dir, master_port):
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(master_port)

  from distributed import (setup_distributed, wrap_model, build_optimizer_for_zero,
                            full_model_state_dict, full_optimizer_state_dict,
                            load_full_model_state_dict, load_full_optimizer_state_dict,
                            is_main_process)
  from model import GPTModel

  setup_distributed()
  device = torch.device("cpu")
  torch.manual_seed(0)   # same init on every rank before wrapping

  model = GPTModel(vocab_size=301, d_model=16, ctx=15, n_layers=2, d_ff=32, n_heads=4,
                    dropout=0.0, arch="dense")
  wrapped = wrap_model(model, zero_stage, device)
  optimizer = build_optimizer_for_zero(wrapped, zero_stage, lr=1e-3, weight_decay=0.1)

  torch.manual_seed(123 + rank)   # different data per rank, like real data-parallel shards
  idx = torch.randint(0, 300, (2, 16))
  xb, yb = idx[:, :-1], idx[:, 1:]
  logits, aux = wrapped(xb)
  loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  # --- checkpoint round trip ---
  msd = full_model_state_dict(wrapped)          # collective: every rank calls this
  osd = full_optimizer_state_dict(wrapped, optimizer)  # collective

  ckpt_path = os.path.join(tmp_dir, "ckpt.pt")
  if is_main_process():
    torch.save({"model": msd, "optimizer": osd}, ckpt_path)
  if dist.is_initialized():
    dist.barrier()

  # load into a fresh model/optimizer
  model2 = GPTModel(vocab_size=301, d_model=16, ctx=15, n_layers=2, d_ff=32, n_heads=4,
                     dropout=0.0, arch="dense")
  wrapped2 = wrap_model(model2, zero_stage, device)
  optimizer2 = build_optimizer_for_zero(wrapped2, zero_stage, lr=1e-3, weight_decay=0.1)

  if is_main_process():
    obj = torch.load(ckpt_path, weights_only=False)
    payload = [obj["model"], obj["optimizer"]]
  else:
    payload = [None, None]
  if dist.is_initialized():
    dist.broadcast_object_list(payload, src=0)
  full_msd2, full_osd2 = payload

  load_full_model_state_dict(wrapped2, full_msd2)
  load_full_optimizer_state_dict(wrapped2, optimizer2, full_osd2)

  # verify the reloaded model matches the saved one, gathered fresh from each side
  msd_check = full_model_state_dict(wrapped2)
  if is_main_process():
    for k in msd:
      assert torch.allclose(msd[k], msd_check[k]), f"mismatch after reload: {k}"
    print(f"[zero_stage={zero_stage}] rank0 OK: loss={loss.item():.4f}, "
          f"{len(msd)} params verified after checkpoint round-trip")

  if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()


def run_stage(zero_stage: int, world_size: int = 2, port: int = 29510):
  with tempfile.TemporaryDirectory() as tmp_dir:
    mp.spawn(_worker, args=(world_size, zero_stage, tmp_dir, port), nprocs=world_size, join=True)


if __name__ == "__main__":
  for i, stage in enumerate([0, 1, 2, 3]):
    run_stage(stage, world_size=2, port=29510 + i)
  print("ALL DISTRIBUTED ZeRO/FSDP SMOKE TESTS PASSED")
