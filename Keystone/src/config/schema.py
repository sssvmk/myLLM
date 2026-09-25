"""Configuration schema (CF-1, CF-2).

Every model forbids unknown keys and declares no defaults: a missing key is a validation error
naming its dotted path. "Optional[X]" here means the key must be present but may be null.
Cross-field rules that need more than one section live in config/validate.py.
"""
from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# --------------------------------------------------------------------------- run / infra
class RunCfg(Strict):
    name: str
    seed: int
    local_work_dir: str
    output_root_uri: str
    device: Literal["auto", "cuda", "cpu"]
    precision: Literal["bf16", "fp16", "fp32"]


class StorageCfg(Strict):
    upload_retries: int = Field(ge=0)
    protocols: Dict[str, Dict[str, object]]


class FoundationCfg(Strict):
    code_path: str


class Architecture(Strict):
    arch: Literal["dense", "deepseek"]
    d_model: int = Field(gt=0)
    n_layer: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    d_ff: int = Field(gt=0)
    ctx: int = Field(gt=0)
    rope_theta: float = Field(gt=0)
    d_latent: int = Field(gt=0)
    d_rope: int = Field(gt=0)
    n_routed_experts: int = Field(gt=0)
    n_shared_experts: int = Field(ge=0)
    moe_top_k: int = Field(gt=0)
    vocab_rows: int = Field(gt=0)

    @model_validator(mode="after")
    def _check(self):
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.d_rope % 2:
            raise ValueError("d_rope must be even for RoPE")
        if self.moe_top_k > self.n_routed_experts:
            raise ValueError("moe_top_k cannot exceed n_routed_experts")
        return self


class BaseModelCfg(Strict):
    checkpoint_uri: str
    architecture: Architecture


class SpecialToken(Strict):
    text: str
    id: int


class TokenizerCfg(Strict):
    ranks_file_uri: str
    ranks_sha256: str
    eot_token_id: int
    pad_token_id: int
    chat_special_tokens: Dict[str, SpecialToken]

    @model_validator(mode="after")
    def _required_names(self):
        missing = {"im_start", "im_end"} - set(self.chat_special_tokens)
        if missing:
            raise ValueError(f"chat_special_tokens must define {sorted(missing)}")
        return self


class ChatTemplateCfg(Strict):
    turn_start: str
    turn_end: str
    generation_prompt: str
    reasoning_open: str
    reasoning_close: str


# --------------------------------------------------------------------------- datasets (DS-1..DS-3a)
DatasetFormat = Literal["parquet", "jsonl", "json", "packed_tokens"]


class DatasetBase(Strict):
    uri: str
    files: Dict[str, str]
    format: DatasetFormat
    adapter: str
    field_map: Dict[str, Optional[str]]
    license: str
    third_party_generated: bool
    max_samples: Optional[int]


# adapter -> (required field_map keys, optional field_map keys)
ADAPTER_FIELDS: Dict[str, tuple] = {
    "text_field": ({"text"}, set()),
    "packed_tokens": ({"tokens"}, {"doc_start"}),
    "messages_list": ({"messages", "role_key", "content_key"}, set()),
    "instruction_output": ({"instruction", "output"}, {"input"}),
    "chosen_rejected_messages": ({"chosen", "rejected"}, {"role_key", "content_key"}),
    "prompt_chosen_rejected_text": ({"prompt", "chosen", "rejected"}, set()),
    "prompt_text": ({"prompt"}, set()),
    "question_answer": ({"question", "answer"}, set()),
    "code_tests": ({"prompt", "tests", "entry_point"}, set()),
    "eval_mmlu": ({"question", "choices", "answer"}, set()),
    "eval_gsm8k": ({"question", "answer"}, set()),
    "eval_math": ({"problem", "answer"}, set()),
    "eval_humaneval": ({"prompt", "test", "entry_point"}, set()),
    "eval_ifeval": ({"key", "prompt", "instruction_id_list", "kwargs"}, set()),
    "eval_alpaca": ({"instruction", "reference_output"}, set()),
    "eval_safety": ({"prompt", "label"}, set()),
}

KIND_ADAPTERS: Dict[str, set] = {
    "text": {"text_field"},
    "packed_tokens": {"packed_tokens"},
    "conversations": {"messages_list", "instruction_output"},
    "preference": {"chosen_rejected_messages", "prompt_chosen_rejected_text"},
    "prompts": {"prompt_text"},
    "rl_math": {"question_answer"},
    "rl_code": {"code_tests"},
    "eval_mmlu": {"eval_mmlu"},
    "eval_gsm8k": {"eval_gsm8k"},
    "eval_math": {"eval_math"},
    "eval_humaneval": {"eval_humaneval"},
    "eval_ifeval": {"eval_ifeval"},
    "eval_alpaca": {"eval_alpaca"},
    "eval_safety": {"eval_safety"},
}


def _check_adapter(ds: DatasetBase, kind: str):
    if ds.adapter not in KIND_ADAPTERS[kind]:
        raise ValueError(f"adapter {ds.adapter!r} not valid for kind {kind!r}; expected one of {sorted(KIND_ADAPTERS[kind])}")
    required, optional = ADAPTER_FIELDS[ds.adapter]
    keys = set(ds.field_map)
    if required - keys:
        raise ValueError(f"field_map missing keys {sorted(required - keys)} for adapter {ds.adapter!r}")
    if keys - required - optional:
        raise ValueError(f"field_map has unknown keys {sorted(keys - required - optional)} for adapter {ds.adapter!r}")
    if ds.format == "packed_tokens" and kind != "packed_tokens":
        raise ValueError("format packed_tokens is only valid for kind packed_tokens")
    return ds


def _simple_kind(kind_name: str):
    class _K(DatasetBase):
        kind: Literal[kind_name]  # type: ignore[valid-type]

        @model_validator(mode="after")
        def _v(self):
            return _check_adapter(self, self.kind)
    _K.__name__ = f"Dataset_{kind_name}"
    return _K


class RlMathDataset(DatasetBase):
    kind: Literal["rl_math"]
    answer_extraction: Literal["regex", "boxed", "raw"]
    answer_regex: Optional[str]

    @model_validator(mode="after")
    def _v(self):
        _check_adapter(self, "rl_math")
        if self.answer_extraction == "regex" and not self.answer_regex:
            raise ValueError("answer_extraction: regex requires answer_regex")
        return self


class RlCodeDataset(DatasetBase):
    kind: Literal["rl_code"]
    prompt_include_tests: int = Field(ge=0)

    @model_validator(mode="after")
    def _v(self):
        return _check_adapter(self, "rl_code")


class EvalSafetyDataset(DatasetBase):
    kind: Literal["eval_safety"]
    safe_label: str

    @model_validator(mode="after")
    def _v(self):
        return _check_adapter(self, "eval_safety")


DatasetCfg = Annotated[
    Union[
        _simple_kind("text"), _simple_kind("packed_tokens"), _simple_kind("conversations"),
        _simple_kind("preference"), _simple_kind("prompts"), RlMathDataset, RlCodeDataset,
        _simple_kind("eval_mmlu"), _simple_kind("eval_gsm8k"), _simple_kind("eval_math"),
        _simple_kind("eval_humaneval"), _simple_kind("eval_ifeval"), _simple_kind("eval_alpaca"),
        EvalSafetyDataset,
    ],
    Field(discriminator="kind"),
]


class DecontamCfg(Strict):
    ngram: int = Field(gt=0)
    overlap_threshold: float = Field(gt=0, le=1)
    against: List[str]
    # Splits of the `against` datasets whose text builds the n-gram set. Not in the PRD: DP-4 says
    # "every text field of every dataset", but Appendix A's gsm8k_test also exposes its *train*
    # split (as `fewshot`), so a literal reading drops all of gsm8k_train. See prd_review.md #18.
    splits: List[str]


class PreparedDataCfg(Strict):
    root_uri: str
    num_workers: int = Field(ge=1)
    shard_rows: int = Field(gt=0)
    holdout_fraction: float = Field(gt=0, lt=1)
    max_drop_fraction: float = Field(ge=0, le=1)
    decontamination: DecontamCfg


# --------------------------------------------------------------------------- services
class JudgePrompts(Strict):
    pairwise_uri: str
    refusal_uri: str
    scoring_uri: str


class JudgeParse(Strict):
    pairwise_regex: str
    refusal_regex: str
    score_regex: str


class JudgeCfg(Strict):
    base_url: str
    model: str
    api_key_env: str
    temperature: float
    max_tokens: int
    timeout_s: float
    max_retries: int
    max_concurrency: int
    prompts: JudgePrompts
    parse: JudgeParse


class OffPolicyTeacher(Strict):
    base_url: str
    model: str
    api_key_env: str
    temperature: float
    top_p: float
    max_tokens: int
    timeout_s: float
    max_retries: int
    max_concurrency: int


class OnPolicyTeacher(Strict):
    checkpoint_uri: str
    architecture: Architecture
    tokenizer_matches: bool


class TeachersCfg(Strict):
    offpolicy: OffPolicyTeacher
    onpolicy: OnPolicyTeacher


class CodeExecCfg(Strict):
    backend: Literal["subprocess"]
    python_executable: str
    timeout_s: float = Field(gt=0)
    memory_mb: int = Field(gt=0)
    max_open_files: int = Field(gt=0)     # SB-1 names this limit; PRD Appendix A omits the key
    max_processes: int = Field(gt=0)      # SB-1 names this limit; PRD Appendix A omits the key
    max_parallel: int = Field(gt=0)
    env_allowlist: List[str]
    isolated_host_confirmed: bool


class GenerationCfg(Strict):
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0)
    top_k: int = Field(ge=0)            # 0 = disabled
    top_p: float = Field(gt=0, le=1)     # 1.0 = disabled
    repetition_penalty: float = Field(gt=0)
    seed: int
    batch_size: int = Field(gt=0)
    use_kv_cache: bool
    overflow: Literal["error", "truncate_left"]


# --------------------------------------------------------------------------- stage building blocks
class OptimCfg(Strict):
    lr: float = Field(gt=0)
    lr_min: float = Field(ge=0)
    betas: List[float] = Field(min_length=2, max_length=2)
    eps: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    grad_clip: float = Field(ge=0)


class WsdSchedule(Strict):
    type: Literal["wsd"]
    warmup_fraction: float = Field(ge=0, le=1)
    stable_fraction: float = Field(ge=0, le=1)
    decay_fraction: float = Field(ge=0, le=1)
    decay_shape: Literal["linear", "cosine"]

    @model_validator(mode="after")
    def _sum(self):
        s = self.warmup_fraction + self.stable_fraction + self.decay_fraction
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"wsd fractions must sum to 1 (got {s})")
        return self


class SimpleSchedule(Strict):
    type: Literal["linear", "cosine", "constant"]
    warmup_fraction: float = Field(ge=0, lt=1)


ScheduleCfg = Annotated[Union[WsdSchedule, SimpleSchedule], Field(discriminator="type")]


class BatchCfg(Strict):
    micro_batch_size: int = Field(gt=0)
    grad_accum: int = Field(gt=0)


class DistributedCfg(Strict):
    zero_stage: Literal[0, 1, 2, 3]
    find_unused_parameters: bool


class MoeCfg(Strict):
    update_routing_bias: bool
    bias_update_rate: float = Field(ge=0)
    aux_loss_weight: float = Field(ge=0)


class SourceRef(Strict):
    dataset: str
    weight: float = Field(gt=0)


class StageCommon(Strict):
    enabled: bool
    init_from: str
    output_uri: str
    optim: OptimCfg
    schedule: ScheduleCfg
    batch: BatchCfg
    distributed: DistributedCfg
    moe: MoeCfg
    dropout: float = Field(ge=0, lt=1)
    checkpoint_every_steps: int = Field(gt=0)
    keep_last_checkpoints: int = Field(gt=0)
    eval_every_steps: int = Field(gt=0)


class LmData(Strict):
    sources: List[SourceRef] = Field(min_length=1)
    token_budget: float = Field(gt=0)
    retention_eval_dataset: str


class EpochData(Strict):
    sources: List[SourceRef] = Field(min_length=1)
    epochs: float = Field(gt=0)


class SourcesOnly(Strict):
    sources: List[SourceRef] = Field(min_length=1)


class MidtrainStage(StageCommon):
    use_doc_mask: bool
    data: LmData


class ArchOverride(Strict):
    ctx: int = Field(gt=0)
    rope_theta: float = Field(gt=0)


class MidtrainLongStage(MidtrainStage):
    architecture_override: ArchOverride


class SftStage(StageCommon):
    use_doc_mask: bool
    data: EpochData


class OnPolicyPrefCfg(Strict):
    enabled: bool
    prompts_dataset: str
    max_prompts: int = Field(gt=0)
    samples_per_prompt: int = Field(ge=2)
    weight: float = Field(gt=0)
    generation: GenerationCfg


class PreferenceStage(StageCommon):
    objective: Literal["dpo", "simpo"]
    beta: float = Field(gt=0)
    simpo_gamma: float
    max_total_tokens: int = Field(gt=0)
    data: EpochData
    on_policy: OnPolicyPrefCfg


class EntryGate(Strict):
    sample_prompts: int = Field(gt=0)
    k: int = Field(gt=0)
    min_pass_at_k: float = Field(ge=0, le=1)
    on_failure: Literal["skip", "stop"]


class ClipCfg(Strict):
    eps_low: float = Field(ge=0)
    eps_high: float = Field(ge=0)


class OverlongCfg(Strict):
    mode: Literal["exclude", "soft_penalty"]
    buffer_tokens: int = Field(gt=0)
    penalty_factor: float = Field(ge=0)


class DapoCfg(Strict):
    dynamic_sampling: bool
    max_resample_rounds: int = Field(ge=0)
    overlong: OverlongCfg


class MathRewardCfg(Strict):
    final_answer_regex: str


class RewardsCfg(Strict):
    correctness_weight: float
    format_weight: float
    math: MathRewardCfg


class EntropyFloor(Strict):
    value: float = Field(ge=0)
    patience_steps: int = Field(gt=0)


class SystemPrompts(Strict):
    rl_math: str
    rl_code: str


class RlvrStage(StageCommon):
    data: SourcesOnly
    system_prompts: SystemPrompts
    entry_gate: EntryGate
    prompts_per_step: int = Field(gt=0)
    group_size: int = Field(ge=2)
    updates_per_step: int = Field(gt=0)
    total_steps: int = Field(gt=0)
    adv_eps: float = Field(gt=0)
    advantage_std_normalization: bool
    clip: ClipCfg
    kl_coef: float = Field(ge=0)
    dapo: DapoCfg
    rewards: RewardsCfg
    generation: GenerationCfg
    entropy_floor: EntropyFloor
    routing_metric_sample_tokens: int = Field(gt=0)
    validation_prompts: int = Field(gt=0)


class OffPolicyDistillStage(StageCommon):
    use_doc_mask: bool
    prompts_datasets: List[str] = Field(min_length=1)
    max_prompts: int = Field(gt=0)
    samples_per_prompt: int = Field(gt=0)
    max_response_tokens: int = Field(gt=0)
    data: EpochData


class OnPolicyDistillStage(StageCommon):
    data: SourcesOnly
    prompts_per_step: int = Field(gt=0)
    total_steps: int = Field(gt=0)
    validation_prompts: int = Field(gt=0)
    generation: GenerationCfg


class DistillCfg(Strict):
    enabled: bool
    offpolicy: OffPolicyDistillStage
    onpolicy: OnPolicyDistillStage


class StagesCfg(Strict):
    midtrain: MidtrainStage
    midtrain_long: MidtrainLongStage
    sft: SftStage
    preference: PreferenceStage
    rlvr: RlvrStage
    distill: DistillCfg


# --------------------------------------------------------------------------- eval / gates / misc
class MmluBench(Strict):
    dataset: str
    max_examples: Optional[int]


class Gsm8kBench(Strict):
    dataset: str
    max_examples: Optional[int]
    n_shot: int = Field(ge=0)
    answer_regex: str
    base_prompt_uri: str
    chat_prompt_uri: str


class Math500Bench(Strict):
    dataset: str
    max_examples: Optional[int]
    chat_prompt_uri: str


class HumanEvalBench(Strict):
    dataset: str
    max_examples: Optional[int]
    n_samples: int = Field(gt=0)
    k_values: List[int] = Field(min_length=1)
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    base_stop_strings: List[str]
    chat_prompt_uri: str

    @model_validator(mode="after")
    def _k(self):
        if any(k > self.n_samples or k < 1 for k in self.k_values):
            raise ValueError("every k in k_values must be in [1, n_samples]")
        return self


class DatasetBench(Strict):
    dataset: str
    max_examples: Optional[int]


class AlpacaLcBench(Strict):
    dataset: str
    max_examples: Optional[int]
    max_iter: int = Field(gt=0)


class CalibrationBench(Strict):
    bins: int = Field(gt=0)


class LengthBench(Strict):
    dataset: str
    n_prompts: int = Field(gt=0)


class AdherenceBench(Strict):
    n_prompts: int = Field(gt=0)


class Benchmarks(Strict):
    """Every benchmark key is required; set it to null to leave that benchmark out."""
    mmlu: Optional[MmluBench]
    gsm8k: Optional[Gsm8kBench]
    math500: Optional[Math500Bench]
    humaneval: Optional[HumanEvalBench]
    ifeval: Optional[DatasetBench]
    alpaca_eval_lc: Optional[AlpacaLcBench]
    safety: Optional[DatasetBench]
    calibration: Optional[CalibrationBench]
    length: Optional[LengthBench]
    adherence: Optional[AdherenceBench]


class EvalCfg(Strict):
    benchmarks: Benchmarks


class GateThresholds(Strict):
    length_ratio_max: float = Field(gt=0)
    template_adherence_min: float = Field(ge=0, le=1)
    safety_unsafe_refusal_min: float = Field(ge=0, le=1)
    safety_safe_compliance_min: float = Field(ge=0, le=1)


STAGE_IDS = ["midtrain", "midtrain_long", "sft", "preference", "rlvr", "distill_offpolicy", "distill_onpolicy"]


class GatesCfg(Strict):
    on_failure: Literal["stop", "continue"]
    min_improvement: float
    lower_is_better: List[str]
    tolerances: Dict[str, float]
    require_improvement: Dict[str, List[str]]
    thresholds: GateThresholds

    @model_validator(mode="after")
    def _stages(self):
        unknown = set(self.require_improvement) - set(STAGE_IDS)
        if unknown:
            raise ValueError(f"require_improvement has unknown stages {sorted(unknown)}")
        return self


class ExportCfg(Strict):
    output_uri: str
    dtype: Literal["bf16", "fp16", "fp32"]


class LauncherCfg(Strict):
    nnodes: int = Field(gt=0)
    nproc_per_node: int = Field(gt=0)
    extra_torchrun_args: List[str]


class LoggingCfg(Strict):
    tensorboard: bool
    log_every_steps: int = Field(gt=0)
    moe_usage_every_steps: int = Field(gt=0)


class PFConfig(Strict):
    run: RunCfg
    storage: StorageCfg
    foundation_llm: FoundationCfg
    base_model: BaseModelCfg
    tokenizer: TokenizerCfg
    chat_template: ChatTemplateCfg
    datasets: Dict[str, DatasetCfg]
    prepared_data: PreparedDataCfg
    judge: JudgeCfg
    teachers: TeachersCfg
    code_execution: CodeExecCfg
    generation: GenerationCfg
    stages: StagesCfg
    eval: EvalCfg
    gates: GatesCfg
    export: ExportCfg
    launcher: LauncherCfg
    logging: LoggingCfg

    def stage_block(self, stage_id: str) -> StageCommon:
        if stage_id == "distill_offpolicy":
            return self.stages.distill.offpolicy
        if stage_id == "distill_onpolicy":
            return self.stages.distill.onpolicy
        return getattr(self.stages, stage_id)

    def stage_enabled(self, stage_id: str) -> bool:
        block = self.stage_block(stage_id)
        if stage_id.startswith("distill_"):
            return self.stages.distill.enabled and block.enabled
        return block.enabled
