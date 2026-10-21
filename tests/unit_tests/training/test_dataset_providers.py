# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Each GPT entry point builds the dataset config and dataset class it built before.

The reference is the frozen factory and provider logic of commit 438ef0025
(``legacy_dataset_factories.py``), run on the same validated arguments as the new path: ``cfg.data``
built under the entry point's policy, the mirrors written by the container, the config prepared
with the run's tokenizer and handed to the builder as is.
"""

import ast
import inspect
import json
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig as MCoreGPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset
from megatron.core.datasets.utils import Split, compile_helpers
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training import argument_utils, global_vars, training
from megatron.training.argument_utils import (
    gpt_data_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import parse_args, validate_args
from megatron.training.config import GPTDatasetConfig
from megatron.training.config import container as container_module
from megatron.training.datasets import sft_dataset, varlen_dataset
from megatron.training.datasets.fim_dataset import GPTFIMDatasetConfig as RuntimeGPTFIMDatasetConfig
from megatron.training.datasets.sft_dataset import SFTDataset
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.training import legacy_dataset_factories as legacy

MODEL_ARGV = (
    "--num-layers 2 --hidden-size 128 --num-attention-heads 8 --micro-batch-size 1 "
    "--global-batch-size 8 --seq-length 32 --max-position-embeddings 32 "
    "--tokenizer-type NullTokenizer --vocab-size 2048"
).split()

# Applied to the validated arguments and mirrored into the container's model and validation
# sections, as gpt_config_from_args would carry them for a real run.
PARALLEL_OVERRIDES = dict(
    context_parallel_size=2,
    tensor_model_parallel_size=2,
    sequence_parallel=True,
    data_parallel_size=4,
    hybrid_context_parallel=True,
    full_validation=True,
    multiple_validation_sets=True,
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset_providers")
    data_args = root / "data_args.txt"
    data_args.write_text("0.3 prefix-a 0.7 prefix-b\n")
    per_split = root / "per_split.json"
    per_split.write_text(
        json.dumps({"train": "prefix-a prefix-b", "valid": ["prefix-c"], "test": "prefix-d"})
    )
    sequences = root / "sequences.json"
    sequences.write_text(json.dumps({"prefix-a": 10, "prefix-b": 20}))
    return {
        "mock": (["--mock-data"], {}),
        "mock_parallel": (["--mock-data"], PARALLEL_OVERRIDES),
        "blend": (["--data-path", "0.3", "prefix-a", "0.7", "prefix-b", "--split", "9,1,0"], {}),
        "blend_default_split": (["--data-path", "prefix-a"], {}),
        "per_split": (
            ["--train-data-path", "prefix-a", "--valid-data-path", "prefix-b", "--test-data-path", "prefix-c"],
            {},
        ),
        "data_args_file": (["--data-args-path", str(data_args), "--split", "9,1,0"], {}),
        "per_split_file": (["--per-split-data-args-path", str(per_split)], {}),
        "sft_mock": (["--sft", "--mock-data"], {}),
        "sft": (["--sft", "--data-path", "prefix-a"], {}),
        "varlen_mock": (
            ["--use-varlen-dataset", "--mock-data", "--calculate-per-token-loss", "--max-seqlen-per-dp-cp-rank", "32"],
            {},
        ),
        "varlen_sbhd": (["--use-varlen-dataset", "--data-path", "prefix-a", "--varlen-sbhd-validation"], {}),
        "fim": (
            ["--fim-data", "--data-path", "prefix-a", "--fim-rate", "0.9", "--fim-spm-rate", "0.4",
             "--fim-split-sample", "<split>", "--fim-fragment-rate", "0.2", "--fim-no-prefix", "nofim",
             "--fim-prefix-token", "<p>", "--fim-eod-token", "<e>"],
            {},
        ),
        "cache": (
            ["--train-data-path", "prefix-a", "--valid-data-path", "prefix-b", "--test-data-path", "prefix-c",
             "--data-cache-path", "/cache", "--dataloader-fast-cache-load", "--dataloader-defer-npy-index-mmap"],
            {},
        ),
        "knobs": (
            ["--data-path", "prefix-a", "--phase-transition-iterations", "20,10",
             "--dataloader-inter-document-masking", "--reset-position-ids", "--reset-attention-mask",
             "--eod-mask-loss", "--no-create-attention-mask-in-dataloader", "--num-dataset-builder-threads", "3",
             "--mid-level-dataset-surplus", "0.1", "--object-storage-cache-path", "/object-cache",
             "--allow-ambiguous-pad-tokens", "--no-mmap-bin-files", "--per-dataset-sequences-path", str(sequences),
             "--sft-mock-dataset-config-json", "{}", "--varlen-mock-dataset-config-json", "{}",
             "--varlen-sbhd-validation", "--num-workers", "0", "--dataloader-type", "cyclic"],
            {},
        ),
        "sft_fim": (["--sft", "--fim-data", "--data-path", "prefix-a"], {}),
        "varlen_fim": (
            ["--use-varlen-dataset", "--fim-data", "--data-path", "prefix-a", "--calculate-per-token-loss",
             "--max-seqlen-per-dp-cp-rank", "32"],
            {},
        ),
        "implicit_mock": ([], {}),
        # Accepted by validate_args; rejected by the factory (see the dedicated test).
        "mock_data_args_file": (["--mock-data", "--data-args-path", str(data_args), "--split", "9,1,0"], {}),
        "mock_per_split_file": (["--mock-data", "--per-split-data-args-path", str(per_split)], {}),
    }  # fmt: skip


CASE_NAMES = [
    "mock", "mock_parallel", "blend", "blend_default_split", "per_split", "data_args_file",
    "per_split_file", "sft_mock", "sft", "varlen_mock", "varlen_sbhd", "fim", "cache", "knobs",
    "sft_fim", "varlen_fim", "implicit_mock",
]  # fmt: skip
MIXED_FIM_CASES = {"sft_fim", "varlen_fim"}
# Documented differences: the entry point has no mock dataset for the family, so the request is
# rejected at validation where the legacy path selected a real class around a mock config (hybrid,
# hybrid-flex: SFTDataset, which could not build) or let the builder fail (prepare_cache).
REJECTED = {
    ("hybrid", "sft_mock"): "no mock dataset for the sft family",
    ("hybrid_flex", "sft_mock"): "no mock dataset for the sft family",
    ("prepare_cache", "mock"): "no mock dataset for the gpt family",
    ("prepare_cache", "mock_parallel"): "no mock dataset for the gpt family",
    ("prepare_cache", "sft_mock"): "no mock dataset for the gpt family",
    ("prepare_cache", "varlen_mock"): "no mock dataset for the gpt family",
    ("prepare_cache", "implicit_mock"): "no mock dataset for the gpt family",
}


def _entry_point(name):
    if name == "gpt":
        import pretrain_gpt as module
    elif name == "hybrid":
        import pretrain_hybrid as module
    elif name == "hybrid_flex":
        from megatron.elastification import pretrain_hybrid_flex as module
    else:
        from tools import prepare_cache as module
    return module


ENTRY_POINTS = {
    "gpt": ("gpt", legacy.legacy_gpt_dataset_config, legacy.legacy_gpt_dataset_class),
    "hybrid": ("hybrid", legacy.legacy_hybrid_dataset_config, legacy.legacy_hybrid_dataset_class),
    "hybrid_flex": (
        "hybrid_flex",
        legacy.legacy_hybrid_flex_dataset_config,
        lambda args: (legacy.legacy_hybrid_flex_dataset_class(args), None),
    ),
    "prepare_cache": ("cache", legacy.legacy_prepare_cache_dataset_config, None),
}


def _validated(extra, overrides):
    with patch("sys.argv", ["test_dataset_providers.py", *MODEL_ARGV, *extra]):
        args = validate_args(parse_args())
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _layout_from_args(args):
    """The model section a real run derives from the same arguments (gpt_config_from_args)."""
    transformer = SimpleNamespace(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        sequence_parallel=args.sequence_parallel,
        hybrid_context_parallel=args.hybrid_context_parallel,
        gtp_weight_remat_size=args.gtp_weight_remat_size,
        gtp_remat_num_sequence_shards=args.gtp_remat_num_sequence_shards,
        cuda_graph_impl="none",
    )
    return SimpleNamespace(transformer=transformer, seq_length=args.max_position_embeddings)


def _world_size_for(args):
    shards = args.gtp_remat_num_sequence_shards
    return (
        args.tensor_model_parallel_size
        * args.pipeline_model_parallel_size
        * (args.context_parallel_size // shards)
        * args.gtp_weight_remat_size
        * args.data_parallel_size
    )


def _prepare(monkeypatch, extra, overrides, policy="gpt"):
    """Validated args, the container as the entry point builds it (not yet finalized) and the
    run's tokenizer, registered as globals."""
    args = _validated(extra, overrides)
    cfg = pretrain_cfg_container_from_args(
        args, data_cfg=gpt_data_config_from_args(args, policy=policy)
    )
    cfg.model = _layout_from_args(args)
    cfg.validation.full_validation = args.full_validation
    cfg.validation.multiple_validation_sets = args.multiple_validation_sets
    monkeypatch.setattr(container_module, "_current_world_size", lambda: _world_size_for(args))
    tokenizer = build_tokenizer(args)
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", cfg)
    monkeypatch.setattr(global_vars, "_GLOBAL_TOKENIZER", tokenizer)
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    monkeypatch.setattr(training, "get_args", lambda: args)
    return args, cfg, tokenizer


def _ready(cfg):
    """What pretrain() and initialize_megatron() do before the provider runs."""
    cfg.validate()
    cfg.finalize()
    global_vars._prepare_dataset_config(cfg)
    return cfg.data


def _tokenizer_signature(tokenizer):
    signature = {"type": type(tokenizer).__name__, "vocab_size": tokenizer.vocab_size}
    for name in ("eod", "pad", "bos", "eos", "mask", "cls", "sep"):
        try:
            signature[name] = getattr(tokenizer, name)
        except Exception:  # noqa: BLE001 - a tokenizer may not define the special token
            signature[name] = "unavailable"
    return signature


def _signature(config, names):
    signature = {name: getattr(config, name) for name in names}
    signature["tokenizer"] = _tokenizer_signature(signature["tokenizer"])
    return signature


@pytest.fixture
def warnings_log(monkeypatch):
    log = []
    monkeypatch.setattr(argument_utils, "_warn", log.append)
    return log


@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize("entry_point", list(ENTRY_POINTS))
def test_entry_points_build_the_same_dataset_config(
    monkeypatch, warnings_log, corpus, entry_point, case
):
    extra, overrides = corpus[case]
    policy, legacy_factory, legacy_classes = ENTRY_POINTS[entry_point]
    module = _entry_point(entry_point)
    args, cfg, tokenizer = _prepare(monkeypatch, extra, overrides, policy)
    if (entry_point, case) in REJECTED:
        with pytest.raises(ValueError, match=REJECTED[(entry_point, case)]):
            if entry_point == "prepare_cache":
                module.core_gpt_dataset_config_from_args(args)
            else:
                cfg.validate()
        if legacy_classes is not None:
            # The legacy path paired the real SFTDataset with a mock runtime config.
            assert legacy_classes(args)[0] is SFTDataset and legacy_factory(args).mock is True
        return
    expected = legacy_factory(args)
    if entry_point == "prepare_cache":
        actual = module.core_gpt_dataset_config_from_args(args)
        dataset_class, packed = None, None
    else:
        data = _ready(cfg)
        dataset_class, actual, packed = data.dataset_build()
        assert actual is data.dataset and actual.is_prepared and actual.tokenizer is tokenizer
    if entry_point == "gpt" and case in MIXED_FIM_CASES:
        # Documented exception: the dataset class and every shared field are unchanged; the unused
        # FIM runtime wrapper around an SFT / varlen dataset is gone.
        assert isinstance(expected, RuntimeGPTFIMDatasetConfig)
        assert not isinstance(actual, RuntimeGPTFIMDatasetConfig)
        names = [f.name for f in fields(MCoreGPTDatasetConfig)]
    else:
        assert isinstance(actual, type(expected))
        names = [f.name for f in fields(expected)]
    assert _signature(actual, names) == _signature(expected, names)
    if legacy_classes is not None:
        expected_class, expected_packed = legacy_classes(args)
        if case == "implicit_mock":
            # Documented difference: the legacy providers keyed on --mock-data and paired a mock
            # runtime config with the real dataset class; the section selects the mock class.
            assert expected_class is GPTDataset and dataset_class is MockGPTDataset
        else:
            assert dataset_class is expected_class
        if expected_packed is not None:
            assert packed is expected_packed
        wrapper = module.core_gpt_dataset_config_from_args(args)
        assert isinstance(wrapper, type(actual)) and wrapper.is_prepared
        assert _signature(wrapper, names) == _signature(actual, names)


@pytest.mark.parametrize("entry_point", ["gpt", "hybrid", "hybrid_flex"])
def test_provider_path_builds_no_second_tokenizer(monkeypatch, corpus, entry_point):
    policy = ENTRY_POINTS[entry_point][0]
    args, cfg, tokenizer = _prepare(monkeypatch, *corpus["mock"], policy)
    module = _entry_point(entry_point)
    if hasattr(module, "build_tokenizer"):
        monkeypatch.setattr(module, "build_tokenizer", Mock(side_effect=AssertionError("rebuilt")))
    data = _ready(cfg)
    _, config, _ = data.dataset_build()
    assert config.tokenizer is global_vars.get_tokenizer() is tokenizer


def test_sft_and_varlen_datasets_read_no_fim_field():
    """The mixed-FIM exception rests on this: the packed datasets never read a fim_* attribute."""
    for module in (sft_dataset, varlen_dataset):
        tree = ast.parse(inspect.getsource(module))
        attribute_reads = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr.lower().startswith("fim")
        }
        assert not attribute_reads, (module.__name__, attribute_reads)


@pytest.mark.parametrize("case", ["mock_data_args_file", "mock_per_split_file"])
@pytest.mark.parametrize("entry_point", ["gpt", "hybrid"])
def test_mock_data_next_to_a_file_backed_source_is_rejected(monkeypatch, corpus, entry_point, case):
    """validate_args accepts --mock-data next to a file-backed source. The legacy providers then
    selected MockGPTDataset (keyed on --mock-data) around a runtime config built from the file
    (mock=False), which MockGPTDataset refuses, so the combination never trained. The factory
    rejects it before the container is built."""
    extra, overrides = corpus[case]
    policy, legacy_factory, legacy_classes = ENTRY_POINTS[entry_point]
    args = _validated(extra, overrides)
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", pretrain_cfg_container_from_args(args))
    legacy_config = legacy_factory(args)
    assert legacy_config.mock is False
    assert legacy_config.blend is not None or legacy_config.blend_per_split is not None
    assert legacy_classes(args)[0] is MockGPTDataset
    with pytest.raises(ValueError, match="single data source"):
        gpt_data_config_from_args(args, policy=policy)


def test_fim_request_without_a_source_is_rejected(monkeypatch, warnings_log):
    args = _validated(["--fim-data"], {})
    data = gpt_data_config_from_args(args, policy="gpt")
    assert isinstance(data.dataset, GPTDatasetConfig) and data.dataset.blend is None
    with pytest.raises(ValueError, match="no mock dataset for the fim family"):
        data.validate()


@pytest.mark.parametrize(
    ("entry_point", "extra", "warns"),
    [
        ("hybrid", ["--fim-data", "--data-path", "prefix-a"], "does not build fim"),
        ("hybrid", ["--use-varlen-dataset", "--data-path", "prefix-a", "--calculate-per-token-loss", "--max-seqlen-per-dp-cp-rank", "32"], None),
        ("hybrid_flex", ["--fim-data", "--data-path", "prefix-a"], "does not build fim"),
        ("hybrid_flex", ["--use-varlen-dataset", "--data-path", "prefix-a", "--calculate-per-token-loss", "--max-seqlen-per-dp-cp-rank", "32"], "does not build varlen"),
        ("hybrid_flex", ["--sft", "--data-path", "prefix-a"], None),
    ],
)  # fmt: skip
def test_entry_point_policies_warn_when_they_substitute_the_family(
    monkeypatch, warnings_log, corpus, entry_point, extra, warns
):
    policy = ENTRY_POINTS[entry_point][0]
    args, cfg, _ = _prepare(monkeypatch, extra, {}, policy)
    data = _ready(cfg)
    if warns is None:
        assert warnings_log == []
        return
    assert len(warnings_log) == 1 and warns in warnings_log[0] and "as before" in warnings_log[0]
    assert data.dataset_type is GPTDataset and type(data.dataset) is GPTDatasetConfig
    assert not hasattr(data.dataset, "fim_rate")


class TestProvidersBuildMockDatasets:
    """The providers build mock datasets through the real builder, as the legacy code did, and the
    builder receives cfg.data.dataset itself."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(1, 1)
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        ("entry_point", "case"),
        [
            ("gpt", "mock"),
            ("gpt", "sft_mock"),
            ("gpt", "varlen_mock"),
            ("hybrid", "mock"),
            ("hybrid", "varlen_mock"),
            ("hybrid_flex", "mock"),
            ("hybrid_flex", "varlen_mock"),
        ],
    )
    def test_provider_matches_the_legacy_build(self, monkeypatch, corpus, entry_point, case):
        policy, legacy_factory, legacy_classes = ENTRY_POINTS[entry_point]
        args, cfg, _ = _prepare(monkeypatch, *corpus[case], policy)
        module = _entry_point(entry_point)
        legacy_class = legacy_classes(args)[0]
        expected = BlendedMegatronDatasetBuilder(
            legacy_class, (8, 8, 8), lambda: True, legacy_factory(args)
        ).build()
        data = _ready(cfg)
        actual = module.train_valid_test_datasets_provider((8, 8, 8), data_config=data)
        fallback = module.train_valid_test_datasets_provider((8, 8, 8))
        for built in (actual, fallback):
            _assert_same_datasets(expected, built)
            assert all(split.config is data.dataset for split in built)

    @pytest.mark.parametrize("case", ["sft_mock", "varlen_mock"])
    def test_packed_mock_datasets_ignore_the_fim_wrapper(self, monkeypatch, corpus, case):
        """Mixed-FIM exception: an SFT / varlen dataset builds the same samples whether its runtime
        config is the FIM wrapper pretrain_gpt used to build for --sft --fim-data or the plain config
        the provider builds now (mock datasets; validate_args rejects --fim-data with --mock-data,
        so the configs are constructed directly here)."""
        args, cfg, _ = _prepare(monkeypatch, *corpus[case])
        data = _ready(cfg)
        plain = data.dataset
        shared = {f.name: getattr(plain, f.name) for f in fields(MCoreGPTDatasetConfig) if f.init}
        wrapped = RuntimeGPTFIMDatasetConfig(
            **shared,
            fim_rate=0.9,
            fim_spm_rate=0.4,
            fim_extra_tokens={"prefix": "<p>", "middle": "<m>", "suffix": "<s>", "pad": "<x>", "eod": "<e>"},
        )  # fmt: skip
        dataset_class = legacy.legacy_gpt_dataset_class(args)[0]
        with_wrapper = BlendedMegatronDatasetBuilder(
            dataset_class, (8, 8, 8), lambda: True, wrapped
        ).build()
        without = BlendedMegatronDatasetBuilder(
            dataset_class, (8, 8, 8), lambda: True, plain
        ).build()
        _assert_same_datasets(with_wrapper, without)

    def test_legacy_mock_data_next_to_a_file_backed_source_did_not_build(self, monkeypatch, corpus):
        """The legacy outcome behind test_mock_data_next_to_a_file_backed_source_is_rejected: the
        class the legacy provider selected refuses the runtime config the legacy factory built.
        Constructed directly (the builder constructs on rank 0 before a barrier, so an exception
        there would leave the other rank waiting)."""
        extra, overrides = corpus["mock_data_args_file"]
        args = _validated(extra, overrides)
        monkeypatch.setattr(
            global_vars, "_GLOBAL_RUN_CONFIG", pretrain_cfg_container_from_args(args)
        )
        legacy_config = legacy.legacy_gpt_dataset_config(args)
        legacy_class = legacy.legacy_gpt_dataset_class(args)[0]
        assert legacy_class is MockGPTDataset and legacy_config.mock is False
        with pytest.raises(AssertionError):
            legacy_class(None, None, None, 8, Split.train, legacy_config)


def _assert_same_datasets(expected, built):
    for expected_split, split in zip(expected, built):
        assert type(split) is type(expected_split)
        assert len(split) == len(expected_split)
        sample, expected_sample = split[0], expected_split[0]
        assert sample.keys() == expected_sample.keys()
        for key in expected_sample:
            assert (sample[key] == expected_sample[key]).all(), key
