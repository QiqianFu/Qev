# Adapted for Qev in 2026; see NOTICE and provenance.json.
import os
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from qev.encoding import Encoder, Limits, SPECIAL
from qev.model import QevModel, ModelSpec
from qev.schema import typed_record

torch.set_num_threads(1)


def tiny_tokenizer():
    symbols = ["<pad>", "<unk>", *SPECIAL, *sorted(pre_tokenizers.ByteLevel.alphabet())]
    tokenizer = Tokenizer(models.BPE({s: i for i, s in enumerate(symbols)}, [], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>",
                                  additional_special_tokens=list(SPECIAL))


def tiny_backbone(vocab, hybrid=False):
    from qev.execution import configure_fp32
    configure_fp32()
    common = dict(vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                  attention_dropout=0.0, pad_token_id=0)
    if hybrid:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        config = Qwen3_5TextConfig(**common, layer_types=["linear_attention", "full_attention"],
                                   linear_num_key_heads=2, linear_num_value_heads=2,
                                   linear_key_head_dim=16, linear_value_head_dim=16,
                                   rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                                    "partial_rotary_factor": 1.0,
                                                    "mrope_section": [2, 3, 3], "mrope_interleaved": True})
        config._attn_implementation = "eager"
        return Qwen3_5TextModel(config)
    config = Qwen3Config(**common)
    config._attn_implementation = "eager"
    return Qwen3Model(config)


@pytest.fixture
def tokenizer():
    return tiny_tokenizer()


@pytest.fixture
def encoder(tokenizer):
    return Encoder(tokenizer, Limits(128, 128, 128, 384, 32))


@pytest.fixture
def record():
    return typed_record({"state": "The signal is blue.", "questions": {
        "color": {"type": "choice", "instructions": "Choose its color.",
                  "criteria": {"blue": "blue", "red": "red", "green": "green"}, "label": "blue"},
        "known": {"type": "noul", "instructions": "Is the color stated?", "label": True},
    }}, source="synthetic", record_id="r0", group_id="g0")


@pytest.fixture(params=[False, True], ids=["qwen3", "qwen35_hybrid"])
def model(request, tokenizer):
    torch.manual_seed(123)
    backbone = tiny_backbone(len(tokenizer), request.param)
    spec = ModelSpec("tiny", head_dim=32, head_heads=4, head_layers=1, lora_rank=0,
                     weights_dtype="fp32", attention="eager", rows_per_forward=2)
    device = os.environ.get("QEV_TEST_DEVICE", "cpu")
    return QevModel(backbone, spec, tokenizer.pad_token_id).to(device).eval()
