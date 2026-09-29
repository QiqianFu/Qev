# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
from dataclasses import replace

import pytest
import torch

from qev.base_evaluate import make_prompt, score_codes


@pytest.mark.parametrize('hybrid',[False,True])
def test_joint_code_scoring_matches_teacher_forced_full_sequences(hybrid):
    from conftest import tiny_backbone
    if hybrid:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as LM
    else:
        from transformers import Qwen3ForCausalLM as LM
    torch.manual_seed(11)
    model=LM(tiny_backbone(32,hybrid).config).eval()
    prefix=[1,2,3,4,5]
    codes=[(11,12),(11,13),(14,15)]
    with torch.no_grad():
        actual=score_codes(model,prefix,codes,'cpu')
        expected=[]
        for seq in codes:
            logits=model(input_ids=torch.tensor([prefix+list(seq[:-1])]),use_cache=False,logits_to_keep=len(seq)).logits[0].float()
            expected.append(sum(logits[i].log_softmax(-1)[token] for i,token in enumerate(seq)))
    torch.testing.assert_close(actual,torch.stack(expected),atol=2e-5,rtol=2e-5)


def test_native_prompt_does_not_expose_gold_or_metadata(record):
    q=record.questions[0];labels=['A','B','C']
    original=make_prompt(record,q,labels,'\nAnswer:')
    altered=replace(q,label=q.candidates[-1].id,target=(0.,0.,1.))
    other=replace(record,id='hidden_id',group_id='hidden_group',source='hidden_source')
    assert make_prompt(other,altered,labels,'\nAnswer:')==original
