"""to_generate_output() against a real generate() call (ADDING_A_METHOD.md §3 case B), and the
teacher-forced batch against an unbatched forward pass."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hbench.pool.adapter import to_generate_output
from hbench.pool.batching import build_pass_batch

ATOL = 1e-5


def _model(path):
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager", torch_dtype=torch.float32).eval()
    return tok, model


def test_adapter_matches_generate(tiny_models):
    tok, model = _model(tiny_models["gen"])
    prompts = [[5, 6, 7, 8, 9, 10], [11, 12], [13, 14, 15, 16]]
    W = max(map(len, prompts))
    ids = torch.full((3, W), tok.pad_token_id)
    mask = torch.zeros((3, W), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, W - len(p) :] = torch.tensor(p)
        mask[i, W - len(p) :] = 1
    with torch.no_grad():
        gen = model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=5, do_sample=False, pad_token_id=tok.pad_token_id,
            output_hidden_states=True, output_attentions=True, output_logits=True, return_dict_in_generate=True,
        )
    seq = gen.sequences
    answers = [seq[i, W:].tolist() for i in range(3)]
    b = build_pass_batch(["a", "b", "c"], ["a", "b", "c"], prompts, answers, [len(a) for a in answers], tok.pad_token_id)
    assert torch.equal(b.input_ids, seq)
    with torch.no_grad():
        out = model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=b.position_ids,
                    output_hidden_states=True, output_attentions=True, use_cache=False)
    ad = to_generate_output(b.input_ids, W, out.hidden_states, out.attentions, out.logits)

    assert len(ad.hidden_states) == len(gen.hidden_states) == 5
    for k in range(5):
        assert len(ad.hidden_states[k]) == len(gen.hidden_states[k])
        for a, g in zip(ad.hidden_states[k], gen.hidden_states[k]):
            assert a.shape == g.shape
            real = b.attention_mask[:, : W + k][:, -a.shape[1]:].bool()  # padded prompt rows are not compared
            assert torch.allclose(a[real], g[real], atol=ATOL)
        for a, g in zip(ad.attentions[k], gen.attentions[k]):
            assert a.shape == g.shape
            rows = b.attention_mask[:, : W + k][:, -a.shape[2]:].bool()
            assert torch.allclose(a.permute(0, 2, 1, 3)[rows], g.permute(0, 2, 1, 3)[rows], atol=ATOL)
        assert torch.allclose(ad.logits[k], gen.logits[k], atol=1e-4)


def test_batched_pass_equals_single(tiny_models):
    """Padding and position IDs do not change the states at real positions."""
    tok, model = _model(tiny_models["gen"])
    prompts, answers = [[5, 6, 7, 8], [9, 10]], [[20, 21, 22], [23]]
    b = build_pass_batch(["a", "b"], ["a", "b"], prompts, answers, [3, 1], tok.pad_token_id)
    with torch.no_grad():
        hb = model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=b.position_ids, output_hidden_states=True).hidden_states
        for i in range(2):
            s = torch.tensor([prompts[i] + answers[i]])
            hs = model(input_ids=s, output_hidden_states=True).hidden_states
            real = b.attention_mask[i] .bool() & (b.input_ids[i] != -1)
            n = len(prompts[i]) + len(answers[i])
            pos = torch.arange(b.input_ids.shape[1])[b.prompt_width - len(prompts[i]) : b.prompt_width + len(answers[i])]
            for hl_b, hl_s in zip(hb, hs):
                assert torch.allclose(hl_b[i, pos], hl_s[0, :n], atol=ATOL)
    assert b.answer_mask[0].sum() == 3 and b.answer_mask[1].sum() == 1
    assert b.last_answer_pos.tolist() == [b.prompt_width + 2, b.prompt_width]
