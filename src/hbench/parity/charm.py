"""CHARM parity: their capture (generate() with output_attentions / output_hidden_states, batch size 1,
data_prep_2/data_collection.py) + get_data_object + get_activations vs ours (teacher-forced batch →
per-example adapter output → the same functions). Edges must be identical; node features, edge
features and activations are compared numerically."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import torch

from hbench.models import LoadedModel
from hbench.parity import ParityResult, compare
from hbench.pool.batching import build_pass_batch, to_device
from hbench.pool.interfaces import Activations
from hbench.pool.methods.charm import _charm, act_layer, charm_item

FIELDS = ("x", "edge_attr", "act")
THRESHOLD = 0.05  # their movies_att_act setting (D71)
FLIP_MARGIN = 0.01  # D82: an edge present on one side only is tolerated if its attention < 0.05 + 0.01


def run_original(lm: LoadedModel, prompts: list[list[int]], max_new_tokens: int):
    graphs, acts_mod, _, _ = _charm()
    items, answers = [], []
    n_layers = lm.model.config.get_text_config().num_hidden_layers
    for p in prompts:
        ids = torch.tensor([p], device=lm.device)
        with torch.no_grad():
            out = lm.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False,
                output_hidden_states=True, output_attentions=True, output_scores=True, return_dict_in_generate=True,
                pad_token_id=lm.tokenizer.pad_token_id, eos_token_id=sorted(lm.eos_ids),
            )
        seq = out.sequences[0, len(p) :].tolist()
        cut = next((k for k, t in enumerate(seq) if t in lm.eos_ids), None)
        answers.append(seq if cut is None else seq[: cut + 1])
        with contextlib.redirect_stdout(io.StringIO()):
            data = graphs.get_data_object(out.attentions, threshold=0.05, prompt_graph=False)
        act, _ = acts_mod.get_activations(out.hidden_states, [act_layer(n_layers)])
        items.append({"x": data.x.float().numpy(), "edge_index": data.edge_index.numpy(), "edge_attr": data.edge_attr.float().numpy(), "act": act.float().numpy()})
    return items, answers


def run_ours(lm: LoadedModel, prompts, answers, batch_size: int) -> list[dict]:
    eos = lm.eos_ids
    items = []
    for s in range(0, len(prompts), batch_size):
        p, a = prompts[s : s + batch_size], answers[s : s + batch_size]
        n = [len(x) - (1 if x and x[-1] in eos else 0) for x in a]
        keys = [str(i) for i in range(s, s + len(p))]
        b = to_device(build_pass_batch(keys, keys, p, a, n, lm.tokenizer.pad_token_id), lm.device)
        with torch.no_grad():
            out = lm.model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=b.position_ids,
                           output_hidden_states=True, output_attentions=True, use_cache=False)
        acts = Activations(hidden_states=out.hidden_states, attentions=out.attentions)
        items += [charm_item(acts, b, i) for i in range(len(p))]
    return items


def parity(lm: LoadedModel, prompts, stored_answers, max_new_tokens: int, batch_size: int, dtype: str) -> ParityResult:
    theirs, their_answers = run_original(lm, prompts, max_new_tokens)
    keep = [i for i in range(len(prompts)) if stored_answers is None or their_answers[i] == list(stored_answers[i])]
    ours = run_ours(lm, [prompts[i] for i in keep], [their_answers[i] for i in keep], batch_size)
    theirs = [theirs[i] for i in keep]
    n_flipped, t_cmp, o_cmp = 0, [], []
    for t, o in zip(theirs, ours):
        r = _align_edges(t, o)
        if r is None:
            # An edge set differing beyond threshold flips is a structural mismatch (decision 6.A: fail).
            return ParityResult("charm", dtype, len(ours), len(prompts) - len(keep), float("inf"), float("inf"), float("nan"), "fail")
        n_flipped += r[0]
        t_cmp.append(np.concatenate([t["x"].ravel(), r[1].ravel(), t["act"].ravel()]))
        o_cmp.append(np.concatenate([o["x"].ravel(), r[2].ravel(), o["act"].ravel()]))
    res = compare("charm", dtype, t_cmp, o_cmp, n_excluded=len(prompts) - len(keep))
    res.n_edges_flipped = n_flipped
    return res


def _align_edges(t: dict, o: dict):
    """(number of flipped edges, their edge_attr, our edge_attr) on the common edges, or None when an
    edge present on one side only has attention ≥ 0.05 + FLIP_MARGIN on that side (D82). The edge
    attention on the side that has it is the max of its edge_attr over heads (their thresholding sets
    values < 0.05 to 0)."""
    et = {tuple(e): k for k, e in enumerate(t["edge_index"].T.tolist())}
    eo = {tuple(e): k for k, e in enumerate(o["edge_index"].T.tolist())}
    only = [(t["edge_attr"], et[e]) for e in et.keys() - eo.keys()] + [(o["edge_attr"], eo[e]) for e in eo.keys() - et.keys()]
    if any(attr[k].max() >= THRESHOLD + FLIP_MARGIN for attr, k in only):
        return None
    common = [e for e in et if e in eo]
    return len(only), t["edge_attr"][[et[e] for e in common]], o["edge_attr"][[eo[e] for e in common]]
