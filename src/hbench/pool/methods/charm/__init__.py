"""CHARM — "Neural Message-Passing on Attention Graphs for Hallucination Detection" (ICLR 2026,
https://arxiv.org/abs/2509.24770), code https://github.com/Noired/charm @ 7ab3ab7. Card: CARD.md.

Features (case B): their `get_data_object` (dataset/graphs.py; attention threshold 0.05, no
prompt→prompt edges) and `get_activations` (dataset/acts.py) run unchanged, one example at a time
(their code indexes batch element 0), on the generate-format output of the teacher-forced pass for
that example without padding. Activations of one layer: round(0.75 · L) (their 24 of 32, D71).
Classifier: their `get_transforms`, `get_model`, `train_model` (training.py) and `CHARM` (model/mp.py)
with their response-level config `configs/charm/movies_att_act.yaml`, read from their repository.
Their best-epoch checkpoint (`checkpoint_best: true`) is written to a temporary file and reloaded.
Label convention: CHARM uses y = 1 for a correct answer (`aupr_hallu = AP(1 − y, −pred)`), so it is
trained on 1 − y and we return P(hallucination) = 1 − sigmoid(logit).
Differs: validation split = 20 % stratified split of the training fold (they use fixed splits); one
seed (they average 3); features stored as float32 (they cast to float16); in-memory loaders with
their batch sizes (32 train, 64 eval) and no worker processes; no batch of a single graph (their
loss squeezes it to a scalar): a final training batch of one is dropped, the eval batch is 63 then.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from argparse import Namespace

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split

from hbench.pool.adapter import to_generate_output
from hbench.pool.interfaces import Activations, Needs, PassBatch
from hbench.pool.thirdparty import THIRD_PARTY

CHARM_DIR = THIRD_PARTY / "charm"
CONFIG = CHARM_DIR / "configs" / "charm" / "movies_att_act.yaml"
ACT_DEPTH = 0.75
os.environ.setdefault("WANDB_MODE", "disabled")


def _charm():
    """Their modules use top-level names (`dataset`, `model`, `training`); the repo is appended to
    sys.path so it never shadows installed packages."""
    if str(CHARM_DIR) not in sys.path:
        sys.path.append(str(CHARM_DIR))
    import dataset.acts as acts  # noqa: E402
    import dataset.graphs as graphs  # noqa: E402
    import training  # noqa: E402
    from torch_geometric.data import Data  # noqa: E402

    return graphs, acts, training, Data


def act_layer(n_layers: int) -> int:
    """Index into hidden_states (0 = embeddings), as their `act_layers` (24 for a 32-layer model)."""
    return int(round(ACT_DEPTH * n_layers))


def charm_item(acts: Activations, batch: PassBatch, i: int) -> dict[str, np.ndarray]:
    graphs, acts_mod, _, _ = _charm()
    sl = batch.example_slice(i)
    P = batch.prompt_len[i]
    seq = batch.input_ids[i : i + 1, sl]
    hs = tuple(h[i : i + 1, sl] for h in acts.hidden_states)
    att = tuple(a[i : i + 1, :, sl, sl] for a in acts.attentions)
    gen = to_generate_output(seq, P, hidden_states=hs, attentions=att)
    import contextlib, io

    try:
        with contextlib.redirect_stdout(io.StringIO()):  # their function prints one line per graph
            data = graphs.get_data_object(gen.attentions, threshold=0.05, prompt_graph=False)
    except RuntimeError as e:
        if "non-empty TensorList" in str(e):  # their tensor_to_sparse on a graph without any edge
            raise ValueError(f"CHARM: no attention ≥ 0.05 for sample {batch.keys[i]}; the graph has no edge") from e
        raise
    layer = act_layer(len(acts.hidden_states) - 1)
    act, _ = acts_mod.get_activations(gen.hidden_states, [layer])
    return {
        "x": data.x.float().numpy(),
        "edge_index": data.edge_index.numpy(),
        "edge_attr": data.edge_attr.float().numpy(),
        "head": data.head.numpy(),
        "layer": data.layer.numpy(),
        "response_index": data.response_index.numpy(),
        "prompt_len": np.int64(data.prompt_len),
        "act": act.float().numpy(),
        "act_layer": np.int64(layer),
    }


class CharmGraph:
    name = "charm_graph"
    needs = Needs(hidden_states=True, attentions=True)
    fitted = False
    ragged = True

    def reduce(self, acts: Activations, batch: PassBatch) -> list[dict]:
        return [charm_item(acts, batch, i) for i in range(len(batch.keys))]


def load_config() -> Namespace:
    return Namespace(**yaml.safe_load(CONFIG.read_text()))


def to_data(item: dict, y: float | None, idx: int):
    _, _, _, Data = _charm()
    d = Data(
        x=torch.tensor(item["x"]),
        edge_index=torch.tensor(item["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(item["edge_attr"]),
        head=torch.tensor(item["head"]),
        layer=torch.tensor(item["layer"]),
        response_index=torch.tensor(item["response_index"], dtype=torch.long),
        prompt_len=int(item["prompt_len"]),
    )
    d.act = torch.tensor(item["act"])
    d.data_idx = torch.tensor(idx)
    d.y = torch.tensor(0.0 if y is None else float(y))
    return d


def _eval_bs(n: int, bs: int = 64) -> int:
    """Their evaluation batch size (64), or 63 when 64 would leave a last batch of one graph."""
    return bs - 1 if n % bs == 1 and n > 1 else bs


class CharmClassifier:
    name = "charm_gnn"

    def __init__(self, seed: int = 0, val_size: float = 0.2):
        self.seed, self.val_size = seed, val_size
        self.args = load_config()

    def _prepare(self, X, y_charm=None):
        _, _, training, _ = _charm()
        T, *_ = training.get_transforms(self.args)
        return [T(to_data(it, None if y_charm is None else y_charm[i], i)) for i, it in enumerate(X)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CharmClassifier":
        from torch_geometric import seed_everything
        from torch_geometric.loader import DataLoader

        _, _, training, _ = _charm()
        a = self.args
        seed_everything(self.seed)
        y_charm = 1.0 - np.asarray(y, dtype=float)  # their convention: 1 = correct
        data = self._prepare(X, y_charm)
        tr, va = train_test_split(np.arange(len(y)), test_size=self.val_size, stratify=y, random_state=self.seed)
        train_list, val_list = [data[i] for i in tr], [data[i] for i in va]
        # Their loss is criterion(out.squeeze(), y): a batch of one graph breaks it (scalar vs [1]), so a
        # final training batch of one is dropped and the evaluation batch size avoids a remainder of one.
        train_loader = DataLoader(train_list, batch_size=a.batch_size, shuffle=True, drop_last=len(train_list) % a.batch_size == 1)
        val_loader = DataLoader(val_list, batch_size=_eval_bs(len(val_list)), shuffle=False)
        dims = {"act": train_list[0]["act"].shape[-1]}
        model = training.get_model(a, train_list, dims)
        tmp = tempfile.mkdtemp(prefix="hbench-charm-")
        try:
            pack = {"checkpoint_path": os.path.join(tmp, "best.pt"), "args": a, "git_sha": "7ab3ab7f7a49fd6ca23defc44ddcd6c794598c61"}
            # Their loop also evaluates a "test" loader each epoch, for logging only: the validation loader is passed.
            self.result = training.train_model(
                model, train_loader, val_loader, val_loader, num_epochs=a.num_epochs, lr=a.learning_rate, scheduler=a.scheduler,
                weight_decay=a.weight_decay, weight_decay_target=a.weight_decay_target, weight_decay_for_target=a.weight_decay_for_target,
                balance=a.balance, train_prebatcher=None, patience=a.patience, log=False, checkpoint_pack=pack, verbose=False,
            )
            state = torch.load(pack["checkpoint_path"], weights_only=False)["model_state_dict"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        model.load_state_dict(state)
        self.model = model.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        from torch_geometric.loader import DataLoader

        _, _, training, _ = _charm()
        device = next(self.model.parameters()).device
        outs = []
        with torch.no_grad():
            for data in DataLoader(self._prepare(X), batch_size=_eval_bs(len(X)), shuffle=False):
                data = data.to(device)
                outs.append(self.model(*training.prepare_args(data)).squeeze(-1).float().cpu())
        logit = torch.cat(outs).numpy()
        return 1.0 - 1.0 / (1.0 + np.exp(-logit))  # P(hallucination) = 1 − P(correct)
