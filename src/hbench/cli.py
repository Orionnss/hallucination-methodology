"""CLI: one command per stage. Each stage reads from disk, writes to disk and can resume."""

from __future__ import annotations

from typing import Optional

import typer

from hbench.config import load_config
from hbench.provenance import save_run_provenance

app = typer.Typer(no_args_is_help=True, add_completion=False)

ConfigOpt = typer.Option(..., "--config", "-c", help="Experiment YAML")
NamesOpt = typer.Option(None, help="Restrict to these names (repeatable)")


@app.command()
def sample(config: str = ConfigOpt, dataset: Optional[list[str]] = NamesOpt) -> None:
    from hbench.stages import sample as st

    cfg = load_config(config)
    save_run_provenance(cfg, "sample")
    st.run(cfg, datasets=dataset)


@app.command()
def generate(config: str = ConfigOpt, generator: Optional[list[str]] = NamesOpt, dataset: Optional[list[str]] = NamesOpt) -> None:
    from hbench.stages import generate as st

    cfg = load_config(config)
    save_run_provenance(cfg, "generate")
    st.run(cfg, generators=generator, datasets=dataset)


@app.command()
def judge(
    config: str = ConfigOpt,
    judge: Optional[list[str]] = NamesOpt,
    generator: Optional[list[str]] = NamesOpt,
    dataset: Optional[list[str]] = NamesOpt,
) -> None:
    from hbench.stages import judge as st

    cfg = load_config(config)
    save_run_provenance(cfg, "judge")
    st.run(cfg, judges=judge, generators=generator, datasets=dataset)


@app.command()
def aggregate(config: str = ConfigOpt) -> None:
    from hbench.stages import aggregate as st

    cfg = load_config(config)
    save_run_provenance(cfg, "aggregate")
    print(st.run(cfg).to_string())


@app.command()
def extract(config: str = ConfigOpt, generator: Optional[list[str]] = NamesOpt, dataset: Optional[list[str]] = NamesOpt) -> None:
    from hbench.stages import extract as st

    cfg = load_config(config)
    save_run_provenance(cfg, "extract")
    st.run(cfg, generators=generator, datasets=dataset)


@app.command()
def evaluate(
    config: str = ConfigOpt,
    detector: Optional[list[str]] = NamesOpt,
    generator: Optional[list[str]] = NamesOpt,
    dataset: Optional[list[str]] = NamesOpt,
) -> None:
    from hbench.stages import evaluate as st

    cfg = load_config(config)
    save_run_provenance(cfg, "evaluate")
    st.run(cfg, detectors=detector, generators=generator, datasets=dataset)


@app.command()
def report(config: str = ConfigOpt) -> None:
    from hbench.stages import report as st

    cfg = load_config(config)
    save_run_provenance(cfg, "report")
    print(st.run(cfg))


@app.command()
def parity(
    config: str = ConfigOpt,
    method: str = typer.Option(..., help="lapeigvals | icr_probe | charm"),
    generator: str = typer.Option(...),
    dtype: str = typer.Option("float32", help="float32 | bfloat16"),
    n_parity: int = typer.Option(16),
) -> None:
    from hbench.stages import parity as st

    cfg = load_config(config)
    save_run_provenance(cfg, "parity")
    print(st.run(cfg, method, generator, dtype, n_parity))


@app.command()
def units(config: str = ConfigOpt, stage: str = typer.Option(..., help="generate | judge | extract | evaluate | parity")) -> None:
    """Print one work unit per line (tab-separated names), for SLURM array jobs."""
    cfg = load_config(config)
    cells = [(g.name, d.name) for g in cfg.generators for d in cfg.datasets]
    if stage in ("generate", "extract", "evaluate"):
        rows = cells
    elif stage == "judge":
        rows = [(j.name, g, d) for j in cfg.judges for g, d in cells]
    elif stage == "parity":
        from hbench.stages.parity import METHODS

        methods = [d.name for d in cfg.detectors if d.name in METHODS]
        rows = [(m, g.name, dt) for m in methods for g in cfg.generators for dt in ("float32", "bfloat16")]
    else:
        raise typer.BadParameter(f"no array units for stage {stage!r}")
    for r in rows:
        print("\t".join(r))


if __name__ == "__main__":
    app()
