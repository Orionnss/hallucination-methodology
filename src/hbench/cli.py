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


if __name__ == "__main__":
    app()
