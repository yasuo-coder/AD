"""All experiment artifacts are constrained to this package's outputs directory."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def output_path(*parts):
    base = (ROOT / 'outputs').resolve()
    result = base.joinpath(*parts).resolve()
    if not result.is_relative_to(base):
        raise ValueError('Experiment output must stay under BrainPromptPro/outputs')
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
