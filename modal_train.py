"""One bounded Modal T4 training call; download results without persistent storage.

Run: .venv/bin/modal run modal_train.py
"""

import io
from pathlib import Path
import zipfile
import yaml

import modal

app = modal.App("see2hear-baseline")
image = (modal.Image.debian_slim(python_version="3.13")
         .pip_install_from_requirements("requirements.txt")
         .add_local_dir("baseline", "/root/baseline", ignore=["__pycache__"])
         .add_local_dir("data/baseline", "/root/data/baseline")
         .add_local_dir("data/v3_320", "/root/data/v3_320")
         .add_local_dir("data/v3_224_clean", "/root/data/v3_224_clean")
         .add_local_dir("artifacts/baseline", "/root/artifacts/baseline"))


@app.function(image=image, gpu="T4", cpu=2, memory=4096,
              timeout=1800, retries=0, max_containers=1,
              scaledown_window=2)
def run_training(settings):
    from baseline.train import train

    output = Path("/tmp/results")
    if "checkpoint" in settings:
        from baseline.evaluate_saved import evaluate_saved
        evaluate_saved(settings, output)
    else:
        train(Path(settings["data_dir"]), output, settings=settings)
    (output / "config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in output.iterdir():
            archive.write(path, path.name)
    return buffer.getvalue()


@app.local_entrypoint()
def main(config: str = "experiments/v3_224/config.yaml"):
    settings = yaml.safe_load(Path(config).read_text())
    output = Path(settings["output_dir"])
    if (output / "metrics.json").exists():
        raise FileExistsError(f"Results already exist in {output}; choose a fresh output_dir")
    payload = run_training.remote(settings)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if Path(name).name != name:
                raise ValueError("Unexpected nested artifact path")
        archive.extractall(output)
    print(f"Saved trained model and results to {output}")
