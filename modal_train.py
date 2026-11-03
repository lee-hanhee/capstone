"""One bounded Modal T4 training call; download results without persistent storage.

Run: .venv/bin/modal run modal_train.py
"""

import io
from pathlib import Path
import zipfile

import modal

app = modal.App("see2hear-baseline")
image = (modal.Image.debian_slim(python_version="3.13")
         .pip_install_from_requirements("requirements.txt")
         .add_local_dir("baseline", "/root/baseline", ignore=["__pycache__"])
         .add_local_dir("data/baseline", "/data"))


@app.function(image=image, gpu="T4", cpu=2, memory=4096,
              timeout=1800, retries=0, max_containers=1,
              scaledown_window=2)
def run_training():
    from baseline.train import train

    output = Path("/tmp/results")
    train(Path("/data"), output, epochs=10, max_seconds=1500)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in output.iterdir():
            archive.write(path, path.name)
    return buffer.getvalue()


@app.local_entrypoint()
def main():
    output = Path("artifacts/baseline")
    if (output / "metrics.json").exists():
        raise FileExistsError("Results already exist; move artifacts/baseline before starting another paid run")
    payload = run_training.remote()
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if Path(name).name != name:
                raise ValueError("Unexpected nested artifact path")
        archive.extractall(output)
    print(f"Saved trained model and results to {output}")
