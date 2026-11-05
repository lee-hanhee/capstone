"""Cache official Lato font files and their license for the presentation figures."""
from pathlib import Path
from urllib.request import urlretrieve

root = Path(__file__).resolve().parents[1] / "artifacts/fonts"
root.mkdir(parents=True, exist_ok=True)
for name in ["Lato-Regular.ttf", "Lato-Bold.ttf", "OFL.txt"]:
    target = root / name
    if not target.exists():
        urlretrieve("https://raw.githubusercontent.com/google/fonts/main/ofl/lato/" + name, target)
    print(target)
