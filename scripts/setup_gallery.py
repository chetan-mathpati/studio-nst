from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "gallery"
OUT.mkdir(parents=True, exist_ok=True)

files = {
    "newyork.jpg": "https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/content_data/newyork.jpg",
    "brad_pitt.jpg": "https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/content_data/brad_pitt.jpg",
    "golden_gate.jpg": "https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/content_data/golden_gate.jpg",
    "flowers.jpg": "https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/content_data/flowers.jpg",
    "picasso_seated_nude_hr.jpg": "https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/style_data/picasso_seated_nude_hr.jpg",
    "mondrian.jpg": "https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/style_data/mondrian.jpg",
    "brushstrokes.jpg": "https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/style_data/brushstrokes.jpg",
    "woman_with_hat_matisse.jpg": "https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/style_data/woman_with_hat_matisse.jpg",
}

for name, url in files.items():
    target = OUT / name
    if not target.exists():
        print("Downloading " + name)
        urlretrieve(url, target)

print("Gallery ready: " + str(OUT))
