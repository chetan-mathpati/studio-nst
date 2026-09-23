import base64
import io
import sys
import tempfile
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.adain_model import AdaINModel
from core.gatys import GatysConfig, run_gatys
from core.johnson import TransformerNet
from core.vgg import build_vgg19

app = FastAPI(title="Studio NST", version="1.0.0")

app.mount("/static", StaticFiles(directory=str(ROOT / "web")), name="static")
app.mount("/assets", StaticFiles(directory=str(ROOT / "assets")), name="assets")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VGG_MODEL = None
ADAIN_MODEL = None
JOHNSON_MODEL = None


def resize(im, n):
    if max(im.size) <= n:
        return im
    s = n / max(im.size)
    return im.resize(
        (int(im.width * s), int(im.height * s)),
        Image.Resampling.LANCZOS,
    )


def b64(p):
    return "data:image/png;base64," + base64.b64encode(
        Path(p).read_bytes()
    ).decode()


def mem():
    if DEVICE.type == "cuda":
        return round(
            torch.cuda.max_memory_allocated() / 1024 / 1024,
            2,
        )
    return None


def vgg():
    global VGG_MODEL
    if VGG_MODEL is None:
        VGG_MODEL = build_vgg19(DEVICE)
    return VGG_MODEL


def adain():
    global ADAIN_MODEL
    if ADAIN_MODEL is None:
        ADAIN_MODEL = AdaINModel(
            str(ROOT / "weights/adain/vgg_normalised.pth"),
            str(ROOT / "weights/adain/decoder.pth"),
            DEVICE,
        )
    return ADAIN_MODEL


def johnson():
    global JOHNSON_MODEL
    if JOHNSON_MODEL is None:
        model = TransformerNet()
        checkpoint = torch.load(
        ROOT / "checkpoints/johnson/transformer_net.pth",
         map_location=DEVICE,
          weights_only=True,
          )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE).eval()
        JOHNSON_MODEL = model
    return JOHNSON_MODEL


@app.get("/")
def home():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }


@app.post("/api/stylize")
async def stylize(
    content: UploadFile = File(...),
    style: UploadFile = File(...),
    method: str = Form("adain"),
    resolution: int = Form(512),
    steps: int = Form(300),
    alpha: float = Form(1.0),
):
    method = method.lower()
    resolution = max(128, min(resolution, 1024))
    steps = max(10, min(steps, 1000))
    alpha = max(0, min(alpha, 1))

    if method not in {"gatys", "johnson", "adain"}:
        return JSONResponse({"error": "Unknown method"}, 400)

    content_image = resize(
        Image.open(io.BytesIO(await content.read())).convert("RGB"),
        resolution,
    )

    style_image = resize(
        Image.open(io.BytesIO(await style.read())).convert("RGB"),
        resolution,
    )

    output_path = Path(tempfile.mktemp(suffix=".png"))

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        content_tensor = (
            transforms.ToTensor()(content_image)
            .unsqueeze(0)
            .to(DEVICE)
        )

        style_tensor = (
            transforms.ToTensor()(style_image)
            .unsqueeze(0)
            .to(DEVICE)
        )

        if method == "gatys":
            config = GatysConfig(
                steps=steps,
                learning_rate=0.02,
                content_weight=1.0,
                style_weight=1e6,
            )

            out, history = run_gatys(
                vgg(),
                content_tensor,
                style_tensor,
                config,
            )

            save_image(out.clamp(0, 1).cpu(), str(output_path))

        elif method == "johnson":
            with torch.no_grad():
                out = johnson()(content_tensor).clamp(0, 1)

            save_image(out.cpu(), str(output_path))
            history = []

        else:
            with torch.no_grad():
                out = adain().stylize(
                    content_tensor,
                    style_tensor,
                    alpha=alpha,
                ).clamp(0, 1)

            save_image(out.cpu(), str(output_path))
            history = []

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        runtime = time.perf_counter() - start

        return {
            "method": method,
            "resolution": resolution,
            "steps": steps if method == "gatys" else None,
            "alpha": alpha if method == "adain" else None,
            "runtime_seconds": round(runtime, 4),
            "peak_gpu_memory_mb": mem(),
            "image": b64(output_path),
            "history": history[-1] if history else None,
        }

    except Exception as exc:
        return JSONResponse(
            {
                "error": str(exc),
                "type": type(exc).__name__,
            },
            status_code=500,
        )

    finally:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass