# main.py
import os
import sys
import json
import argparse
import mimetypes
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from google import genai
from google.genai import types
from dotenv import load_dotenv

def init_client(api_key: str | None):
    # 嘗試從 .env 載入環境變數（使用 python-dotenv）
    load_dotenv()
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set (put it in a .env file or pass -k)")
    client = genai.Client(api_key=key)
    print("genai version:", genai.__version__)
    return client

COLOR_MAP = {
    "crack": ("red", "red"),
    "spalling": ("orange", "darkorange"),
    "seepage": ("blue", "blue"),
    "efflorescence": ("green", "green"),
    "corrosion": ("purple", "purple"),
}

PROMPT = (
    "Detect every tunnel anomaly (crack, spalling, seepage, efflorescence, corrosion).信心度小於0.2的不要偵測 "
    "Return JSON ONLY: [{label, box_2d:[ymin,xmin,ymax,xmax]}], 0–1000 normalized."
)

def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext in (".tif", ".tiff"):
        return "image/tiff"
    return "application/octet-stream"

def call_model(client, img_path: Path, prompt: str) -> str:
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    mime = guess_mime(img_path)
    img_part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
    txt_part = types.Part(text=prompt)
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[img_part, txt_part],
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    text = response.candidates[0].content.parts[0].text
    return text

def process_single(client, img_path: Path, out_dir: Path, min_conf: float):
    print(f"Processing {img_path}")
    raw_text = call_model(client, img_path, PROMPT)
    try:
        detections = json.loads(raw_text)
    except Exception as e:
        print("Failed to parse model output as JSON:", e)
        detections = []

    image = Image.open(img_path)
    w, h = image.size
    boxes_abs = []
    for det in detections:
        # 支援可選的 confidence 欄位 (confidence 或 score)
        conf = det.get("confidence", det.get("score", None))
        if conf is not None:
            try:
                conf_val = float(conf)
            except Exception:
                conf_val = None
            if conf_val is not None and conf_val < min_conf:
                continue
        y1, x1, y2, x2 = det.get("box_2d", [0, 0, 0, 0])
        boxes_abs.append({
            "label": det.get("label"),
            "box_px": [
                int(x1 / 1000 * w),
                int(y1 / 1000 * h),
                int(x2 / 1000 * w),
                int(y2 / 1000 * h)
            ],
            "confidence": conf
        })

    stem = img_path.stem
    json_path = out_dir / f"{stem}_detections.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"detections": detections, "filtered": boxes_abs}, jf, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    for det in boxes_abs:
        xmin, ymin, xmax, ymax = det["box_px"]
        edge_c, text_c = COLOR_MAP.get(det["label"], ("white", "white"))
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor=edge_c, facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, max(0, ymin - 5), f"{det['label']}", fontsize=12, color=text_c,
                bbox=dict(facecolor="black", alpha=0.5))
    ax.axis('off')
    out_img = out_dir / f"{stem}_detect.png"
    plt.savefig(out_img, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_img} and {json_path}")
    return json_path, out_img

def iter_images_in_dir(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def main():
    parser = argparse.ArgumentParser(description="Tunnel anomaly detection CLI")
    parser.add_argument("-i", "--input", required=True, help="Input image file or folder")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory (defaults to input parent or folder)")
    parser.add_argument("-k", "--api-key", default=None, help="Google API key (env GOOGLE_API_KEY overrides)")
    parser.add_argument("-c", "--min-confidence", type=float, default=0.2, help="Minimum confidence to keep (0-1)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print("Input path does not exist:", input_path)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else (input_path.parent if input_path.is_file() else input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = init_client(args.api_key)

    if input_path.is_file():
        process_single(client, input_path, out_dir, args.min_confidence)
    else:
        for p in sorted(iter_images_in_dir(input_path)):
            process_single(client, p, out_dir, args.min_confidence)

    print("Done.")

if __name__ == "__main__":
    main()