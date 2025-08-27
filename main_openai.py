# main_openai.py
import os
import sys
import json
import argparse
import mimetypes
import re
from pathlib import Path
from typing import List, Tuple, Optional, Any
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import base64
from dotenv import load_dotenv

# ---------- 常數與預設值 ----------
COLOR_MAP = {
    "crack": ("red", "red"),
    "spalling": ("orange", "darkorange"),
    "seepage": ("blue", "blue"),
    "efflorescence": ("green", "green"),
    "corrosion": ("purple", "purple"),
}

PROMPT = (
    "Detect every tunnel anomaly (crack, spalling, seepage, efflorescence, corrosion)."
    " 信心度小於0.2的不要偵測。"
    " Return JSON ONLY: [{label, box_2d:[ymin,xmin,ymax,xmax], confidence?}], 0–1000 normalized."
)

def init_client(api_key: Optional[str]) -> Any:
    """
    初始化 OpenAI 客戶端（使用 openai Python 套件）。
    1) 從 .env 載入環境變數（若存在）
    2) 若提供 api_key 參數則覆寫
    3) 回傳 OpenAI client 實例（使用 openai.OpenAI）
    """
    load_dotenv()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set (put it in a .env file or pass -k)")
    try:
        import openai
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed") from e
    client = OpenAI(api_key=key)
    print("openai version:", getattr(openai, "__version__", "unknown"))
    return client

def guess_mime(path: Path) -> str:
    """
    根據檔名或副檔名推測 MIME type，覆蓋常見影像格式。
    """
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

def call_model(client: Any, img_path: Path, prompt: str) -> str:
    # 讀檔並轉成 data:URL（不要把 base64 當成純文字塞進去）
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    mime = guess_mime(img_path)

    system_msg = (
        "You are a strict JSON responder. Only output JSON. "
        'Return as {"detections": [{"label":"crack","box_2d":[ymin,xmin,ymax,xmax],"confidence":0.85}, ...]} '
        "No markdown, no extra text."
    )

    resp = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.0,
        max_tokens=800,  # 夠用即可
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": "low"  # "auto"/"high" 也可；low 比較省成本
                        },
                    },
                ],
            },
        ],
    )
    return resp.choices[0].message.content

def extract_json(text: str) -> Optional[str]:
    """
    嘗試從模型回傳的任意文字中提取第一個 JSON 陣列或物件子字串。
    回傳匹配到的 JSON 字串或 None。
    """
    if not isinstance(text, str):
        return None
    # 移除常見非 JSON 前後標記
    # 找到第一個完整的 [...] 或 {...} 結構（非貪婪）
    patterns = [
        r'(\[.*?\])',  # array
        r'(\{.*?\})',  # object
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.DOTALL)
        if m:
            return m.group(1)
    return None

def normalize_and_filter_detections(detections: List[dict], image_size: Tuple[int, int], min_conf: float) -> List[dict]:
    """
    將模型輸出（標準化 0–1000）轉換為像素座標，並基於 min_conf 過濾。
    傳回的清單每個 item 含: label, box_px, confidence
    """
    w, h = image_size
    results: List[dict] = []
    for det in detections:
        conf_raw = det.get("confidence", det.get("score", None))
        conf_val: Optional[float] = None
        if conf_raw is not None:
            try:
                conf_val = float(conf_raw)
            except Exception:
                conf_val = None
        if (conf_val is not None) and (conf_val < min_conf):
            continue
        box = det.get("box_2d", [0, 0, 0, 0])
        try:
            y1, x1, y2, x2 = map(float, box)
        except Exception:
            y1 = x1 = y2 = x2 = 0.0
        xmin = int(x1 / 1000.0 * w)
        ymin = int(y1 / 1000.0 * h)
        xmax = int(x2 / 1000.0 * w)
        ymax = int(y2 / 1000.0 * h)
        results.append({
            "label": det.get("label", "unknown"),
            "box_px": [xmin, ymin, xmax, ymax],
            "confidence": conf_raw
        })
    return results

def draw_and_save(image: Image.Image, boxes: List[dict], out_img_path: Path) -> None:
    """
    在影像上繪製 bounding boxes 並儲存檔案。
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    for det in boxes:
        xmin, ymin, xmax, ymax = det["box_px"]
        edge_c, text_c = COLOR_MAP.get(det["label"], ("white", "white"))
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor=edge_c, facecolor="none")
        ax.add_patch(rect)
        ax.text(xmin, max(0, ymin - 5), f"{det['label']}", fontsize=12, color=text_c,
                bbox=dict(facecolor="black", alpha=0.5))
    ax.axis("off")
    plt.savefig(out_img_path, bbox_inches="tight")
    plt.close(fig)

def process_single(client: Any, img_path: Path, out_dir: Path, min_conf: float) -> Tuple[Path, Path]:
    """
    處理單張影像：呼叫模型、解析 JSON、過濾與繪製，並儲存結果。
    """
    print(f"Processing {img_path}")
    try:
        raw_text = call_model(client, img_path, PROMPT)
        print(raw_text)
    except Exception as e:
        print("Model call failed:", e)
        raw_text = "[]"

    stem = img_path.stem
    detections = []
    # 1) 直接解析
    try:
        detections = json.loads(raw_text)
        if not isinstance(detections, list):
            detections = detections.get("detections", []) if isinstance(detections, dict) else []
    except Exception as e:
        print("Failed to parse model output as JSON:", e)
        # 2) 嘗試從回傳文字中擷取 JSON 子字串再解析
        candidate = extract_json(raw_text)
        if candidate:
            try:
                detections = json.loads(candidate)
                if not isinstance(detections, list):
                    detections = detections.get("detections", []) if isinstance(detections, dict) else []
                print("Parsed JSON after extraction.")
            except Exception as e2:
                print("Extraction failed to parse JSON:", e2)
                detections = []
                # 將原始回應寫入檔案以便偵錯
                if raw_text and not raw_text.strip().startswith("[]"):
                    raw_path = out_dir / f"{stem}_raw.txt"
                    try:
                        with open(raw_path, "w", encoding="utf-8") as rf:
                            rf.write(raw_text)
                        print(f"Wrote raw model output to {raw_path}")
                    except Exception as e3:
                        print("Failed to write raw model output:", e3)
        else:
            detections = []
            if raw_text and not raw_text.strip().startswith("[]"):
                raw_path = out_dir / f"{stem}_raw.txt"
                try:
                    with open(raw_path, "w", encoding="utf-8") as rf:
                        rf.write(raw_text)
                    print(f"Wrote raw model output to {raw_path}")
                except Exception as e3:
                    print("Failed to write raw model output:", e3)

    image = Image.open(img_path)
    w, h = image.size
    filtered = normalize_and_filter_detections(detections, (w, h), min_conf)

    stem = img_path.stem
    json_path = out_dir / f"{stem}_detections.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"detections": detections, "filtered": filtered}, jf, ensure_ascii=False, indent=2)

    out_img = out_dir / f"{stem}_detect.png"
    draw_and_save(image, filtered, out_img)
    print(f"Saved: {out_img} and {json_path}")
    return json_path, out_img

def iter_images_in_dir(path: Path):
    """
    產生器：列出目錄下所有常見影像檔案（不含子目錄）
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    封裝 argparse，方便在測試時注入 argv。
    """
    parser = argparse.ArgumentParser(description="Tunnel anomaly detection CLI (OpenAI)")
    parser.add_argument("-i", "--input", required=True, help="Input image file or folder")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory (defaults to input parent or folder)")
    parser.add_argument("-k", "--api-key", default=None, help="OpenAI API key (env OPENAI_API_KEY overrides)")
    parser.add_argument("-c", "--min-confidence", type=float, default=0.2, help="Minimum confidence to keep (0-1)")
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    """
    主程式入口。
    """
    args = parse_args(argv)

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