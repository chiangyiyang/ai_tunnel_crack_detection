# main.py
import os
import sys
import json
import argparse
import mimetypes
from pathlib import Path
from typing import List, Tuple, Optional, Any
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ---------- 常數與預設值 ----------
COLOR_MAP = {
    "crack": ("red", "red"),
    "spalling": ("orange", "darkorange"),
    "seepage": ("blue", "blue"),
    "efflorescence": ("green", "green"),
    "corrosion": ("purple", "purple"),
}

# 模型提示詞（英文為主），在 prompt 裡已要求回傳 JSON
PROMPT = (
    "Detect every tunnel anomaly (crack, spalling, seepage, efflorescence, corrosion)."
    " 信心度小於0.2的不要偵測。"
    " Return JSON ONLY: [{label, box_2d:[ymin,xmin,ymax,xmax], confidence?}], 0–1000 normalized."
)

# ---------- 工具函式 ----------

def init_client(api_key: Optional[str]) -> Any:
    """
    初始化 Google GenAI 客戶端。
    1) 嘗試從 .env 載入環境變數（使用 python-dotenv）
    2) 若提供 api_key，會以參數優先覆寫環境變數
    3) 若沒有找到金鑰會丟出 RuntimeError
    回傳 genai.Client 實例。
    """
    load_dotenv()  # 從 .env 載入（若存在）
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set (put it in a .env file or pass -k)")
    client = genai.Client(api_key=key)
    print("genai version:", getattr(genai, "__version__", "unknown"))
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
    """
    將影像與文字 prompt 一併送到模型，並回傳模型回應中的文字部分（預期為 JSON）。
    發生例外會向上拋出。
    """
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
    # 取出第一個候選回應的文字部分
    try:
        text = response.candidates[0].content.parts[0].text
    except Exception:
        # 若回傳格式不如預期，嘗試把整個 response 轉成字串回傳，讓呼叫端可以紀錄或解析
        text = getattr(response, "text", str(response))
    return text

def normalize_and_filter_detections(detections: List[dict], image_size: Tuple[int, int], min_conf: float) -> List[dict]:
    """
    將模型輸出（標準化 0–1000）轉換為像素座標，並基於 min_conf 過濾。
    傳回的清單每個 item 含: label, box_px, confidence
    """
    w, h = image_size
    results: List[dict] = []
    for det in detections:
        # 支援不同欄位名稱的 confidence（confidence 或 score）
        conf_raw = det.get("confidence", det.get("score", None))
        conf_val: Optional[float] = None
        if conf_raw is not None:
            try:
                conf_val = float(conf_raw)
            except Exception:
                conf_val = None
        # 若存在數值且低於門檻則跳過
        if (conf_val is not None) and (conf_val < min_conf):
            continue
        # 取得標準化座標（預期為 [ymin,xmin,ymax,xmax]，0–1000）
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
    處理單張影像：
    - 呼叫模型取得原始回應（JSON 字串）
    - 解析 JSON（若失敗，將 detections 設為空）
    - 進行座標轉換與過濾
    - 儲存 JSON 與標註圖
    回傳 (json_path, out_img_path)
    """
    print(f"Processing {img_path}")
    try:
        raw_text = call_model(client, img_path, PROMPT)
    except Exception as e:
        print("Model call failed:", e)
        raw_text = "[]"

    try:
        detections = json.loads(raw_text)
        if not isinstance(detections, list):
            # 若模型回傳物件而非清單，嘗試從欄位取出清單
            detections = detections.get("detections", []) if isinstance(detections, dict) else []
    except Exception as e:
        print("Failed to parse model output as JSON:", e)
        detections = []

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
    parser = argparse.ArgumentParser(description="Tunnel anomaly detection CLI")
    parser.add_argument("-i", "--input", required=True, help="Input image file or folder")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory (defaults to input parent or folder)")
    parser.add_argument("-k", "--api-key", default=None, help="Google API key (env GOOGLE_API_KEY overrides)")
    parser.add_argument("-c", "--min-confidence", type=float, default=0.2, help="Minimum confidence to keep (0-1)")
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    """
    主程式入口。argv 可在測試時傳入模擬參數。
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