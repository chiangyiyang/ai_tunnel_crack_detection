AI隧道裂縫偵測

簡介
本專案為範例 CLI，用於呼叫 Google Gemini 模型對隧道影像進行異常（裂縫、剝落、滲水、析鹽、腐蝕）檢測，並輸出 JSON 與標註影像。

需求
- Python 3.9+
- 依賴請參考 [`requirements.txt`](requirements.txt:1)

安裝
1. 建議建立虛擬環境：python -m venv .venv
2. 啟用虛擬環境並安裝依賴：pip install -r requirements.txt

設定
1. 複製 `.env.example` 為 `.env`，並填入 `GOOGLE_API_KEY`（或在執行時使用 `-k` 傳入）。

使用範例
單張影像：
python main.py -i path/to/image.jpg -o out_dir

目錄（多張影像）：
python main.py -i path/to/images_folder -o out_dir

可選參數
- -k / --api-key: 傳入 Google API key（會覆寫環境變數）
- -c / --min-confidence: 最小信心值（0-1），預設 0.2

輸出
- 會在輸出資料夾產生 <image>_detections.json（原始回應與過濾後結果）
- 會產生 <image>_detect.png（標註影像）

注意
- 本專案需正確的 Google Gemini API 設定才能完整執行模型呼叫

授權
MIT