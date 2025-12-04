FROM python:3.11-slim

# OpenCV が動作するための最低限のネイティブライブラリ
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python ライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# スクリプト本体
COPY maze_main.py .

# デフォルトでは「引数なし」（粗い＋細かい）で実行
ENTRYPOINT ["python", "maze_main.py"]
