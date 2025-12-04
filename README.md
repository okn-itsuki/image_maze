
# ローカルで粗い＋細かい
make run

# ローカルで細かさ 50
make run-detail N=50

# Docker イメージ作成
make docker-build

# Dockerで粗い＋細かい
make docker-run

# Dockerで細かさ 80
make docker-run-detail N=80

# 画像・Dockerイメージをまとめて削除
make clean# image_maze
