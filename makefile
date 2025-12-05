PYTHON     := python3
DOCKER_IMG := maze-image

# 
run:
	$(PYTHON) maze_create.py

# 細かさを指定（例: make run-detail N=50）
run-detail:
	@if [ -z "$(N)" ]; then \
		echo "Usage: make run-detail N=50"; \
		exit 1; \
	fi
	$(PYTHON) maze_create.py $(N)

# -----------------------------------------
#  Docker
# -----------------------------------------
docker-build:
	docker-compose build -t $(DOCKER_IMG) .

docker-run:
	docker run --rm -v "$(PWD)":/app $(DOCKER_IMG)

docker-run-detail:
	@if [ -z "$(N)" ]; then \
		echo "Usage: make docker-run-detail N=50"; \
		exit 1; \
	fi
	docker run --rm -v "$(PWD)":/app $(DOCKER_IMG) $(N)

# -----------------------------------------
#  生成物削除（画像＋Dockerイメージ）
# -----------------------------------------
clean:
	rm -f maze_coarse.png maze_fine.png maze_*.png
	@if docker images | grep -q "$(DOCKER_IMG)"; then \
		echo "Removing Docker image: $(DOCKER_IMG)"; \
		docker rmi -f $(DOCKER_IMG); \
	else \
		echo "Docker image '$(DOCKER_IMG)' not found."; \
	fi

.PHONY: run run-detail docker-build docker-run docker-run-detail clean
