#!/usr/bin/env python3
import sys
import os
import random

import cv2
import numpy as np

# Debug時書き換える部分
IMAGE_PATH = "target_image.png"


def load_mask(path: str, thresh: int = 200) -> np.ndarray:
    abs_path = os.path.abspath(path)
    if not os.path.exists(path):
        print(f"ERROR : Image not found: {abs_path}", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR : Failed to load image via OpenCV: {abs_path}", file=sys.stderr)
        sys.exit(1)

    _, bin_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    white_ratio = np.mean(bin_img == 255)
    if white_ratio > 0.5:
        bin_img = cv2.bitwise_not(bin_img)

    mask = bin_img == 255
    return mask


def resize_mask_to_cells(mask: np.ndarray, target_long_side_cells: int) -> np.ndarray:
    h, w = mask.shape
    if h >= w:
        scale = target_long_side_cells / h
    else:
        scale = target_long_side_cells / w

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    mask_u8 = mask.astype(np.uint8) * 255
    resized = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cell_mask = resized > 0  
    return cell_mask


def generate_maze(cell_mask: np.ndarray) -> np.ndarray:
    h_cells, w_cells = cell_mask.shape

    maze_h = 2 * h_cells + 1
    maze_w = 2 * w_cells + 1
    maze = np.zeros((maze_h, maze_w), dtype=np.uint8)

    visited = np.zeros_like(cell_mask, dtype=bool)

    def neighbors(y, x):
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h_cells and 0 <= nx < w_cells:
                if cell_mask[ny, nx]:
                    yield ny, nx

    cell_indices = [(y, x) for y in range(h_cells) for x in range(w_cells) if cell_mask[y, x]]
    random.shuffle(cell_indices)

    for sy, sx in cell_indices:
        if visited[sy, sx]:
            continue

        stack = [(sy, sx)]
        visited[sy, sx] = True
        maze[2 * sy + 1, 2 * sx + 1] = 255 

        while stack:
            cy, cx = stack[-1]
            nbs = [(ny, nx) for ny, nx in neighbors(cy, cx) if not visited[ny, nx]]
            if not nbs:
                stack.pop()
                continue

            ny, nx = random.choice(nbs)
            visited[ny, nx] = True

            wall_y = cy + ny + 1
            wall_x = cx + nx + 1
            maze[wall_y, wall_x] = 255
            maze[2 * ny + 1, 2 * nx + 1] = 255

            stack.append((ny, nx))

    return maze


def save_maze(maze: np.ndarray, out_path: str, scale: int = 4) -> None:
    h, w = maze.shape
    if scale > 1:
        maze_large = cv2.resize(
            maze,
            (w * scale, h * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        maze_large = maze

    cv2.imwrite(out_path, maze_large)
    print(f"INFO : saved: {out_path}")


def run_default():
    coarse_cells = 30
    fine_cells = 70

    print("INFO : loading image...")
    mask = load_mask(IMAGE_PATH)

    print(f"INFO : generating coarse maze (cells={coarse_cells})...")
    coarse_cell_mask = resize_mask_to_cells(mask, coarse_cells)
    coarse_maze = generate_maze(coarse_cell_mask)
    save_maze(coarse_maze, "maze_coarse.png")

    print(f"INFO : generating fine maze (cells={fine_cells})...")
    fine_cell_mask = resize_mask_to_cells(mask, fine_cells)
    fine_maze = generate_maze(fine_cell_mask)
    save_maze(fine_maze, "maze_fine.png")


def run_with_detail(detail: int):
    print(f"INFO : loading image...")
    mask = load_mask(IMAGE_PATH)

    cells = detail
    print(f"INFO : generating maze (cells={cells})...")
    cell_mask = resize_mask_to_cells(mask, cells)
    maze = generate_maze(cell_mask)
    out_name = f"maze_{cells}.png"
    save_maze(maze, out_name)


def main():
    if len(sys.argv) == 1:
        run_default()
    elif len(sys.argv) == 2:
        try:
            detail = int(sys.argv[1])
        except ValueError:
            print("Usage: python maze_main.py [1-100]", file=sys.stderr)
            sys.exit(1)

        if not (1 <= detail <= 100):
            print("Error: detail must be between 1 and 100", file=sys.stderr)
            sys.exit(1)

        run_with_detail(detail)
    else:
        print("Usage:", file=sys.stderr)
        print("  python maze_main.py          # 粗い＋細かい迷路を生成", file=sys.stderr)
        print("  python maze_main.py N        # N=1〜100 の細かさで1枚生成", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
