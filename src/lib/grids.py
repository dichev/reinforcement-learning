from collections import deque

FOUR_WAY = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

def is_reachable(grid, start, goal, obstacles=()):
    rows, cols = grid.shape
    q, seen = deque([start]), {start}
    while q: # bfs
        pos = q.popleft()
        if pos == goal:
            return True

        cur_i, cur_j = pos
        for di, dj in FOUR_WAY:
            i, j = cur_i + di, cur_j + dj
            if 0 <= i < rows and 0 <= j < cols:
                if grid[i, j] not in obstacles and (i, j) not in seen:
                    seen.add((i, j))
                    q.append((i, j))
    return False
