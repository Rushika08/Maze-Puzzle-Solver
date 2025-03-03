import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import time

# Maze Generation (DFS Based)
def generate_maze(width, height):
    maze = np.ones((height * 2 + 1, width * 2 + 1), dtype=int)
    start_x, start_y = random.randint(0, width - 1), random.randint(0, height - 1)
    maze[start_y * 2 + 1, start_x * 2 + 1] = 0

    stack = [(start_x, start_y)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while stack:
        x, y = stack[-1]
        random.shuffle(directions)
        found = False

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < width and 0 <= ny < height and maze[ny * 2 + 1, nx * 2 + 1] == 1:
                maze[y * 2 + 1 + dy, x * 2 + 1 + dx] = 0
                maze[ny * 2 + 1, nx * 2 + 1] = 0
                stack.append((nx, ny))
                found = True
                break

        if not found:
            stack.pop()

    # Entrance and Exit
    maze[1, 0] = 0
    maze[-2, -1] = 0

    return maze

# BFS Algorithm
def bfs(maze, start, end):
    start_time = time.time()
    queue = deque([start])
    visited = np.zeros_like(maze)
    parent = {}
    visited[start] = 1
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        x, y = queue.popleft()

        if (x, y) == end:
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0 and not visited[nx, ny]:
                queue.append((nx, ny))
                visited[nx, ny] = 1
                parent[(nx, ny)] = (x, y)
    duration = time.time() - start_time
    print(f"BFS took {duration:.4f} seconds")
    return reconstruct_path(parent, start, end), visited

# DFS Algorithm
def dfs(maze, start, end):
    start_time = time.time()
    stack = [start]
    visited = np.zeros_like(maze)
    parent = {}
    visited[start] = 1
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while stack:
        x, y = stack.pop()

        if (x, y) == end:
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0 and not visited[nx, ny]:
                stack.append((nx, ny))
                visited[nx, ny] = 1
                parent[(nx, ny)] = (x, y)
    duration = time.time() - start_time
    print(f"DFS took {duration:.4f} seconds")
    return reconstruct_path(parent, start, end), visited

# Dijkstra's Algorithm
def dijkstra(maze, start, end):
    import heapq
    start_time = time.time()
    queue = [(0, start)]
    distances = {start: 0}
    parent = {}
    visited = np.zeros_like(maze)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        dist, (x, y) = heapq.heappop(queue)
        if visited[x, y]:
            continue
        visited[x, y] = 1

        if (x, y) == end:
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                new_distance = dist + 1
                if (nx, ny) not in distances or new_distance < distances[(nx, ny)]:
                    distances[(nx, ny)] = new_distance
                    heapq.heappush(queue, (new_distance, (nx, ny)))
                    parent[(nx, ny)] = (x, y)
    duration = time.time() - start_time
    print(f"Dijkstra took {duration:.4f} seconds")
    return reconstruct_path(parent, start, end), visited

# Path Reconstruction
def reconstruct_path(parent, start, end):
    path = []
    node = end
    while node in parent:
        path.append(node)
        node = parent[node]
    path.append(start)
    path.reverse()
    return path

# Visualization
def visualize_maze(maze, path=None, visited=None):
    plt.figure(figsize=(10, 5))
    plt.imshow(maze, cmap='binary')

    if visited is not None:
        for (x, y) in zip(*np.where(visited == 1)):
            plt.plot(y, x, marker='.', color='blue', markersize=2)
            # plt.pause(0.005)

    if path:
        for (x, y) in path:
            plt.plot(y, x, marker='o', color='red', markersize=5)
            # plt.pause(0.01)

    plt.axis('off')
    plt.title("Maze Solving Visualization")
    plt.show()

if __name__ == "__main__":
    width = int(input("Enter maze width: "))
    height = int(input("Enter maze height: "))

    maze = generate_maze(width, height)
    visualize_maze(maze)

    start = (1, 0)
    end = (maze.shape[0] - 2, maze.shape[1] - 1)

    while True:
        print("Choose Algorithm:")
        print("1. BFS")
        print("2. DFS")
        print("3. Dijkstra")
        print("4. Exit")
        choice = int(input("Enter Choice: "))

        if choice == 1:
            path, visited = bfs(maze, start, end)
        elif choice == 2:
            path, visited = dfs(maze, start, end)
        elif choice == 3:
            path, visited = dijkstra(maze, start, end)
        elif choice == 4:
            print("Exiting...")
            break
        else:
            print("Invalid Choice")
            continue

        visualize_maze(maze, path, visited)