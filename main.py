# import numpy as np
# import matplotlib.pyplot as plt
# import random

# def generate_maze(width, height):
#     # Initialize grid with walls (1 represents wall, 0 represents path)
#     maze = np.ones((height * 2 + 1, width * 2 + 1), dtype=int)

#     # Random starting point
#     start_x, start_y = random.randint(0, width - 1), random.randint(0, height - 1)
#     maze[start_y * 2 + 1, start_x * 2 + 1] = 0

#     # Stack for DFS
#     stack = [(start_x, start_y)]
#     directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

#     while stack:
#         x, y = stack[-1]
#         random.shuffle(directions)
#         found = False

#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy

#             if 0 <= nx < width and 0 <= ny < height and maze[ny * 2 + 1, nx * 2 + 1] == 1:
#                 # Break wall between current cell and next cell
#                 maze[y * 2 + 1 + dy, x * 2 + 1 + dx] = 0
#                 maze[ny * 2 + 1, nx * 2 + 1] = 0
#                 stack.append((nx, ny))
#                 found = True
#                 break

#         if not found:
#             stack.pop()

#     # Create entrance (Left border)
#     maze[1, 0] = 0

#     # Create exit (Right border)
#     maze[-2, -1] = 0

#     return maze

# def display_maze(maze):
#     plt.figure(figsize=(10, 5))
#     plt.imshow(maze, cmap='binary')
#     plt.axis('off')
#     plt.title("Generated Maze")
#     plt.show()

# if __name__ == "__main__":
#     width = int(input("Enter width of the maze: "))
#     height = int(input("Enter height of the maze: "))

#     maze = generate_maze(width, height)
#     display_maze(maze)

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

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
    maze[1, 0] = 0   # Entrance (Left border)
    maze[-2, -1] = 0 # Exit (Right border)

    return maze

def bfs(maze, start, end):
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

    # Reconstruct path
    path = []
    node = end
    while node in parent:
        path.append(node)
        node = parent[node]
    path.append(start)
    path.reverse()
    
    return path

def visualize_maze(maze, path=None):
    plt.figure(figsize=(10, 5))
    plt.imshow(maze, cmap='binary')

    if path:
        for (x, y) in path:
            plt.plot(y, x, marker='o', color='red', markersize=5)

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

    path = bfs(maze, start, end)
    visualize_maze(maze, path)
