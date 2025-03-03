import numpy as np
import matplotlib.pyplot as plt
import random

def generate_maze(width, height):
    # Initialize grid with walls (1 represents wall, 0 represents path)
    maze = np.ones((height * 2 + 1, width * 2 + 1), dtype=int)

    # Random starting point
    start_x, start_y = random.randint(0, width - 1), random.randint(0, height - 1)
    maze[start_y * 2 + 1, start_x * 2 + 1] = 0

    # Stack for DFS
    stack = [(start_x, start_y)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while stack:
        x, y = stack[-1]
        random.shuffle(directions)
        found = False

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < width and 0 <= ny < height and maze[ny * 2 + 1, nx * 2 + 1] == 1:
                # Break wall between current cell and next cell
                maze[y * 2 + 1 + dy, x * 2 + 1 + dx] = 0
                maze[ny * 2 + 1, nx * 2 + 1] = 0
                stack.append((nx, ny))
                found = True
                break

        if not found:
            stack.pop()

    # Create entrance (Left border)
    maze[1, 0] = 0

    # Create exit (Right border)
    maze[-2, -1] = 0

    return maze

def display_maze(maze):
    plt.figure(figsize=(10, 5))
    plt.imshow(maze, cmap='binary')
    plt.axis('off')
    plt.title("Generated Maze")
    plt.show()

if __name__ == "__main__":
    width = int(input("Enter width of the maze: "))
    height = int(input("Enter height of the maze: "))

    maze = generate_maze(width, height)
    display_maze(maze)
