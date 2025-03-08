import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import time
import heapq
from PIL import Image
import cv2

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

# Image to Maze Conversion
def image_to_maze(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the image. Please check the file path.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to handle varying lighting conditions
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Apply morphological operations to fill gaps and connect paths
    kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size as needed
    dilated = cv2.dilate(binary, kernel, iterations=1)  # Connect paths
    eroded = cv2.erode(dilated, kernel, iterations=1)  # Smooth paths

    # Debug: Show the processed binary maze
    cv2.imshow("Processed Binary Maze", eroded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find contours in the processed binary image
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise) based on area
    min_contour_area = 100  # Adjust this threshold as needed
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Combine all filtered contours into a single contour
    combined_contour = np.vstack(filtered_contours)

    # Find the bounding rectangle of the combined contour
    x, y, w, h = cv2.boundingRect(combined_contour)

    # Define the four corners of the bounding rectangle
    corners = np.array([
        [x, y],  # Top-left
        [x + w, y],  # Top-right
        [x + w, y + h],  # Bottom-right
        [x, y + h]  # Bottom-left
    ])

    # Define the target dimensions for the straightened maze
    width, height = 400, 400  # Adjust as needed
    target_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Compute the perspective transform matrix and apply it
    transform_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target_corners)
    straightened = cv2.warpPerspective(image, transform_matrix, (width, height))

    # Convert to grayscale and threshold to binary
    gray_straightened = cv2.cvtColor(straightened, cv2.COLOR_BGR2GRAY)
    _, binary_maze = cv2.threshold(gray_straightened, 128, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to ensure paths are fully connected
    kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size as needed
    binary_maze = cv2.dilate(binary_maze, kernel, iterations=1)  # Connect paths
    binary_maze = cv2.erode(binary_maze, kernel, iterations=1)  # Smooth paths

    # Debug: Show the final binary maze
    cv2.imshow("Final Binary Maze", binary_maze)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert to 0s and 1s (0 = path, 1 = wall)
    binary_maze = np.where(binary_maze > 128, 1, 0)

    # Use the final binary image as the maze
    maze = binary_maze

    # Identify start and end points (no border on the left and right sides)
    start_point = None
    end_point = None

    # Check the left side for the start point
    for i in range(maze.shape[0]):
        if maze[i, 0] == 0:
            start_point = (i, 0)
            break

    # Check the right side for the end point
    for i in range(maze.shape[0]):
        if maze[i, -1] == 0:
            end_point = (i, maze.shape[1] - 1)
            break

    if start_point is None or end_point is None:
        raise ValueError("Could not detect start or end points. Please ensure the maze has clear entry and exit points.")

    # Add entrance and exit
    maze[start_point[0], start_point[1]] = 0  # Entrance
    maze[end_point[0], end_point[1]] = 0  # Exit

    return maze, start_point, end_point

## Below function worked properly but the number of cells are more

# import cv2
# import numpy as np
# from PIL import Image

# def image_to_maze(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Could not load the image. Please check the file path.")

#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply GaussianBlur to reduce noise and improve edge detection
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Use adaptive thresholding to handle varying lighting conditions
#     binary = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
#     )

#     # Apply morphological operations to fill gaps and connect paths
#     kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size as needed
#     dilated = cv2.dilate(binary, kernel, iterations=1)  # Connect paths
#     eroded = cv2.erode(dilated, kernel, iterations=1)  # Smooth paths

#     # Debug: Show the processed binary maze
#     cv2.imshow("Processed Binary Maze", eroded)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Find contours in the processed binary image
#     contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Filter out small contours (noise) based on area
#     min_contour_area = 100  # Adjust this threshold as needed
#     filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

#     # Combine all filtered contours into a single contour
#     combined_contour = np.vstack(filtered_contours)

#     # Find the bounding rectangle of the combined contour
#     x, y, w, h = cv2.boundingRect(combined_contour)

#     # Define the four corners of the bounding rectangle
#     corners = np.array([
#         [x, y],  # Top-left
#         [x + w, y],  # Top-right
#         [x + w, y + h],  # Bottom-right
#         [x, y + h]  # Bottom-left
#     ])

#     # Define the target dimensions for the straightened maze
#     width, height = 400, 400  # Adjust as needed
#     target_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

#     # Compute the perspective transform matrix and apply it
#     transform_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target_corners)
#     straightened = cv2.warpPerspective(image, transform_matrix, (width, height))

#     # Convert to grayscale and threshold to binary
#     gray_straightened = cv2.cvtColor(straightened, cv2.COLOR_BGR2GRAY)
#     _, binary_maze = cv2.threshold(gray_straightened, 128, 255, cv2.THRESH_BINARY_INV)

#     # Apply morphological operations to ensure paths are fully connected
#     kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size as needed
#     binary_maze = cv2.dilate(binary_maze, kernel, iterations=1)  # Connect paths
#     binary_maze = cv2.erode(binary_maze, kernel, iterations=1)  # Smooth paths

#     # Debug: Show the final binary maze
#     cv2.imshow("Final Binary Maze", binary_maze)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Convert to 0s and 1s (0 = path, 1 = wall)
#     binary_maze = np.where(binary_maze > 128, 1, 0)

#     # Downsample the maze to ensure one cell between walls
#     # Calculate the new grid size
#     grid_height = binary_maze.shape[0] // 2
#     grid_width = binary_maze.shape[1] // 2

#     # Initialize the downsampled maze
#     downsampled_maze = np.ones((grid_height, grid_width), dtype=int)

#     # Fill the downsampled maze
#     for i in range(grid_height):
#         for j in range(grid_width):
#             # Extract the 2x2 block
#             block = binary_maze[
#                 i * 2 : (i + 1) * 2,
#                 j * 2 : (j + 1) * 2
#             ]
#             # If any cell in the block is a path, mark it as 0 (path)
#             if np.any(block == 0):
#                 downsampled_maze[i, j] = 0

#     # Debug: Show the downsampled maze
#     plt.imshow(downsampled_maze, cmap='binary')
#     plt.title("Downsampled Maze")
#     plt.show()

#     # Identify start and end points (no border on the left and right sides)
#     start_point = None
#     end_point = None

#     # Check the left side for the start point
#     for i in range(downsampled_maze.shape[0]):
#         if downsampled_maze[i, 0] == 0:
#             start_point = (i, 0)
#             break

#     # Check the right side for the end point
#     for i in range(downsampled_maze.shape[0]):
#         if downsampled_maze[i, -1] == 0:
#             end_point = (i, downsampled_maze.shape[1] - 1)
#             break

#     if start_point is None or end_point is None:
#         raise ValueError("Could not detect start or end points. Please ensure the maze has clear entry and exit points.")

#     # Add entrance and exit
#     downsampled_maze[start_point[0], start_point[1]] = 0  # Entrance
#     downsampled_maze[end_point[0], end_point[1]] = 0  # Exit

#     return downsampled_maze, start_point, end_point


# BFS Algorithm with Visualization
def bfs(maze, start, end, visualize=True):
    start_time = time.time()
    queue = deque([start])
    visited = np.zeros_like(maze)
    parent = {}
    visited[start] = 1
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    plt.figure(figsize=(10, 5))
    plt.imshow(maze, cmap='binary')
    plt.title("BFS Step-by-Step Visualization")
    plt.axis('off')

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
                # Fill the cell with color
                plt.fill([ny - 0.5, ny + 0.5, ny + 0.5, ny - 0.5],
                         [nx - 0.5, nx - 0.5, nx + 0.5, nx + 0.5], color='blue')
                
                # Pause for visualization
                if visualize:
                    plt.pause(0.005)

    duration = time.time() - start_time
    print(f"BFS took {duration:.4f} seconds")
    return reconstruct_path(parent, start, end), visited

# DFS Algorithm with Visualization
def dfs(maze, start, end, visualize=True):
    start_time = time.time()
    stack = [start]
    visited = np.zeros_like(maze)
    parent = {}
    visited[start] = 1
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    plt.figure(figsize=(10, 5))
    plt.imshow(maze, cmap='binary')
    plt.title("DFS Step-by-Step Visualization")
    plt.axis('off')

    while stack:
        x, y = stack.pop()

        # If we reach the end, stop
        if (x, y) == end:
            break

        # Explore neighbors in random order (to make DFS more interesting)
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check if the neighbor is within bounds, is a path, and is unvisited
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0 and not visited[nx, ny]:
                stack.append((nx, ny))
                visited[nx, ny] = 1
                parent[(nx, ny)] = (x, y)
                # Fill the cell with color
                plt.fill([ny - 0.5, ny + 0.5, ny + 0.5, ny - 0.5],
                         [nx - 0.5, nx - 0.5, nx + 0.5, nx + 0.5], color='blue')
                
                # Pause for visualization
                if visualize:
                    plt.pause(0.005)

    duration = time.time() - start_time
    print(f"DFS took {duration:.4f} seconds")
    return reconstruct_path(parent, start, end), visited

# Dijkstra's Algorithm with Visualization
def dijkstra(maze, start, end, visualize=True):
    import heapq
    start_time = time.time()
    queue = [(0, start)]
    distances = {start: 0}
    parent = {}
    visited = np.zeros_like(maze)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    plt.figure(figsize=(10, 5))
    plt.imshow(maze, cmap='binary')
    plt.title("Dijkstra's Step-by-Step Visualization")
    plt.axis('off')

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
                    # Fill the cell with color
                    plt.fill([ny - 0.5, ny + 0.5, ny + 0.5, ny - 0.5],
                             [nx - 0.5, nx - 0.5, nx + 0.5, nx + 0.5], color='blue')
                    
                    # Pause for visualization
                    if visualize:
                        plt.pause(0.005)

    duration = time.time() - start_time
    print(f"Dijkstra took {duration:.4f} seconds")
    return reconstruct_path(parent, start, end), visited

# Heuristic functions
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def chebyshev_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def diagonal_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    D = 1  # Cost of horizontal/vertical movement
    D2 = 1.414  # Cost of diagonal movement (sqrt(2))
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def octile_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    D = 1  # Cost of horizontal/vertical movement
    D2 = 1.414  # Cost of diagonal movement (sqrt(2))
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

# A* Algorithm with Visualization
def astar(maze, start, end, heuristic_func, visualize=True):
    start_time = time.time()
    queue = [(0, start)]  # Priority queue: (f(n), (x, y))
    g_values = {start: 0}  # Cost from start to current node
    parent = {}
    visited = np.zeros_like(maze)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    plt.figure(figsize=(10, 5))
    plt.imshow(maze, cmap='binary')
    plt.title("A* Step-by-Step Visualization")
    plt.axis('off')

    while queue:
        _, (x, y) = heapq.heappop(queue)

        if (x, y) == end:
            break

        if visited[x, y]:
            continue
        visited[x, y] = 1

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                new_g = g_values[(x, y)] + 1  # Cost to move to neighbor
                if (nx, ny) not in g_values or new_g < g_values[(nx, ny)]:
                    g_values[(nx, ny)] = new_g
                    f_value = new_g + heuristic_func((nx, ny), end)  # f(n) = g(n) + h(n)
                    heapq.heappush(queue, (f_value, (nx, ny)))
                    parent[(nx, ny)] = (x, y)
                    # Fill the cell with color
                    plt.fill([ny - 0.5, ny + 0.5, ny + 0.5, ny - 0.5],
                             [nx - 0.5, nx - 0.5, nx + 0.5, nx + 0.5], color='blue')
                    
                    # Pause for visualization
                    if visualize:
                        plt.pause(0.005)

    duration = time.time() - start_time
    print(f"A* took {duration:.4f} seconds")
    return reconstruct_path(parent, start, end), visited

def ant_colony_optimization(maze, start, end, num_ants=10, max_iter=100, evaporation_rate=0.5, alpha=1, beta=2):
    start_time = time.time()
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    pheromones = np.ones_like(maze, dtype=float)  # Initialize pheromones

    # Initialize plot
    plt.figure(figsize=(10, 5))
    plt.imshow(maze, cmap='binary')
    plt.title("Ant Colony Optimization Step-by-Step Visualization")
    plt.axis('off')
    plt.draw()
    plt.pause(0.1)  # Initial pause to show the maze

    best_path = None
    best_path_length = float('inf')

    for iteration in range(max_iter):
        paths = []
        path_lengths = []

        for ant in range(num_ants):
            current = start
            path = [current]
            visited = set([current])

            while current != end:
                x, y = current
                neighbors = []

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0 and (nx, ny) not in visited:
                        neighbors.append((nx, ny))

                if not neighbors:
                    break  # Dead end

                # Calculate probabilities based on pheromones and heuristic (Manhattan distance)
                probabilities = []
                for neighbor in neighbors:
                    pheromone = pheromones[neighbor]
                    distance = abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                    probabilities.append((pheromone ** alpha) * ((1 / (distance + 1)) ** beta))

                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()

                # Choose the next node
                next_node = neighbors[np.random.choice(len(neighbors), p=probabilities)]
                path.append(next_node)
                visited.add(next_node)
                current = next_node

            if current == end:
                paths.append(path)
                path_lengths.append(len(path))

                if len(path) < best_path_length:
                    best_path = path
                    best_path_length = len(path)

        # Update pheromones
        pheromones *= evaporation_rate  # Evaporate pheromones
        for path, length in zip(paths, path_lengths):
            for node in path:
                pheromones[node] += 1 / length

        # Visualization
        plt.clf()
        plt.imshow(maze, cmap='binary')
        plt.title(f"Ant Colony Optimization (Iteration {iteration + 1})")
        plt.axis('off')

        # Plot explored cells (pheromone trails)
        for (x, y) in zip(*np.where(pheromones > 1)):
            plt.fill([y - 0.5, y + 0.5, y + 0.5, y - 0.5],
                     [x - 0.5, x - 0.5, x + 0.5, x + 0.5], color='blue', alpha=min(pheromones[x, y] / 10, 1))

        # Plot the best path
        if best_path:
            for (x, y) in best_path:
                plt.fill([y - 0.5, y + 0.5, y + 0.5, y - 0.5],
                         [x - 0.5, x - 0.5, x + 0.5, x + 0.5], color='red')

        plt.draw()
        plt.pause(0.1)  # Pause to show the updated plot

        # Stop if a path is found
        if best_path:
            print(f"Path found in iteration {iteration + 1}. Stopping early.")
            break

    duration = time.time() - start_time
    print(f"Ant Colony Optimization took {duration:.4f} seconds")
    return best_path, pheromones

# Not working properly

# def jump_point_search(maze, start, end):
#     start_time = time.time()
#     directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
#     open_set = [(0, start)]
#     heapq.heapify(open_set)
#     g_values = {start: 0}
#     parent = {}
#     visited = np.zeros_like(maze)

#     plt.figure(figsize=(10, 5))
#     plt.imshow(maze, cmap='binary')
#     plt.title("Jump Point Search Step-by-Step Visualization")
#     plt.axis('off')

#     def jump(x, y, dx, dy):
#         nx, ny = x + dx, y + dy
#         if not (0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0):
#             return None  # Out of bounds or blocked

#         if (nx, ny) == end:
#             return (nx, ny)  # Goal reached

#         # Check for forced neighbors
#         if dx != 0 and dy != 0:  # Diagonal movement
#             if (maze[x, ny] == 1 and maze[nx, y] == 0) or (maze[x, ny] == 1 and maze[nx, y] == 0):
#                 return (nx, ny)  # Forced neighbor found
#         else:  # Horizontal or vertical movement
#             if dx == 0:  # Vertical movement
#                 if (maze[x + 1, ny] == 1 and maze[x + 1, ny + dy] == 0) or (maze[x - 1, ny] == 1 and maze[x - 1, ny + dy] == 0):
#                     return (nx, ny)  # Forced neighbor found
#             else:  # Horizontal movement
#                 if (maze[nx, y + 1] == 1 and maze[nx + dx, y + 1] == 0) or (maze[nx, y - 1] == 1 and maze[nx + dx, y - 1] == 0):
#                     return (nx, ny)  # Forced neighbor found

#         # Recursively jump in the same direction
#         return jump(nx, ny, dx, dy)

#     while open_set:
#         _, (x, y) = heapq.heappop(open_set)

#         if (x, y) == end:
#             break

#         if visited[x, y]:
#             continue
#         visited[x, y] = 1

#         for dx, dy in directions:
#             result = jump(x, y, dx, dy)
#             if result is not None:
#                 jx, jy = result
#                 new_g = g_values[(x, y)] + abs(jx - x) + abs(jy - y)
#                 if (jx, jy) not in g_values or new_g < g_values[(jx, jy)]:
#                     g_values[(jx, jy)] = new_g
#                     f_value = new_g + abs(jx - end[0]) + abs(jy - end[1])
#                     heapq.heappush(open_set, (f_value, (jx, jy)))
#                     parent[(jx, jy)] = (x, y)
#                     plt.fill([jy - 0.5, jy + 0.5, jy + 0.5, jy - 0.5],
#                              [jx - 0.5, jx - 0.5, jx + 0.5, jx + 0.5], color='blue')
#                     # plt.pause(0.005)

#     duration = time.time() - start_time
#     print(f"Jump Point Search took {duration:.4f} seconds")
#     return reconstruct_path(parent, start, end), visited

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
            # Fill the cell with color
            plt.fill([y - 0.5, y + 0.5, y + 0.5, y - 0.5],
                     [x - 0.5, x - 0.5, x + 0.5, x + 0.5], color='blue', alpha=0.5)

    if path:
        for (x, y) in path:
            # Fill the cell with color for the path
            plt.fill([y - 0.5, y + 0.5, y + 0.5, y - 0.5],
                     [x - 0.5, x - 0.5, x + 0.5, x + 0.5], color='red', alpha=0.8)

    plt.axis('off')
    plt.title("Maze Solving Visualization")
    plt.show()


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Generate a maze")
    print("2. Upload an image of a maze")
    option = int(input("Enter your choice (1 or 2): "))

    if option == 1:
        # Generate a maze
        width = int(input("Enter maze width: "))
        height = int(input("Enter maze height: "))
        maze = generate_maze(width, height)
        visualize_maze(maze)
        # Define start and end points for generated maze
        start = (1, 0)
        end = (maze.shape[0] - 2, maze.shape[1] - 1)
    elif option == 2:
        # Upload an image of a maze
        image_path = input("Enter the path to the maze image: ")
        maze, start, end = image_to_maze(image_path)  # Get maze, start, and end points
        visualize_maze(maze)
    else:
        print("Invalid choice. Exiting...")
        exit()

    # Print start and end points for debugging
    print("Start Point:", start)
    print("End Point:", end)

    while True:
        print("Choose Algorithm:")
        print("1. BFS")
        print("2. DFS")
        print("3. Dijkstra")
        print("4. A*")
        print("5. Ant Colony Optimization")
        # print("6. Jump Point Search")
        print("7. Exit")
        choice = int(input("Enter Choice: "))

        if choice == 1:
            if option == 2:
                print("Choose Visualization Option:")
                print("1. Step-by-Step Visualization (Slower)")
                print("2. Final Path Visualization")
                visualization_choice = int(input("Enter Visualization Choice: "))
                if visualization_choice == 1:
                    path, visited = bfs(maze, start, end, visualize=True)
                elif visualization_choice == 2:
                    path, visited = bfs(maze, start, end, visualize=False)
            else:
                path, visited = bfs(maze, start, end)

        elif choice == 2:
            if option == 2:
                print("Choose Visualization Option:")
                print("1. Step-by-Step Visualization (Slower)")
                print("2. Final Path Visualization")
                visualization_choice = int(input("Enter Visualization Choice: "))
                if visualization_choice == 1:
                    path, visited = dfs(maze, start, end, visualize=True)
                elif visualization_choice == 2:
                    path, visited = dfs(maze, start, end, visualize=False)
            else:
                path, visited = dfs(maze, start, end)

        elif choice == 3:
            if option == 2:
                print("Choose Visualization Option:")
                print("1. Step-by-Step Visualization (Slower)")
                print("2. Final Path Visualization")
                visualization_choice = int(input("Enter Visualization Choice: "))
                if visualization_choice == 1:
                    path, visited = dfs(maze, start, end, visualize=True)
                elif visualization_choice == 2:
                    path, visited = dfs(maze, start, end, visualize=False)
            else:
                path, visited = dijkstra(maze, start, end)

        elif choice == 4:
            print("Choose Heuristic for A*:")
            print("1. Manhattan Distance")
            print("2. Euclidean Distance")
            print("3. Chebyshev Distance")
            print("4. Diagonal Distance")
            print("5. Octile Distance")
            heuristic_choice = int(input("Enter Heuristic Choice: "))

            if heuristic_choice == 1:
                heuristic_func = manhattan_distance
            elif heuristic_choice == 2:
                heuristic_func = euclidean_distance
            elif heuristic_choice == 3:
                heuristic_func = chebyshev_distance
            elif heuristic_choice == 4:
                heuristic_func = diagonal_distance
            elif heuristic_choice == 5:
                heuristic_func = octile_distance
            else:
                print("Invalid Heuristic Choice. Using Manhattan Distance by default.")
                heuristic_func = manhattan_distance

            if option == 2:
                print("Choose Visualization Option:")
                print("1. Step-by-Step Visualization (Slower)")
                print("2. Final Path Visualization")
                visualization_choice = int(input("Enter Visualization Choice: "))
                if visualization_choice == 1:
                    path, visited = dfs(maze, start, end, visualize=True)
                elif visualization_choice == 2:
                    path, visited = dfs(maze, start, end, visualize=False)
            else:
                path, visited = astar(maze, start, end, heuristic_func)
        elif choice == 5:
            path, visited = ant_colony_optimization(maze, start, end)
        # elif choice == 6:
            # path, visited = jump_point_search(maze, start, end)
        elif choice == 7:
            print("Exiting...")
            break
        else:
            print("Invalid Choice")
            continue

        visualize_maze(maze, path, visited)