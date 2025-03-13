import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import time
from collections import deque
import heapq
import random

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

    # Set entrance and exit
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
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Find contours in the processed binary image
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise) based on area
    min_contour_area = 100
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Combine all filtered contours into a single contour
    combined_contour = np.vstack(filtered_contours)

    # Find the bounding rectangle of the combined contour
    x, y, w, h = cv2.boundingRect(combined_contour)

    # Define perspective transform points
    corners = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ])

    # Perspective transform
    width, height = 400, 400
    target_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target_corners)
    straightened = cv2.warpPerspective(image, transform_matrix, (width, height))

    # Convert to final binary format
    gray_straightened = cv2.cvtColor(straightened, cv2.COLOR_BGR2GRAY)
    _, binary_maze = cv2.threshold(gray_straightened, 128, 255, cv2.THRESH_BINARY_INV)
    binary_maze = cv2.dilate(binary_maze, kernel, iterations=1)
    binary_maze = cv2.erode(binary_maze, kernel, iterations=1)

    # Convert to 0s and 1s (0 = path, 1 = wall)
    binary_maze = np.where(binary_maze > 128, 1, 0).astype(int)

    # Identify start and end points
    start_point, end_point = None, None
    for i in range(binary_maze.shape[0]):
        if binary_maze[i, 0] == 0:
            start_point = (i, 0)
            break
            
    for i in range(binary_maze.shape[0]):
        if binary_maze[i, -1] == 0:
            end_point = (i, binary_maze.shape[1] - 1)
            break

    if start_point is None or end_point is None:
        raise ValueError("Could not detect start/end points. Check maze entry/exit points.")

    return binary_maze, start_point, end_point

def custom_maze_editor():
    st.sidebar.header("Custom Maze Editor")
    
    # Initialize session state for maze, dimensions, and drawing mode
    if 'custom_maze' not in st.session_state:
        st.session_state.custom_maze = None
        st.session_state.start_point = None
        st.session_state.end_point = None
        st.session_state.drawing_mode = 'wall'
        st.session_state.rows = 5  # Default row count
        st.session_state.cols = 5  # Default column count

    # Maze dimensions input
    cols_in = st.sidebar.number_input("Columns", 3, 10, st.session_state.cols)
    rows_in = st.sidebar.number_input("Rows", 3, 10, st.session_state.rows)
    
    # Update maze size only when "Create New Maze" button is pressed
    if st.sidebar.button("Create New Maze"):
        st.session_state.rows = rows_in
        st.session_state.cols = cols_in
        st.session_state.custom_maze = np.zeros((st.session_state.rows, st.session_state.cols), dtype=int)
        st.session_state.start_point = None
        st.session_state.end_point = None
    
    if st.session_state.custom_maze is not None:
        # Drawing modes
        st.session_state.drawing_mode = st.sidebar.radio(
            "Drawing Mode",
            ['wall', 'erase', 'start', 'end'],
            horizontal=True
        )
        
        # Grid display
        st.header("Draw Your Maze")
        col1 = st.columns([1])  # Only 1 column to fill entire width
        
        with col1[0]:
            # Create grid using columns
            grid = st.columns(st.session_state.cols)
            for i in range(st.session_state.rows):
                for j in range(st.session_state.cols):
                    with grid[j]:
                        cell_key = f"cell_{i}_{j}"
                        cell_value = st.session_state.custom_maze[i, j]
                        
                        # Determine button color based on the cell's value
                        if (i, j) == st.session_state.start_point:
                            btn_color = 'green'
                        elif (i, j) == st.session_state.end_point:
                            btn_color = 'red'
                        else:
                            btn_color = 'white' if cell_value == 0 else 'black'
                        
                        # Create button with dynamic color representation
                        btn_label = " "
                        if btn_color == 'green':
                            btn_label = "🟢"  # Green for Start
                        elif btn_color == 'red':
                            btn_label = "🔴"  # Red for End
                        elif btn_color == 'black':
                            btn_label = "🟥"  # Wall representation
                        
                        # Create the button and handle click
                        if st.button(
                            btn_label, 
                            key=cell_key,
                            on_click=handle_cell_click,
                            args=(i, j),
                            help=f"Cell ({i}, {j})"
                        ):
                            pass

            # Reset the maze button
            if st.button("Reset Maze"):
                st.session_state.custom_maze = np.zeros((st.session_state.rows, st.session_state.cols), dtype=int)
                st.session_state.start_point = None
                st.session_state.end_point = None

        return st.session_state.custom_maze, st.session_state.start_point, st.session_state.end_point
    return None, None, None

def handle_cell_click(i, j):
    """Handle cell click event to modify the maze."""
    if st.session_state.drawing_mode == 'wall':
        st.session_state.custom_maze[i, j] = 1  # Set the cell as a wall
    elif st.session_state.drawing_mode == 'erase':
        st.session_state.custom_maze[i, j] = 0  # Erase the wall
    elif st.session_state.drawing_mode == 'start':
        st.session_state.start_point = (i, j)  # Set start point
    elif st.session_state.drawing_mode == 'end':
        st.session_state.end_point = (i, j)  # Set end point

def handle_grid_messages():
    if 'messages' in st.session_state:
        messages = st.session_state.messages
        st.session_state.grid_clicks = [
            msg['data'] for msg in messages 
            if msg['type'] == 'CELL_CLICK'
        ]
        del st.session_state.messages

# BFS Algorithm 
def bfs(maze, start, end, speed,visualize=True):
    start_time = time.time()
    queue = deque([start])
    visited = np.zeros_like(maze)
    parent = {}
    visited[start] = 1
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Create a placeholder for dynamic updates
    plot_placeholder = st.empty()

    while queue:
        x, y = queue.popleft()
        
        if visualize:
        
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(maze, cmap='binary')
            ax.axis('off')
           
            # Draw visited nodes
            for (vx, vy) in parent.keys():
                ax.add_patch(plt.Rectangle((vy - 0.5, vx - 0.5), 1, 1, color='blue', alpha=0.6))

            # Update the plot
            plot_placeholder.pyplot(fig)
            plt.close(fig)  # Close the figure to prevent memory leaks
            time.sleep(speed)

        if (x, y) == end:
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0 and not visited[nx, ny]:
                queue.append((nx, ny))
                visited[nx, ny] = 1
                parent[(nx, ny)] = (x, y)

    duration = time.time() - start_time
    st.write(f"BFS took {duration:.4f} seconds")
    return reconstruct_path(parent, start, end), visited

# DFS Algorithm 
def dfs(maze, start, end,speed, visualize=True):
    start_time = time.time()
    stack = [start]
    visited = np.zeros_like(maze)
    parent = {}
    visited[start] = 1
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Create visualization elements
    plot_placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(maze, cmap='binary')
    ax.set_title("DFS Step-by-Step Visualization")
    ax.axis('off')

    while stack:
        x, y = stack.pop()

        if (x, y) == end:
            break
        
        
            # Explore neighbors in random order
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0 and not visited[nx, ny]:
                stack.append((nx, ny))
                visited[nx, ny] = 1
                parent[(nx, ny)] = (x, y)

        if visualize:
            # Update visualization
            ax.clear()
            ax.imshow(maze, cmap='binary')
            ax.axis('off')
            
            # Draw visited nodes
            for (vx, vy) in parent.keys():
                ax.add_patch(plt.Rectangle((vy - 0.5, vx - 0.5), 1, 1, color='blue', alpha=0.6))
            
            plot_placeholder.pyplot(fig)
            time.sleep(speed)

    duration = time.time() - start_time
    st.write(f"DFS took {duration:.4f} seconds")
    return reconstruct_path(parent, start, end), visited

# Dijkstra's Algorithm 
def dijkstra(maze, start, end,speed, visualize=True):
    start_time = time.time()
    queue = [(0, start)]
    distances = {start: 0}
    parent = {}
    visited = np.zeros_like(maze)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Create visualization elements
    plot_placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(maze, cmap='binary')
    ax.set_title("Dijkstra's Step-by-Step Visualization")
    ax.axis('off')

    while queue:
        dist, (x, y) = heapq.heappop(queue)
        if visited[x, y]:
            continue
        visited[x, y] = 1

        if (x, y) == end:
            break
        
        if visualize:
            # Update visualization
            ax.clear()
            ax.imshow(maze, cmap='binary')
            ax.axis('off')
            
            # Draw visited nodes
            for (vx, vy) in parent.keys():
                ax.add_patch(plt.Rectangle((vy - 0.5, vx - 0.5), 1, 1, color='blue', alpha=0.6))
            
            plot_placeholder.pyplot(fig)
            time.sleep(speed)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                new_distance = dist + 1
                if (nx, ny) not in distances or new_distance < distances[(nx, ny)]:
                    distances[(nx, ny)] = new_distance
                    heapq.heappush(queue, (new_distance, (nx, ny)))
                    parent[(nx, ny)] = (x, y)

    duration = time.time() - start_time
    st.write(f"Dijkstra took {duration:.4f} seconds")
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

# A* Algorithm 
def astar(maze, start, end, heuristic_func,speed, visualize=True):
    start_time = time.time()
    queue = [(0, start)]
    g_values = {start: 0}
    parent = {}
    visited = np.zeros_like(maze)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Create visualization elements
    plot_placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(maze, cmap='binary')
    ax.set_title("A* Step-by-Step Visualization")
    ax.axis('off')

    while queue:
        _, (x, y) = heapq.heappop(queue)

        if visited[x, y]:
            continue
        visited[x, y] = 1

        if (x, y) == end:
            break
        
        if visualize:
            # Update visualization
            ax.clear()
            ax.imshow(maze, cmap='binary')
            ax.axis('off')
            
            # Draw visited nodes
            for (vx, vy) in parent.keys():
                ax.add_patch(plt.Rectangle((vy - 0.5, vx - 0.5), 1, 1, color='blue', alpha=0.6))
            
            plot_placeholder.pyplot(fig)
            time.sleep(speed)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                new_g = g_values.get((x, y), 0) + 1
                if new_g < g_values.get((nx, ny), float('inf')):
                    g_values[(nx, ny)] = new_g
                    f_value = new_g + heuristic_func((nx, ny), end)
                    heapq.heappush(queue, (f_value, (nx, ny)))
                    parent[(nx, ny)] = (x, y)

    duration = time.time() - start_time
    st.write(f"A* took {duration:.4f} seconds")
    return reconstruct_path(parent, start, end), visited

#antcolony 
def ant_colony_optimization(maze, start, end,speed, num_ants=50, max_iter=100, evaporation_rate=0.5, alpha=1, beta=2,visualize=True):
    start_time = time.time()
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    pheromones = np.ones_like(maze, dtype=float)
    
    # Create visualization elements
    plot_placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(maze, cmap='binary')
    ax.set_title("Ant Colony Optimization")
    ax.axis('off')
    progress_bar = st.progress(0)
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
                    break

                probabilities = []
                for neighbor in neighbors:
                    pheromone = pheromones[neighbor]
                    distance = abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                    probabilities.append((pheromone ** alpha) * ((1 / (distance + 1)) ** beta))

                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()

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

        pheromones *= evaporation_rate
        for path, length in zip(paths, path_lengths):
            for node in path:
                pheromones[node] += 1 / length
        current_progress = int((iteration+1)/max_iter*100)

        if visualize:
            # Update visualization
            ax.clear()
            ax.imshow(maze, cmap='binary')
            ax.imshow(pheromones, alpha=0.5, cmap='hot')
            ax.axis('off')
            plot_placeholder.pyplot(fig)
            time.sleep(speed)

        progress_bar.progress(current_progress)

    duration = time.time() - start_time
    st.write(f"Ant Colony Optimization took {duration:.4f} seconds")
    return best_path, pheromones   

# Modified visualize_maze to return figure
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
    plt.show()



# Streamlit UI
st.title("Maze Solver")
# Maze generation/upload
st.sidebar.header("Maze Configuration")
option = st.sidebar.selectbox("Choose Maze Source", ["Generate Maze", "Upload Image", "Create Custom Maze"],key="maze_source_select") 

#check box
checkbox_value = st.sidebar.checkbox("Enable visulization")

if option == "Create Custom Maze":
    handle_grid_messages()
    maze, start, end = custom_maze_editor()
    
    if maze is not None:
        st.header("Custom Maze Preview")
        try:
            # Ensure maze is a numpy array
            if not isinstance(maze, np.ndarray):
                maze = np.array(maze, dtype=np.int8)
            
            # Validate maze content
            if np.any((maze != 0) & (maze != 1)):
                st.error("Invalid maze values detected! Only 0s and 1s allowed.")
            else:
                fig, ax = plt.subplots(figsize=(10, 5))  # Create the figure and axes explicitly
                ax.imshow(maze, cmap='binary', vmin=0, vmax=1)  # Visualize the maze
                # Optionally, you can add additional visualizations or styling to the axes if needed
                ax.axis('off')  # Hide the axis
                # Now pass the figure to st.pyplot() to avoid the warning
                st.pyplot(fig)
                
                if st.sidebar.button("Validate Custom Maze"):
                    if start is None or end is None:
                        st.error("Please set both start and end points!")
        except Exception as e:
            st.error(f"Error processing maze: {str(e)}")
    else:
        st.warning("Please create a new maze first using the sidebar controls!")

elif option == "Generate Maze":
    cols = st.sidebar.slider("Maze Width", 5, 50, 20)
    rows = st.sidebar.slider("Maze Height", 5, 50, 20)
    maze = generate_maze(cols, rows)
    start = (1, 0)
    end = (maze.shape[0]-2, maze.shape[1]-1)

elif option == "Upload Image":
    uploaded = st.sidebar.file_uploader("Upload Maze Image", type=["png","jpg","jpeg"])
    if uploaded:
        with open("temp.png","wb") as f:
            f.write(uploaded.getbuffer())
        try:
            maze, start, end = image_to_maze("temp.png")
        except ValueError as e:
            st.error(str(e))
            st.stop()
    else:
        st.warning("Please add a maze to solve!")
        st.stop()


# Display original maze
if option != "Create Custom Maze":
    st.header("Original Maze")
    # Explicitly create the figure using plt.subplots
    fig, ax = plt.subplots(figsize=(10, 5))  # Create the figure and axes explicitly
    ax.imshow(maze, cmap='binary', vmin=0, vmax=1)  # Visualize the maze
    # Optionally, you can add additional visualizations or styling to the axes if needed
    ax.axis('off')  # Hide the axis
    # Now pass the figure to st.pyplot() to avoid the warning
    st.pyplot(fig)

speed =0
antCount = 20
iteration = 100

if checkbox_value == True:
    #speed select
    speed_slider = st.sidebar.select_slider(
        "Select Speed", 
        options=["Slow", "Normal", "Faster"],
        value="Normal"  # Set the default value to "Normal"
    )
    speed_map = {
        "Slow": 1,    # Slow speed, more delay
        "Normal": 0.01,  # Normal speed, moderate delay
        "Faster": 0.005  # Faster speed, less delay
    }
    speed = speed_map[speed_slider]

# Algorithm selection
algorithm = st.sidebar.selectbox("Algorithm", ["BFS", "DFS", "Dijkstra", "A*", "Ant Colony"],key="algorithm_select")  
st.session_state.algorithm = algorithm
# Show the heuristic menu only if A* is selected
if st.session_state.algorithm == "A*":
    heuristic = st.sidebar.selectbox(
        "Heuristic",
        ["Manhattan", "Euclidean", "Chebyshev", "Diagonal", "Octile"]
    )
    heuristic_func = {
        "Manhattan": manhattan_distance,
        "Euclidean": euclidean_distance,
        "Chebyshev": chebyshev_distance,
        "Diagonal": diagonal_distance,
        "Octile": octile_distance
    }[heuristic]

if st.session_state.algorithm =="Ant Colony":
    antCount = st.sidebar.slider("Ant Count", 5, 50, 20)
    iteration = st.sidebar.slider("Iteration Count", 50, 1000, 100)

if st.sidebar.button("Solve Maze"):
    try:
        if checkbox_value == True:
            st.header("Solution Process")    
        # Run the selected algorithm
        if st.session_state.algorithm == "BFS":
            path, visited = bfs(maze, start, end, speed, visualize=checkbox_value)
        elif st.session_state.algorithm == "DFS":
            path, visited = dfs(maze, start, end, speed, visualize=checkbox_value)
        elif st.session_state.algorithm == "Dijkstra":
            path, visited = dijkstra(maze, start, end, speed, visualize=checkbox_value)
        elif st.session_state.algorithm == "A*":
            path, visited = astar(maze, start, end, heuristic_func, speed, visualize=checkbox_value)
        elif st.session_state.algorithm == "Ant Colony":
            if start is None or end is None:
                st.error("Please set both start and end points!")
                st.stop()  
            path, pheromones = ant_colony_optimization(maze, start, end,speed,visualize=checkbox_value)
            visited = pheromones

    except TypeError:
        st.error("Please create or generate new maze")
        st.stop()

    st.header("Final Solution")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(maze, cmap='binary')
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_y, path_x, c='red', linewidth=2)
    ax.axis('off')
    st.pyplot(fig)
    
    if path and (end in path):
        st.success("Path found!")
    else:
        st.error("No path found")

