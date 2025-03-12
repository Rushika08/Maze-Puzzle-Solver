import numpy as np
import matplotlib.pyplot as plt
import random

def generate_maze(width, height):
    """Generate a maze using recursive backtracking algorithm"""
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

class ACO_MazeSolver:
    def __init__(self, maze, n_ants=50, evaporation=0.25, alpha=1, beta=3):
        self.maze = maze
        self.height, self.width = maze.shape
        self.start = (1, 0)  # Entrance position
        self.end = (self.height-2, self.width-1)  # Exit position
        self.n_ants = n_ants
        self.evaporation = evaporation
        self.alpha = alpha
        self.beta = beta
        
        # Initialize pheromone matrix
        self.pheromones = np.zeros_like(maze, dtype=np.float32)
        self.pheromones[maze == 0] = 0.1  # Initial pheromone on paths
        
        # Create heuristic matrix (inverse distance to exit)
        self.heuristic = np.zeros_like(maze, dtype=np.float32)
        for y in range(self.height):
            for x in range(self.width):
                if maze[y, x] == 0:
                    dist = abs(y - self.end[0]) + abs(x - self.end[1])
                    self.heuristic[y, x] = 1 / (1 + dist)

    def run(self, iterations=100):
        plt.figure(figsize=(10, 8))
        best_path = None
        best_length = float('inf')
        
        for iteration in range(iterations):
            all_paths = []
            path_lengths = []
            
            # Simulate ants
            for _ in range(self.n_ants):
                path = self._move_ant()
                if path and path[-1] == self.end:
                    all_paths.append(path)
                    path_lengths.append(len(path))
            
            # Update pheromones
            self._update_pheromones(all_paths, path_lengths)
            
            # Find best path
            if path_lengths:
                current_best = np.argmin(path_lengths)
                if path_lengths[current_best] < best_length:
                    best_length = path_lengths[current_best]
                    best_path = all_paths[current_best]
            
            # Visualize every 5 iterations
            if iteration % 5 == 0 or iteration == iterations-1:
                self._visualize(iteration, best_path, all_paths)
        
        plt.show()

    def _move_ant(self):
        path = [self.start]
        current = self.start
        visited = set([current])
        
        while current != self.end:
            y, x = current
            neighbors = []
            probabilities = []
            
            # Check possible moves (up, down, left, right)
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.maze[ny, nx] == 0 and (ny, nx) not in visited:
                        pheromone = self.pheromones[ny, nx] ** self.alpha
                        heuristic = self.heuristic[ny, nx] ** self.beta
                        neighbors.append((ny, nx))
                        probabilities.append(pheromone * heuristic)
            
            if not neighbors:
                break  # Stuck ant
            
            # Normalize probabilities with safety checks
            prob_sum = np.sum(probabilities)
            
            # Handle zero-sum case (when all probabilities are 0)
            if prob_sum <= 0:
                probabilities = [1/len(neighbors)] * len(neighbors)
            else:
                # Ensure probabilities sum to exactly 1
                probabilities = np.array(probabilities)
                probabilities = probabilities / prob_sum
            
            # Add small epsilon to prevent numerical issues
            probabilities = np.clip(probabilities, 1e-8, 1.0)
            probabilities = probabilities / np.sum(probabilities)
            
            # Select next move
            try:
                next_idx = np.random.choice(len(neighbors), p=probabilities)
            except ValueError:
                # Fallback if there's still numerical issues
                next_idx = np.random.randint(len(neighbors))
            
            current = neighbors[next_idx]
            path.append(current)
            visited.add(current)
        
        return path if current == self.end else None

    def _update_pheromones(self, paths, path_lengths):
        # Evaporate pheromones
        self.pheromones[self.maze == 0] *= (1 - self.evaporation)
        
        # Deposit new pheromones
        for path, length in zip(paths, path_lengths):
            if length == 0:
                continue
            deposit = 1 / length
            for (y, x) in path:
                self.pheromones[y, x] += deposit

    def _visualize(self, iteration, best_path, all_paths):
        plt.clf()
        
        # Create colormap for maze
        cmap = plt.cm.colors.ListedColormap(['black', 'white'])
        plt.imshow(self.maze, cmap=cmap, interpolation='nearest')
        
        # Overlay pheromones
        pheromone_normalized = self.pheromones / self.pheromones.max() if self.pheromones.max() > 0 else self.pheromones
        plt.imshow(pheromone_normalized, cmap='Blues', alpha=0.4)
        
        # Draw all ant paths
        for path in all_paths:
            if path:
                xs = [x+0.5 for (y, x) in path]
                ys = [y+0.5 for (y, x) in path]
                plt.plot(xs, ys, 'grey', linewidth=0.5, alpha=0.2)
        
        # Draw best path
        if best_path:
            xs = [x+0.5 for (y, x) in best_path]
            ys = [y+0.5 for (y, x) in best_path]
            plt.plot(xs, ys, 'r-', linewidth=2)
        
        # Mark start and end
        plt.plot(self.start[1]+0.5, self.start[0]+0.5, 'gs', markersize=10)
        plt.plot(self.end[1]+0.5, self.end[0]+0.5, 'rs', markersize=10)
        
        plt.title(f"Iteration {iteration+1}\nShortest Path: {len(best_path) if best_path else 'N/A'}")
        plt.axis('off')
        plt.draw()
        plt.pause(0.1)

# Generate and solve maze
maze = generate_maze(width=15, height=15)
solver = ACO_MazeSolver(maze, n_ants=100, evaporation=0.3, alpha=1, beta=3)
solver.run(iterations=100)