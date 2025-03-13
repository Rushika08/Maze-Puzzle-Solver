
```
███    ███  █████  ███████ ███████                
████  ████ ██   ██    ███  ██                     
██ ████ ██ ███████   ███   █████                  
██  ██  ██ ██   ██  ███    ██                     
██      ██ ██   ██ ███████ ███████                
                                                  
                                                  
███████  ██████  ██      ██    ██ ███████ ██████  
██      ██    ██ ██      ██    ██ ██      ██   ██ 
███████ ██    ██ ██      ██    ██ █████   ██████  
     ██ ██    ██ ██       ██  ██  ██      ██   ██ 
███████  ██████  ███████   ████   ███████ ██   ██ 
                                                                                                                                                                                                
```                                                                                                           
# Maze Solver Application
A visualization tool for maze generation and pathfinding algorithms, built with Streamlit.

## Features

- **Maze Generation**:  
  ✔️ Depth-First Search (DFS) algorithm  
  ✔️ Customizable dimensions (width/height)

- **Maze Conversion**:  
  ✔️ Image-to-maze conversion (JPG/PNG)  
  ✔️ Automatic start/end point detection

- **Custom Maze Editor**:  
  🎨 Interactive grid editor  
  🟩 Set start points  
  🟥 Set end points  
  ⬛ Draw walls/paths

- **Pathfinding Algorithms**:  
  🧭 BFS (Breadth-First Search)  
  🌀 DFS (Depth-First Search)  
  ⚖️ Dijkstra's Algorithm  
  ⭐ A* Search (with multiple heuristics)  
  🐜 Ant Colony Optimization

- **Visualization**:  
  🎥 Real-time algorithm visualization  
  ⏱️ Speed control (Slow/Normal/Fast)  
  🔴 Final path highlighting  
  🔵 Exploration process visualization

## Requirements

- Python 3.7+
- `streamlit>=1.12.0`
- `numpy>=1.19.5`
- `matplotlib>=3.5.0`
- `opencv-python-headless>=4.6.0.66`

## Installation

1. Clone repository:
```bash
git clone https://github.com/Rushika08/Maze-Puzzle-Solver.git
cd maze-solver
streamlit run Maze_solver.py
