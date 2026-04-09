import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import heapq
from collections import deque
from itertools import count

# -------------------------------
# Maze Definition
# -------------------------------
#Example 1.
# maze = [
#     ["S", 0, 1, 0, 0, 0, 1, 0, 0, 0],
#     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
#     [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 1, 1, 1, 1, 0, "G", 0]
# ]

#Example 2.
maze = [
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
[1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
[1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1],
[1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1],
[1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,"G",0,0,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1],
[1,0,1,0,1,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
[1,"S",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
]

#Example 3.
# maze = [
# [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
# [1,"S",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
# [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
# [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
# [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1],
# [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1],
# [1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,"G",0,0,1,0,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1],
# [1,0,1,0,1,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
# [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
# [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# ]

# #Example 4.
# maze = [[1,1,0,0,0,0,1],
#         [1,1,0,1,1,0,1],
#         [1,"G",0,1,0,0,1],
#         [1,0,1,1,0,1,1],
#         [0,0,0,0,0,1,1],
#         ["S",1,1,1,1,1,1]]

#Example 5.
# maze = [
# [0,1,0,1,0,1,1,1,0,0,1,"G"],
# [0,1,0,1,0,0,0,1,0,1,1,0],
# [0,1,0,1,0,1,0,0,0,1,1,0],
# [0,0,0,1,0,1,0,1,0,1,1,0],
# [1,0,0,0,0,1,0,1,0,0,0,0],
# [1,1,1,0,1,1,0,1,1,1,1,1],
# ["S",0,0,0,1,1,0,0,0,0,0,0]
# ]

ROWS = len(maze)
COLS = len(maze[0])

grid = np.zeros((ROWS, COLS))
# print(grid)
for i in range(ROWS):
    for j in range(COLS):
        if maze[i][j] == 1:
            grid[i][j] = 1
        elif maze[i][j] == "S":
            start = (i, j)
        elif maze[i][j] == "G":
            goal = (i, j)

# -------------------------------
# Node Class
# -------------------------------
class Node:
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost  # depth (step cost)

# -------------------------------
# Problem Definition
# -------------------------------
def actions(state):
    x, y = state
    moves = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1)
    }

    possible = []
    for action, (dx, dy) in moves.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < ROWS and 0 <= ny < COLS:
            if maze[nx][ny] != 1:
                possible.append(action)
    return possible


def result(state, action):
    moves = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1)
    }
    dx, dy = moves[action]
    return (state[0] + dx, state[1] + dy)


def goal_test(state):
    return state == goal


def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    path.reverse()
    return path

# -------------------------------
# DFS
# -------------------------------
def dfs_steps():
    frontier = [Node(start, cost=0)]
    explored = set()
    # print(frontier)
    while True:
        
        if not frontier:
            return

        node = frontier.pop()

        yield node.state, len(explored), len(frontier), node.cost

        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return

        explored.add(node.state)

        for action in actions(node.state):
            child_state = result(node.state, action)

            if child_state not in explored and \
               all(child_state != n.state for n in frontier):

                frontier.append(
                    Node(child_state, node, action, node.cost + 1)
                )

# -------------------------------
# BFS
# -------------------------------
def bfs_steps():
    frontier = deque([Node(start, cost=0)])
    explored = set()

    while True:

        if not frontier:
            return

        node = frontier.popleft()

        yield node.state, len(explored), len(frontier), node.cost

        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return

        explored.add(node.state)

        for action in actions(node.state):
            child_state = result(node.state, action)

            if child_state not in explored and \
               all(child_state != n.state for n in frontier):

                frontier.append(
                    Node(child_state, node, action, node.cost + 1)
                )
                

#Heuristic Function
#=============================
def heuristic(state):
    x1, y1 = state
    x2, y2 = goal
    return abs(x1 - x2) + abs(y1 - y2)



#=====================================
# #A* Search(Uses f(n) = g(n) + h(n))
#==================================
def astar_steps():
    frontier = []
    counter = count()

    start_node = Node(start, cost=0)

    # MUST initialize heap
    heapq.heappush(frontier, (0, next(counter), start_node))

    explored = set()

    while True:
        if not frontier:
            return

        _, _, node = heapq.heappop(frontier)

        yield node.state, len(explored), len(frontier), node.cost

        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return

        explored.add(node.state)

        for action in actions(node.state):
            child_state = result(node.state, action)

            if child_state in explored:
                continue

            g = node.cost + 1
            h = heuristic(child_state)
            f = g + h

            child_node = Node(child_state, node, action, g)

            #  MUST push into heap
            heapq.heappush(frontier, (f, next(counter), child_node))

            
#==========================================   
#Greedy Best-First Search(Uses f(n) = h(n))
#========================================            
def greedy_steps():
    frontier = []
    counter = count()

    start_node = Node(start, cost=0)

    # MUST initialize heap
    heapq.heappush(frontier, (heuristic(start), next(counter), start_node))

    explored = set()

    while True:
        if not frontier:
            return   # means no solution found OR bug

        _, _, node = heapq.heappop(frontier)

        yield node.state, len(explored), len(frontier), node.cost

        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return

        explored.add(node.state)

        for action in actions(node.state):
            child_state = result(node.state, action)

            if child_state in explored:
                continue

            child_node = Node(child_state, node, action, node.cost + 1)

            heapq.heappush(
                frontier,
                (heuristic(child_state), next(counter), child_node)
            )
            
# -------------------------------
# Animation
# -------------------------------
def animate_solver(algorithm="BFS"):
    fig, ax = plt.subplots()
    # ax.set_title(f"AI {algorithm} Maze Solver")
    ax.set_facecolor("lightgray")

    maze_img = np.copy(grid)
    img = ax.imshow(maze_img, cmap="gray_r")
    # print(img)
    # ax.grid(True)


    ax.scatter(start[1], start[0], c="green", s=100)
    ax.scatter(goal[1], goal[0], c="red", s=100)

    # Text overlays
    frontier_text = ax.text(0, -1, "", fontsize=10)
    explored_text = ax.text(3, -1, "", fontsize=10)
    # cost_text = ax.text(6, -1, "", fontsize=10)

    if algorithm == "BFS":
        steps = bfs_steps()
        
    elif algorithm == "DFS":
        steps = dfs_steps()
        
    elif algorithm == "A*":
        steps = list(astar_steps())
        if not steps:
            print("No path found OR A* returned empty!")
            return

    elif algorithm == "GREEDY":
        steps = greedy_steps()
        
    else:
        raise ValueError("Unknown algorithm")
    
    
    
    def update(frame):
        nonlocal maze_img

        if isinstance(frame[1], str):
            for x, y in frame[0]:
                maze_img[x][y] = 0.9
        else:
            state, explored_size, frontier_size, cost = frame
            x, y = state
            maze_img[x][y] = 0.5

            frontier_text.set_text(f"Frontier: {frontier_size} ")
            explored_text.set_text(f"Explored: {explored_size}")
            # cost_text.set_text(f"Cost: {cost}")

        img.set_data(maze_img)
        return [img]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=50,
        repeat=False,
        cache_frame_data=False

    )

    plt.show()

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":

    print("\n=== MAZE SEARCH ALGORITHMS ===")
    print("1. BFS")
    print("2. DFS")
    print("3. A*")
    print("4. Greedy Best-First")
    print("Start:", start)
    print("Goal:", goal)

    choice = input("Select algorithm (1-4): ")

    if choice == "1":
        animate_solver("BFS")

    elif choice == "2":
        animate_solver("DFS")

    elif choice == "3":
        animate_solver("A*")

    elif choice == "4":
        animate_solver("GREEDY")

    else:
        print("Invalid choice. Running BFS by default.")
        animate_solver("BFS")