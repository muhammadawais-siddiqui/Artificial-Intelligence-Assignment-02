"""
AI2002 - Assignment 02 - Section C
Question 13 - A* for Variable-Terrain Maze Pathfinding
Terrain: 0 = normal ground (step cost 1), 2 = mud (step cost 3), 1 = wall (impassable)
"""
import heapq

maze = [
    [0, 0, 1, 0, 0, 2, 0],
    [0, 1, 1, 0, 1, 2, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0],
    [1, 1, 0, 1, 2, 0, 0],
    [0, 0, 0, 0, 2, 0, 1],
    [0, 2, 2, 0, 0, 0, 0],
]
start = (0, 0)   # S
goal = (6, 6)    # G

STEP_COST = {0: 1, 2: 3}   # cost of ENTERING a cell of this terrain type


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Part (b) - why Manhattan distance is still admissible here:
#   Manhattan distance equals the cost of the cheapest conceivable route if every
#   step cost 1 and there were no walls. In this maze the cheapest possible step
#   costs exactly 1 (normal ground) and mud only makes a step MORE expensive (3),
#   while walls only force longer detours. Every real route to the goal needs at
#   least Manhattan-many moves and each move costs >= 1, so the true remaining
#   cost is always >= the Manhattan distance. h(n) <= h*(n)  =>  admissible
#   (and also consistent, since moving one cell changes h by at most 1 <= step cost).


def astar(maze, start, goal):
    """
    Part (a)/(c): A* with Manhattan heuristic, 4-directional moves.
    Returns (path_as_list_of_(row, col), total_cost) or None if no path exists.
    """
    rows, cols = len(maze), len(maze[0])

    # Part (c): immediate exit if start or goal is a wall (or outside the grid)
    for r, c in (start, goal):
        if not (0 <= r < rows and 0 <= c < cols) or maze[r][c] == 1:
            return None

    frontier = [(manhattan(start, goal), 0, start)]   # (f, g, cell)
    parent = {start: None}
    best_g = {start: 0}
    closed = set()

    while frontier:
        f, g, cell = heapq.heappop(frontier)
        if cell in closed:
            continue                      # stale entry
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            return path[::-1], g
        closed.add(cell)

        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):   # Up, Down, Left, Right
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            terrain = maze[nr][nc]
            if terrain == 1:                                  # wall
                continue
            new_g = g + STEP_COST[terrain]
            nxt = (nr, nc)
            if new_g < best_g.get(nxt, float('inf')):
                best_g[nxt] = new_g
                parent[nxt] = cell
                heapq.heappush(frontier, (new_g + manhattan(nxt, goal), new_g, nxt))
    return None


def draw(maze, path, start, goal):
    """Print the maze with the path marked: S, G, '*' = path, '#' = wall, '~' = mud, '.' = ground."""
    on_path = set(path or [])
    for r, row in enumerate(maze):
        line = []
        for c, val in enumerate(row):
            if (r, c) == start:
                ch = 'S'
            elif (r, c) == goal:
                ch = 'G'
            elif (r, c) in on_path:
                ch = '*'
            elif val == 1:
                ch = '#'
            elif val == 2:
                ch = '~'
            else:
                ch = '.'
            line.append(ch)
        print("   " + " ".join(line))


if __name__ == "__main__":
    result = astar(maze, start, goal)
    if result is None:
        print("No path exists (or start/goal is a wall).")
    else:
        path, cost = result
        print("Path (row, col):")
        print("  ", path)
        print("Total cost      :", cost)
        print("Number of moves :", len(path) - 1)
        print("\nMaze  (S start, G goal, * path, # wall, ~ mud, . ground):")
        draw(maze, path, start, goal)

    # Edge-case demo: start on a wall -> returns None immediately
    print("\nEdge case, start on a wall (4,0):", astar(maze, (4, 0), goal))
