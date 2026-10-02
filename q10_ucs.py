"""
AI2002 - Assignment 02 - Section C
Question 10 - Uniform Cost Search (UCS)
"""
import heapq

graph = {
    'S': {'A': 4, 'B': 2, 'C': 9},
    'A': {'S': 4, 'D': 5, 'E': 11},
    'B': {'S': 2, 'D': 8, 'F': 7},
    'C': {'S': 9, 'F': 2},
    'D': {'A': 5, 'B': 8, 'G': 6},
    'E': {'A': 11, 'G': 1},
    'F': {'B': 7, 'C': 2, 'G': 5},
    'G': {'D': 6, 'E': 1, 'F': 5}
}


def ucs(graph, start, goal, verbose=True):
    """
    Uniform Cost Search using a heapq priority queue keyed by cumulative cost g(n).

    Part (a): returns (path, cost), or (None, inf) if the goal is unreachable,
              and prints the order in which nodes are popped (expanded).
    Part (b): also returns the number of nodes popped, INCLUDING stale queue
              entries (a node that was already expanded via a cheaper path).
              Final return value is therefore (path, cost, nodes_expanded).
    """
    # Queue entries: (cumulative_cost, node, path_so_far). Ties are broken
    # alphabetically by node name, which makes the trace deterministic.
    frontier = [(0, start, [start])]
    best_g = {start: 0}          # cheapest known cost to each discovered node
    explored = set()             # nodes already expanded
    expanded = 0                 # every pop counts, stale ones included

    if verbose:
        print(f"UCS: {start} -> {goal}")

    while frontier:
        cost, node, path = heapq.heappop(frontier)
        expanded += 1
        stale = node in explored

        if verbose:
            tag = "   <-- stale entry, skipped" if stale else ""
            print(f"  pop #{expanded:<2} node={node}  g={cost:<3} queue_size={len(frontier)}{tag}")

        if stale:
            continue

        # Goal test on POP (not on push) -> guarantees optimality
        if node == goal:
            return path, cost, expanded

        explored.add(node)
        for neighbour, weight in graph[node].items():
            new_cost = cost + weight
            if neighbour not in explored and new_cost < best_g.get(neighbour, float('inf')):
                best_g[neighbour] = new_cost
                heapq.heappush(frontier, (new_cost, neighbour, path + [neighbour]))

    return None, float('inf'), expanded


if __name__ == "__main__":
    # ---- Part (c): driver ----
    path, cost, expanded = ucs(graph, 'S', 'G')
    print()
    print("Path found      :", " -> ".join(path) if path else None)
    print("Total cost      :", cost)
    print("Nodes expanded  :", expanded)
