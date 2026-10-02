"""
AI2002 - Assignment 02 - Section C
Question 14 - Search-Strategy Comparison Harness (UCS vs Greedy vs A*)
Reuses ucs() from Question 10 and greedy_best_first() from Question 12.
"""
import heapq
from q10_ucs import ucs
from q12_greedy import greedy_best_first, path_cost

graph = {
    'S': {'A': 2, 'B': 6},
    'A': {'S': 2, 'C': 3, 'D': 8},
    'B': {'S': 6, 'D': 2, 'E': 7},
    'C': {'A': 3, 'F': 5},
    'D': {'A': 8, 'B': 2, 'F': 1, 'G': 9},
    'E': {'B': 7, 'G': 3},
    'F': {'C': 5, 'D': 1, 'G': 4},
    'G': {'D': 9, 'E': 3, 'F': 4}
}
heuristics = {'S': 8, 'A': 6, 'B': 5, 'C': 5, 'D': 3, 'E': 2, 'F': 2, 'G': 0}


def astar_graph(graph, heuristics, start, goal):
    """A* on a weighted graph: frontier ordered by f = g + h. Returns (path, cost, expanded)."""
    frontier = [(heuristics[start], 0, start, [start])]    # (f, g, node, path)
    best_g = {start: 0}
    explored = set()
    expanded = 0
    while frontier:
        f, g, node, path = heapq.heappop(frontier)
        expanded += 1                           # every pop counts (stale ones too)
        if node in explored:
            continue
        if node == goal:
            return path, g, expanded
        explored.add(node)
        for nb, w in graph[node].items():
            new_g = g + w
            if new_g < best_g.get(nb, float('inf')):    # cheaper path found -> update/re-open
                best_g[nb] = new_g
                heapq.heappush(frontier, (new_g + heuristics[nb], new_g, nb, path + [nb]))
    return None, float('inf'), expanded


def compare_searches(graph, heuristics, start, goal):
    """Part (a)+(b): run UCS, Greedy and A* on the same graph and print one aligned table."""
    results = []

    u_path, u_cost, u_exp = ucs(graph, start, goal, verbose=False)
    results.append(("UCS", u_path, u_cost, u_exp))

    g_path, g_exp = greedy_best_first(graph, heuristics, start, goal,
                                      verbose=False, return_stats=True)
    results.append(("Greedy Best-First", g_path, path_cost(graph, g_path), g_exp))

    a_path, a_cost, a_exp = astar_graph(graph, heuristics, start, goal)
    results.append(("A*", a_path, a_cost, a_exp))

    # ---- aligned console table ----
    rows = [(name, " -> ".join(p), str(c), str(e)) for name, p, c, e in results]
    header = ("Algorithm", "Path returned", "True cost", "Nodes expanded")
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(4)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt(r):
        return "| " + " | ".join(r[i].ljust(widths[i]) for i in range(4)) + " |"

    print(line); print(fmt(header)); print(line)
    for r in rows:
        print(fmt(r))
    print(line)
    return results


# ----------------------------------------------------------------------------
# Part (c): unknown edge costs, known heuristic
#   UCS and A* are NOT implementable as specified: both order the frontier by the
#   accumulated cost g(n), which needs every edge weight BEFORE the robot travels
#   it. UCS is the clearest case (it has no other information to fall back on),
#   and A* breaks too because f = g + h cannot be computed for undiscovered edges.
#   Greedy Best-First is the only one that still works: it ranks nodes by h(n)
#   alone and never reads an edge cost - though it gives no optimality guarantee.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Comparison on the Question 14 graph: S -> G\n")
    compare_searches(graph, heuristics, 'S', 'G')
