"""
AI2002 - Assignment 02 - Section C
Question 12 - Greedy Best-First Search
"""
import heapq
from q10_ucs import ucs          # reuse UCS from Question 10 for the comparison in part (b)

graph = {
    'S': {'A': 3, 'B': 7, 'C': 6},
    'A': {'S': 3, 'D': 4, 'E': 9},
    'B': {'S': 7, 'D': 2, 'F': 5},
    'C': {'S': 6, 'F': 3},
    'D': {'A': 4, 'B': 2, 'G': 7},
    'E': {'A': 9, 'G': 2},
    'F': {'B': 5, 'C': 3, 'G': 4},
    'G': {'D': 7, 'E': 2, 'F': 4}
}
heuristics = {'S': 9, 'A': 6, 'B': 5, 'C': 7, 'D': 3, 'E': 2, 'F': 4, 'G': 0}


def greedy_best_first(graph, heuristics, start, goal, verbose=True, return_stats=False):
    """
    Part (a): Greedy Best-First Search. The priority queue is keyed ONLY by h(n);
    the accumulated path cost g(n) is never used for ordering.
    Returns the path found (not necessarily optimal) and prints each expanded
    node together with its h(n).
    If return_stats=True it returns (path, nodes_expanded) - used by Question 14.
    """
    frontier = [(heuristics[start], start)]     # (h, node)
    parent = {start: None}                      # also acts as the "discovered" set
    explored = set()
    expanded = 0

    while frontier:
        h, node = heapq.heappop(frontier)
        if node in explored:
            continue
        explored.add(node)
        expanded += 1
        if verbose:
            shown = sorted(frontier)
            print(f"  expand {node}  h={h}   frontier after pop: "
                  f"{[(n, hh) for hh, n in shown]}")

        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return (path, expanded) if return_stats else path

        for neighbour in graph[node]:
            if neighbour not in explored and neighbour not in parent:
                parent[neighbour] = node
                heapq.heappush(frontier, (heuristics[neighbour], neighbour))

    return (None, expanded) if return_stats else None


def path_cost(graph, path):
    """Part (b): true cost of a path = sum of the real edge weights."""
    return sum(graph[a][b] for a, b in zip(path, path[1:]))


# ----------------------------------------------------------------------------
# Part (c): turning this into A*
#   1. Key the priority queue by f(n) = g(n) + h(n) instead of h(n) alone, which
#      means storing g(n) with each entry (g(child) = g(parent) + edge weight).
#   2. Allow a node to be RE-opened/updated when a cheaper path to it is found
#      (keep best_g[] and push again if new g < best_g), instead of marking a
#      node "discovered" forever the first time it is seen.
#   3. Keep h(n) admissible (never overestimates) so the first goal pop is optimal.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Greedy Best-First Search: S -> G")
    path = greedy_best_first(graph, heuristics, 'S', 'G')
    greedy_cost = path_cost(graph, path)

    print("\nGreedy path        :", " -> ".join(path))
    print("Greedy true cost   :", greedy_cost)

    # Part (b): compare with UCS (optimal) on the same graph
    ucs_path, ucs_cost, _ = ucs(graph, 'S', 'G', verbose=False)
    print("UCS (optimal) path :", " -> ".join(ucs_path))
    print("UCS (optimal) cost :", ucs_cost)

    diff = greedy_cost - ucs_cost
    print(f"\nGreedy is {diff} units worse than optimal "
          f"({diff / ucs_cost * 100:.1f}% higher cost)." if diff > 0
          else "\nGreedy happened to find an optimal path here.")
