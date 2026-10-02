"""
AI2002 - Assignment 02 - Section C
Question 11 - Iterative Deepening Search (IDS) built on Depth-Limited Search (DLS)
"""

tree = {
    'Root': ['N1', 'N2', 'N3'],
    'N1': ['N4', 'N5'], 'N2': ['N6', 'N7'], 'N3': ['N8', 'N9'],
    'N4': [], 'N5': [], 'N6': [],
    'N7': ['N12', 'N15'],
    'N8': [],
    'N9': ['N13', 'N14'],
    'N12': [], 'N13': [], 'N14': [],
    'N15': ['GOAL', 'N16'],
    'GOAL': [], 'N16': []
}


def dls(tree, node, goal, limit, depth=0, counter=None):
    """
    Part (a): recursive Depth-Limited Search.
    Returns the path (list of nodes) to `goal` if found within `limit`, else None.
    Every visited node is printed with its depth (indentation shows the depth too).
    `counter` is an optional one-element list used to accumulate node visits.
    """
    if counter is not None:
        counter[0] += 1
    print(f"{'    ' * depth}[depth {depth}] {node}")

    if node == goal:
        return [node]
    if depth == limit:              # depth bound reached -> cut off here
        return None

    for child in tree[node]:        # left-to-right, depth-first
        result = dls(tree, child, goal, limit, depth + 1, counter)
        if result is not None:
            return [node] + result
    return None


def ids(tree, root, goal, max_limit):
    """
    Part (b): Iterative Deepening Search.
    Calls dls for limit = 0, 1, ..., max_limit and returns
    (first_path_found, cumulative_node_visits_across_all_iterations).
    """
    counter = [0]
    for limit in range(max_limit + 1):
        before = counter[0]
        print(f"\n===== Iteration L = {limit} =====")
        path = dls(tree, root, goal, limit, 0, counter)
        print(f"----- L = {limit}: {counter[0] - before} node visit(s), "
              f"goal {'FOUND' if path else 'not found'}")
        if path is not None:
            return path, counter[0]
    return None, counter[0]


if __name__ == "__main__":
    # ---- Part (c): driver ----
    path, total_visits = ids(tree, 'Root', 'GOAL', 5)

    print("\n" + "=" * 50)
    print("Final path           :", " -> ".join(path) if path else None)
    print("Total node visits    :", total_visits)

    if path:
        goal_depth = len(path) - 1
        print("GOAL depth           :", goal_depth)
        print(f"Wasted iterations    : {goal_depth} (L = 0 .. {goal_depth - 1}) "
              f"- none of them could reach GOAL, because GOAL sits at depth {goal_depth}.")

    # Comment: IDS only "wastes" the shallow iterations L = 0 .. d-1. Their cost is
    # small compared with the last iteration because the number of nodes grows
    # geometrically with depth (most nodes live on the deepest level), so the
    # overhead is only a constant factor: O(b^d) overall, same as BFS, but with
    # O(b*d) memory instead of O(b^d).
