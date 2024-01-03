import numpy as np
import time

import helpers


def q24():
    with open("2023-24.txt") as f:
        lines = f.read().splitlines()

    hailstones = []
    for line in lines:
        position_raw, velocity_raw = line.split(' @ ')
        position = np.array([int(n) for n in position_raw.split(', ')])
        velocity = np.array([int(n) for n in velocity_raw.split(', ')])
        hailstones.append({
            'p': position,
            'v': velocity
        })

    # Part 1

    boundary_min = 200000000000000
    boundary_max = 400000000000000

    def inverse(matrix):
        try:
            return np.linalg.inv(matrix)
        # Handle on our own since we expect this to happen
        except np.linalg.LinAlgError:
            return None

    total_crosses = 0
    for i1 in range(len(hailstones)):
        h1 = hailstones[i1]
        p1 = h1['p']
        v1 = h1['v']
        for i2 in range(i1 + 1, len(hailstones)):
            h2 = hailstones[i2]
            p2 = h2['p']
            v2 = h2['v']
            matrix = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
            inverse_of_matrix = inverse(matrix)

            # Both lines have the same slope and thus will never cross
            if inverse_of_matrix is None:
                continue

            equals = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            times = np.matmul(inverse_of_matrix, equals)

            if (times <= 0).any():
                # The crossing happened in the past for at least one line
                continue

            t1 = times[0]
            x, y, _ = p1 + v1 * t1

            should_continue = False
            for variable in (x, y):
                if variable < boundary_min or variable > boundary_max:
                    # The crossing happened outside the boundary
                    should_continue = True
            if should_continue:
                continue

            total_crosses += 1

    # Part 2

    return total_crosses, None

def q25():
    with open("2023-25.txt") as f:
        lines = f.read().splitlines()

    # Create a graph of the input
    graph = {}
    for line in lines:
        node, neighbors = line.split(': ')
        neighbors = neighbors.split(' ')
        for neighbor in neighbors:
            if node in graph:
                graph[node].add(neighbor)
            else:
                graph[node] = set([neighbor])
            if neighbor in graph:
                graph[neighbor].add(node)
            else:
                graph[neighbor] = set([node])

    # Part 1

    # Using breadth-first search (BFS), calculate minimal spanning trees (MSTs) starting from every node
    # BFS works for MST calculation because all edges have the same weight of 1
    msts = []
    for node in graph.keys():
        _, mst = helpers.bfs(node, graph)
        msts.append(mst)

    # Get a count of how times an edge appears among the spanning trees
    edge_appearances = {}
    def edge_hash(edge):
        return edge[0] + edge[1]
    for mst in msts:
        for edge in mst:
            hash = edge_hash(edge)
            if hash in edge_appearances:
                edge_appearances[hash] += 1
            else:
                edge_appearances[hash] = 1

    # Thought process:
    #   - The problem statement implies that the three edges are unique.
    #   - Relatedly, it implies that for any other three edges, even if we remove them
    #     the graph is still fully connected.
    #   - Based on that, a hypothesis is that the edges to remove are the three edges
    #     that appear in the most MSTs.
    edges_sorted = sorted(edge_appearances, key=edge_appearances.get, reverse=True)
    hashes_of_edges_to_break = tuple(edges_sorted[:3])
    edges_to_break = [[hash[:3], hash[3:]] for hash in hashes_of_edges_to_break]

    # Edit graph in place
    for v1, v2 in edges_to_break:
        graph[v1].remove(v2)
        graph[v2].remove(v1)

    # Pick an arbitary node and get the size of the first group
    nodes = set(graph.keys())
    visited1, _ = helpers.bfs(next(n for n in nodes), graph)
    answer = len(visited1)

    # Pick a node not in the first group to calculate the size of the second group
    visited2, _ = helpers.bfs(next(n for n in nodes if n not in visited1), graph)
    answer *= len(visited2)

    # Sanity checks to ensure what we have left is two full-connected groups disjoint from each other
    assert len(visited1.intersection(visited2)) == 0
    assert len(visited1) + len(visited2) == len(graph.keys())

    # Part 2

    return answer, None

start = time.time()
print(f"Answer: {q24()}")
end = time.time()
print(f"Time: {end - start:.2f}s")