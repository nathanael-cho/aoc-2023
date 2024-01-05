import math
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
            
    # TODO: Cleanup?

    # No Numpy! Numpy cuts off numbers...

    def cross_product(v1, v2):
        assert len(v1) == 3
        assert len(v2) == 3
        v_i = v1[1] * v2[2] - v1[2] * v2[1]
        v_j = v1[2] * v2[0] - v1[0] * v2[2]
        v_k = v1[0] * v2[1] - v1[1] * v2[0]
        return [v_i, v_j, v_k]
    
    def scalar_multiply(s, v):
        assert len(v) == 3
        return [s * v_i for v_i in v]
    
    def vector_subtract(v1, v2):
        assert len(v1) == 3
        assert len(v2) == 3
        return [
            v1[0] - v2[0],
            v1[1] - v2[1],
            v1[2] - v2[2]
        ]
    
    def simplify_vector(v):
        assert len(v) == 3
        gcd = math.gcd(math.gcd(v[0], v[1]), v[2])
        if not gcd:
            return v
        simplified = [vi // gcd for vi in v]
        if simplified[0] < 0:
            return scalar_multiply(-1, simplified)
        else:
            return simplified
    
    hailstones = []
    for line in lines:
        position_raw, velocity_raw = line.split(' @ ')
        position = [int(n) for n in position_raw.split(', ')]
        velocity = [int(n) for n in velocity_raw.split(', ')]
        hailstones.append({
            'p': position,
            'v': velocity,
            'p x v': cross_product(position, velocity)
        })

    # We don't actually have to go through all the points to calculate the starting position vector
    n = 16

    storage = []
    for i1 in range(n):
        h1 = hailstones[i1]
        p1 = h1['p']
        cp1 = h1['p x v']
        for i2 in range(i1 + 1, n):
            h2 = hailstones[i2]
            p2 = h2['p']
            cp2 = h2['p x v']
            first = vector_subtract(
                scalar_multiply(p2[0], p1),
                scalar_multiply(p1[0], p2)
            )
            second = vector_subtract(
                scalar_multiply(p2[0], cp1),
                scalar_multiply(p1[0], cp2)
            )
            storage.append((first, second))

    for i in range(1, 3):
        intermediate = []
        for i1 in range(n):
            x1 = storage[i1][0]
            y1 = storage[i1][1]
            for i2 in range(i1 + 1, n):
                x2 = storage[i2][0]
                y2 = storage[i2][1]
                first = vector_subtract(
                    scalar_multiply(x2[i], x1),
                    scalar_multiply(x1[i], x2)
                )
                second = vector_subtract(
                    scalar_multiply(x2[i], y1),
                    scalar_multiply(x1[i], y2)
                )
                intermediate.append((first, second))
        storage = intermediate

    final = []
    for x, y in storage:
        assert x == [0, 0, 0]
        if y != [0, 0, 0]:
            gcd = math.gcd(math.gcd(y[0], y[1]), y[2])
            final.append([
                y_i // gcd for y_i in y
            ])
    final = [(scalar_multiply(-1, v) if v[0] < 0 else v) for v in final]
    final = sorted(final)
    intermediate = [final[0]]
    for i in range(1, len(final)):
        if final[i] != final[i - 1]:
            intermediate.append(final[i])
    final = intermediate

    choices = []
    for i in range(len(final)):
        for j in range(i + 1, len(final)):
            choices.append(simplify_vector(cross_product(final[i], final[j])))
    choices = sorted(choices)
    intermediate = [choices[0]]
    for i in range(1, len(choices)):
        if choices[i] != choices[i - 1]:
            intermediate.append(choices[i])
    choices = intermediate
    assert len(choices) == 1

    # We calculate the position vector as: [239756157786030, 463222539161932, 273997500449219]
    # With similar code we calculate the slope vector as: [47, -360, 18]
    # The code below verifies that these vectors are indeed a solution
    p_alpha = [239756157786030, 463222539161932, 273997500449219]
    v_alpha = [47, -360, 18]
    for h in hailstones:
        p_difference = vector_subtract(
            p_alpha,
            h['p']
        )
        v_difference = vector_subtract(
            h['v'],
            v_alpha
        )
        t = p_difference[0] // v_difference[0]
        assert t > 0 and (scalar_multiply(t, v_difference) == p_difference)

    return total_crosses, sum(choices[0])

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