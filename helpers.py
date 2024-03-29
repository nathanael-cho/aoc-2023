from collections import deque
from time import perf_counter


class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self
    
    def __exit__(self, *_):
        self.time = perf_counter() - self.start
        print(f'Time taken: {self.time:.3f}s')


def bfs(node, graph):
    """BFS of an undirected, unweighted graph that also returns the edges traversed"""
    visited = set([node])
    edges = []
    nodes_to_visit = deque([node])
    while len(nodes_to_visit):
        current_node = nodes_to_visit.popleft()
        neighbors = graph[current_node]
        for neighbor in neighbors:
            if neighbor not in visited:
                if current_node < neighbor:
                    edges.append([current_node, neighbor])
                else:
                    edges.append([neighbor, current_node])
                visited.add(neighbor)
                nodes_to_visit.append(neighbor)
    return visited, edges