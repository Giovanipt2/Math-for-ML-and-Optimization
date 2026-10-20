"""
Modified Kruskal (Greedy Edge Insertion) Heuristic for the Traveling Salesman Problem (TSP)

Course: Mathematics for Machine Learning & Optimization - Metaheuristics
Authors: Student Team
"""

import sys
import os
import math
import time


class DisjointSetUnion:
    """Disjoint Set Union (DSU / Union-Find) with Path Compression and Rank Heuristic."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            if self.rank[root_i] == self.rank[root_j]:
                self.rank[root_i] += 1
            return True
        return False


def read_instance(file_path):
    """Reads a TSP instance file containing coordinates."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    coords = []
    for line in lines[1:n+1]:
        parts = line.split()
        coords.append((float(parts[0]), float(parts[1])))
    return n, coords


def euclidean_distance(p1, p2):
    """Calculates the 2D Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def save_solution(tour, total_distance, cpu_time, output_path):
    """Saves tour, total distance, and CPU time to a text file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(" ".join(map(str, tour)) + "\n")
        f.write(f"{total_distance:.3f}\n")
        f.write(f"{cpu_time:.5f}\n")


def solve_kruskal_tsp(n, coords, output_path=None):
    """
    Executes the Modified Kruskal / Greedy Edge Insertion Heuristic.
    Returns: tour, total_distance, cpu_time
    """
    start_time = time.perf_counter()

    # Step 1: Generate all N*(N-1)/2 edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(coords[i], coords[j])
            edges.append((dist, i, j))

    # Step 2: Sort edges globally by distance (ascending)
    edges.sort(key=lambda x: x[0])

    # Step 3: Greedy Edge Selection
    dsu = DisjointSetUnion(n)
    degrees = [0] * n
    adj_list = [[] for _ in range(n)]
    selected_edges_count = 0
    total_distance = 0.0

    for dist, u, v in edges:
        if degrees[u] < 2 and degrees[v] < 2:
            if selected_edges_count < n - 1:
                if dsu.find(u) != dsu.find(v):
                    dsu.union(u, v)
                    degrees[u] += 1
                    degrees[v] += 1
                    adj_list[u].append(v)
                    adj_list[v].append(u)
                    selected_edges_count += 1
                    total_distance += dist
            elif selected_edges_count == n - 1:
                degrees[u] += 1
                degrees[v] += 1
                adj_list[u].append(v)
                adj_list[v].append(u)
                selected_edges_count += 1
                total_distance += dist
                break

    # Step 4: Reconstruct Hamiltonian Tour from 2-regular graph
    tour = [0]
    visited = [False] * n
    visited[0] = True
    current = 0

    for _ in range(n - 1):
        for nxt in adj_list[current]:
            if not visited[nxt]:
                visited[nxt] = True
                tour.append(nxt)
                current = nxt
                break

    cpu_time = time.perf_counter() - start_time

    if output_path:
        save_solution(tour, total_distance, cpu_time, output_path)

    return tour, total_distance, cpu_time


def main():
    if len(sys.argv) > 1:
        instance_files = sys.argv[1:]
    else:
        data_dir = "data"
        if os.path.exists(data_dir):
            instance_files = [os.path.join(data_dir, f"Pb{i}.txt") for i in range(1, 6)]
        else:
            instance_files = [f"Pb{i}.txt" for i in range(1, 6)]

    print("=" * 65)
    print("Modified Kruskal (Greedy Edge Insertion) Heuristic Results")
    print("=" * 65)
    print(f"{'Instance':<15} | {'Size (n)':<10} | {'Best Value':<15} | {'CPU Time (s)':<12}")
    print("-" * 65)

    for file_path in instance_files:
        if not os.path.exists(file_path):
            continue
        instance_name = os.path.basename(file_path)
        out_dir = os.path.join("results", "kruskal_tsp")
        out_file = os.path.join(out_dir, f"sol_{instance_name}")

        n, coords = read_instance(file_path)
        tour, best_val, cpu_t = solve_kruskal_tsp(n, coords, output_path=out_file)

        print(f"{instance_name:<15} | {n:<10} | {best_val:<15.3f} | {cpu_t:<12.5f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
