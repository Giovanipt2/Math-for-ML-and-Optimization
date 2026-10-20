"""
Multi-Start Nearest Neighbor (MS-NN) Heuristic for the Traveling Salesman Problem (TSP)

Course: Mathematics for Machine Learning & Optimization - Metaheuristics
Authors: Student Team
"""

import sys
import os
import math
import time


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


def compute_tour_length(tour, dist_matrix):
    """Computes the total Euclidean length of a closed Hamiltonian tour."""
    n = len(tour)
    length = sum(dist_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))
    return length


def save_solution(tour, total_distance, cpu_time, output_path):
    """Saves tour, total distance, and CPU time to a text file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(" ".join(map(str, tour)) + "\n")
        f.write(f"{total_distance:.3f}\n")
        f.write(f"{cpu_time:.5f}\n")


def solve_ms_nn(n, coords, output_path=None):
    """
    Executes the Multi-Start Nearest Neighbor Heuristic.
    Returns: best_tour, best_length, cpu_time
    """
    start_time = time.perf_counter()

    # Precompute Euclidean distance matrix
    dist_matrix = [[euclidean_distance(coords[i], coords[j]) for j in range(n)] for i in range(n)]

    best_length = float('inf')
    best_tour = []

    for start_city in range(n):
        visited = [False] * n
        tour = [start_city]
        visited[start_city] = True
        current_city = start_city

        # Build tour greedily
        for _ in range(n - 1):
            next_city = -1
            min_dist = float('inf')

            for candidate in range(n):
                if not visited[candidate]:
                    d = dist_matrix[current_city][candidate]
                    if d < min_dist:
                        min_dist = d
                        next_city = candidate

            visited[next_city] = True
            tour.append(next_city)
            current_city = next_city

        # Complete the loop
        tour_len = compute_tour_length(tour, dist_matrix)

        # Save solution incrementally upon improvement
        if tour_len < best_length:
            best_length = tour_len
            best_tour = tour.copy()
            if output_path:
                current_time = time.perf_counter() - start_time
                save_solution(best_tour, best_length, current_time, output_path)

    cpu_time = time.perf_counter() - start_time
    return best_tour, best_length, cpu_time


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
    print("Multi-Start Nearest Neighbor (MS-NN) Heuristic Results")
    print("=" * 65)
    print(f"{'Instance':<15} | {'Size (n)':<10} | {'Best Value':<15} | {'CPU Time (s)':<12}")
    print("-" * 65)

    for file_path in instance_files:
        if not os.path.exists(file_path):
            continue
        instance_name = os.path.basename(file_path)
        out_dir = os.path.join("results", "ms_nn")
        out_file = os.path.join(out_dir, f"sol_{instance_name}")

        n, coords = read_instance(file_path)
        best_tour, best_val, cpu_t = solve_ms_nn(n, coords, output_path=out_file)

        print(f"{instance_name:<15} | {n:<10} | {best_val:<15.3f} | {cpu_t:<12.5f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
