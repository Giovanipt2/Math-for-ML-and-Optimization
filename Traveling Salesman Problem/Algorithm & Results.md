# Traveling Salesman Problem (TSP) — Heuristic Resolution Report

**Course:** Mathematics for Machine Learning and Optimization — Metaheuristics
**Author:** Giovani Mambrim Leme
**Date:** October 2026

---

## 1. Introduction & Problem Overview

The Traveling Salesman Problem (TSP) is a classic NP-hard combinatorial optimization problem. Given a set of $n$ cities in a 2D Euclidean plane, the goal is to find a closed Hamiltonian circuit of minimum total distance that visits every city exactly once.

Due to the factorial growth of the search space ($(n-1)! / 2$ possible tours), exact enumeration becomes intractable for large values of $n$. In this project, we implement and compare two distinct constructive greedy heuristics:

1. **Multi-Start Nearest Neighbor (MS-NN)**
2. **Modified Kruskal / Greedy Edge Insertion**

---

## 2. Implemented Algorithms

### 2.1. Multi-Start Nearest Neighbor (MS-NN)

#### Intuition & Strategy
The classic Nearest Neighbor heuristic builds a tour iteratively by starting at a given city and greedily stepping to the closest unvisited city. However, its solution quality strongly depends on the choice of the initial vertex. To eliminate this bias, **Multi-Start NN** runs the algorithm $n$ times — using each city as a starting point once — and retains the globally best tour found.

#### Pseudocode
```text
Algorithm MultiStartNearestNeighbor(n, coords):
    Compute distance matrix Dist[n][n]
    best_dist = INFINITY
    best_tour = []

    for start_node = 0 to n - 1:
        visited = array of size n (all False)
        tour = [start_node]
        visited[start_node] = True
        current = start_node

        for step = 1 to n - 1:
            next_node = argmin_{v not in visited} (Dist[current][v])
            visited[next_node] = True
            tour.append(next_node)
            current = next_node

        total_dist = length(tour) + Dist[current][start_node]

        if total_dist < best_dist:
            best_dist = total_dist
            best_tour = tour

    return best_tour, best_dist
```

#### Complexity Analysis

- **Time Complexity:** $(O(n^3))$. Evaluating $(n)$ candidate starting cities, where each run performs $(n-1)$ steps inspecting up to $(n)$ unvisited nodes, requires $(O(n^2))$ per start, yielding $(O(n^3))$ overall.
- **Space Complexity:** $(O(n^2))$ to store the precomputed $(n \times n)$ Euclidean distance matrix.

### 2.2. Modified Kruskal (Greedy Edge Insertion)

#### Intuition & Strategy

Kruskal's classic algorithm constructs a Minimum Spanning Tree (MST) by sorting all edges and greedily inserting the shortest valid edge that does not form a cycle. To adapt Kruskal for the TSP, we enforce two topological constraints to guarantee a **2-regular graph (Hamiltonian cycle):**

1. **Degree Constraint:** No vertex can have a degree greater than 2.
2. **Sub-cycle Prevention:** For the first $(n-1)$ edges, an edge cannot connect two vertices already in the same connected component. This condition is tracked efficiently using a **Disjoint Set Union (DSU) / Union-Find** structure.

The $(n)$-th edge connects the two remaining degree-1 endpoints, closing the complete tour.

#### Pseudocode

```text
Algorithm ModifiedKruskalTSP(n, coords):
    Generate all n*(n-1)/2 edges (dist, u, v)
    Sort edges in ascending order by dist

    Initialize DSU structure of size n
    Initialize degrees array of size n (all 0)
    selected_edges = 0
    total_dist = 0

    for edge (d, u, v) in sorted_edges:
        if degrees[u] < 2 and degrees[v] < 2:
            if selected_edges < n - 1:
                if DSU.find(u) != DSU.find(v):
                    DSU.union(u, v)
                    degrees[u] += 1
                    degrees[v] += 1
                    add (u, v) to adjacency list
                    selected_edges += 1
                    total_dist += d

            else if selected_edges == n - 1:
                degrees[u] += 1
                degrees[v] += 1
                add (u, v) to adjacency list
                selected_edges += 1
                total_dist += d
                break

    Reconstruct tour from 2-regular adjacency list
    return tour, total_dist
```

#### Complexity Analysis

- **Time Complexity:** $(O(n^2 \log n))$. Generating $(E = n(n-1)/2 = O(n^2))$ edges takes $(O(n^2))$ time. Sorting these $(E)$ edges takes $(O(E \log E) = O(n^2 \log n))$. DSU operations with path compression run in near-constant amortized time, $(O(\alpha(n)))$, where $(\alpha)$ is the inverse Ackermann function. Reconstructing the tour takes $(O(n))$ time.
- **Space Complexity:** $(O(n^2))$ to store the edge list containing $(n(n-1)/2)$ edges.

## 3. Comparative Experimental Results

Both algorithms were executed on the five provided benchmark instances. All metrics were measured using Python's `time.perf_counter()` on Apple Silicon.

| Instance | Size ($(n)$) | MS-NN Best Value | MS-NN CPU Time (s) | Modified Kruskal Best Value | Modified Kruskal CPU Time (s) | Best Heuristic |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **Pb1.txt** | 100 | **313.195** | 0.01728 | 337.576 | **0.00151** | MS-NN |
| **Pb2.txt** | 200 | 433.493 | 0.10721 | **400.777** | **0.00477** | Modified Kruskal |
| **Pb3.txt** | 400 | 1939.946 | 0.80037 | **1876.742** | **0.02123** | Modified Kruskal |
| **Pb4.txt** | 800 | 6651.265 | 6.56021 | **6640.599** | **0.08281** | Modified Kruskal |
| **Pb5.txt** | 1000 | 10203.117 | 13.08530 | **9935.725** | **0.14595** | Modified Kruskal |

## 4. Discussion & Key Takeaways

### 4.1. Solution Quality

- On smaller instances ($(n = 100)$), **Multi-Start Nearest Neighbor (MS-NN)** achieved a slightly better tour value (313.195 versus 337.576).
- As $(n)$ increases ($(n \geq 200)$), **Modified Kruskal** consistently outperforms MS-NN in terms of solution quality. On `Pb5.txt` ($(n = 1000)$), Modified Kruskal achieved a total tour distance of **9935.725**, compared to **10203.117** for MS-NN.
- **Reasoning:** Nearest Neighbor can fall into a local-view trap in which the final unvisited cities are far apart, forcing long jumps to complete the tour. In contrast, Modified Kruskal evaluates edges globally across the entire graph, which can help produce shorter tours.

### 4.2. Computational Efficiency

- **Modified Kruskal** is significantly faster than MS-NN. For $(n = 1000)$, Modified Kruskal finished in **0.146 seconds**, whereas MS-NN required **13.085 seconds**, making Modified Kruskal approximately **90 times faster**.
- This result is consistent with the theoretical complexity estimates: $(O(n^2 \log n))$ scales significantly better than $(O(n^3))$ as the problem size increases.

### 4.3. Conclusion

Both heuristics completed their executions within the 60-second CPU time limit. However, **Modified Kruskal / Greedy Edge Insertion** proved to be the superior overall strategy for the larger benchmark instances, offering both faster execution and better solution quality.
