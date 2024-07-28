import numpy as np
import networkx as nx
from typing import Optional


def optimal_sequence_bijection_rij(r_ij, C: Optional[float] = None):
    """
    @brief Computes the optimal sequence bijection according
    Latecki et al. Optimal Subsequence Bijection
    @param r_ij: Matrix defined by the sequences A and B
    @param C: Optional jump cost, if none is given it is automatically computed based on the input
    @return: Optimal sequence bijection
    """
    m = np.shape(r_ij)[0]
    n = np.shape(r_ij)[1]
    if m > n:
        r_ij = np.pad(r_ij, ((0, 0), (0, m - n)), 'constant', constant_values=np.inf)
        n = m

    g = nx.DiGraph()
    r = np.ones((m, n))

    for i in range(m):
        for j in range(n):
            r[i][j] = abs(r_ij[i, j])
            g.add_node("{},{}".format(i, j))

    jump_cost = 1.0
    if C is None:
        min_value = np.amin(r, axis=1)
        jump_cost = np.mean(min_value) + np.std(min_value)
    else:
        jump_cost = C

    for i in range(m):
        for j in range(n):
            for k in range(m):
                for l in range(n):
                    if i < k and j < l:
                        src = "{},{}".format(i, j)
                        dst = "{},{}".format(k, l)
                        r[i][j] = np.sqrt((k - i - 1) ** 2
                                          + (l - j - 1) ** 2) * jump_cost \
                                  + (r_ij[i, j]) ** 2
                        r[i][j] = round(r[i][j], 3)
                        g.add_edge(src, dst, length=r[i][j])
                    # else: # other nodes have infinite weight therfore are not visitable
                    #    g.add_edge("{},{}".format(i, j), "{},{}".format(k, l), weight=np.inf)

    best_path = []
    best_length = np.inf
    for i in range(m):
        for j in range(n):
            src = "{},{}".format(i, j)
            paths = nx.single_source_dijkstra_path(g, src, weight="length")
            lengths = nx.single_source_dijkstra_path_length(g, src, weight="length")
            del paths[src]
            for dst, path in paths.items():
                if lengths[dst] <= best_length and len(path) >= len(best_path):
                    best_length = lengths[dst]
                    best_path = path

    map_dict = {}
    for match in best_path:
        split = match.split(",")
        map_dict[int(split[0])] = int(split[1])
    return map_dict, best_length


def optimal_sequence_bijection(seq_a: list, seq_b: list, C: Optional[float] = None):
    """
    @brief Computes the optimal sequence bijection according
    Latecki et al. Optimal Subsequence Bijection
    @param seq_a: Sequence a (needs to be shorter or equal to b)
    @param seq_b: Sequence b (needs to be longer or equal to a)
    @param C: Optional jump cost, if none is given it is automatically computed based on the input
    @return: Optimal sequence bijection
    """
    m = len(seq_a)
    n = len(seq_b)
    assert (m <= n)
    r = np.zeros((m, n)) * np.inf

    for i in range(m):
        for j in range(n):
            r[i][j] = abs(seq_a[i] - seq_b[j])

    return optimal_sequence_bijection_rij(r, C)
