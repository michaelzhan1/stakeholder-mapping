import leidenalg as la
import igraph as ig
from collections import defaultdict


def main():
    graph_dict = defaultdict(dict)
    graph_dict[1][2] = 1
    graph_dict[1][3] = 1
    graph_dict[2][3] = 1

    graph_dict[4][5] = 1
    graph_dict[5][6] = 1
    graph_dict[6][4] = 1

    graph_dict[1][4] = 10
    graph_dict[2][5] = 10
    graph_dict[3][6] = 10

    adj = [[0] * 7 for _ in range(7)]
    for i in range(1, 7):
        for j in range(1, 7):
            if i in graph_dict and j in graph_dict[i]:
                adj[i][j] = graph_dict[i][j]
                adj[j][i] = graph_dict[i][j]
    
    for row in adj:
        print(row)

    g = ig.Graph.Weighted_Adjacency(adj, mode="undirected")
    

    # print the edge weights of g
    print(g.es["weight"])

    # compute the partition of the graph, accounting for edge weights
    partition = la.find_partition(g, la.ModularityVertexPartition, weights=g.es["weight"])
    
    # label each edge with its edge weight
    g.es["label"] = g.es["weight"]

    # create a plot of partition and save it to "out.png"
    ig.plot(partition, "out.png", vertex_size=30, vertex_label_size=10, edge_width=1, edge_arrow_size=1)


if __name__ == "__main__":
    main()