import leidenalg as la
import igraph as ig
from collections import defaultdict


def main():
    # example from https://chatgpt.com/share/66f72316-df6c-8000-88e9-b726fb93e52f
    graph = defaultdict(dict)
    graph[0][1] = 5
    graph[0][2] = 7
    graph[0][3] = 6
    graph[0][4] = 3

    graph[1][5] = 8
    graph[1][6] = 4

    graph[2][3] = 8
    graph[2][7] = 5

    graph[3][4] = 6
    
    graph[4][5] = 5
    graph[4][6] = 5

    graph[5][6] = 7
    graph[5][7] = 6
    graph[6][7] = 1
    

    adj = [[0] * 8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            if i in graph and j in graph[i]:
                adj[i][j] = graph[i][j]
                adj[j][i] = graph[i][j]
    
    for row in adj:
        print(row)

    g = ig.Graph.Weighted_Adjacency(adj, mode="undirected")
    

    # print the edge weights of g
    print(g.es["weight"])

    # compute the partition of the graph, accounting for edge weights
    partition = la.find_partition(g, la.ModularityVertexPartition, weights=g.es["weight"])
    
    # label each edge with its edge weight
    g.es["label"] = g.es["weight"]

    g.vs["label"] = [
        'Alex',
        'Darren',
        'Lina',
        'Maya',
        'Leila',
        'Sophie',
        'Emily',
        'Jonas'
    ]

    # create a plot of partition and save it to "out.png"
    ig.plot(partition, "out.png", vertex_size=30, vertex_label_size=10, edge_width=1, edge_arrow_size=1)


if __name__ == "__main__":
    main()