import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class AnimatedGraph:
    def __init__(self, adj_matrices, actions, node_labels=None, interval=1000):
        self.adj_matrices = adj_matrices
        self.actions = actions
        self.interval = interval
        self.node_labels = node_labels if node_labels else {i: str(i + 1) for i in range(adj_matrices[0].shape[0])}
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.pos = None  
        plt.close(self.fig)  

    def update(self, frame):
        self.ax.clear() 

        adj_matrix = self.adj_matrices[frame]

        G = nx.from_numpy_array(adj_matrix)

        if self.pos is None:
            self.pos = nx.spring_layout(G)

        nx.draw(G, self.pos, with_labels=False, node_size=500, node_color='skyblue',
                font_size=10, font_weight='bold', edge_color='gray', ax=self.ax)
        
        nx.draw_networkx_labels(G, self.pos, labels=self.node_labels, font_size=12, font_color="black", ax=self.ax)

        self.ax.set_title(f"Stakeholder Network Graph at Timestep {frame + 1}")

        if isinstance(self.actions[frame], tuple):
            (negotiator, recipient) = self.actions[frame]
            subtitle = f'{self.node_labels[negotiator]} reaches out to {self.node_labels[recipient]}'
        else:
            subtitle = self.actions[frame]
        self.ax.text(0.5, 0.96, subtitle, transform=self.ax.transAxes,
                        ha='center', fontsize=12, color='gray')


    def animate(self):
        ani = FuncAnimation(self.fig, self.update, frames=len(self.adj_matrices),
                            interval=self.interval, repeat=True)
        return HTML(ani.to_jshtml())


