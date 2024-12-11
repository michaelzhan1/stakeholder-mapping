import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib import animation as anim
from matplotlib.lines import Line2D
from IPython.display import HTML

matplotlib.use('Agg')

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

        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

        if self.pos is None:
            self.pos = nx.spring_layout(G)

        nx.draw(G, self.pos, with_labels=False, node_size=900, node_color='skyblue',
                font_size=10, font_weight='bold', edge_color='grey', ax=self.ax, width=2, arrows=False)
        
        nx.draw_networkx_labels(G, self.pos, labels=self.node_labels, font_size=12, font_color="black", ax=self.ax)

        self.ax.set_title(f"Stakeholder Network Graph at Timestep {frame + 1}")

        if isinstance(self.actions[frame], tuple):
            (negotiator, recipient) = self.actions[frame]
            subtitle = f'{self.node_labels[negotiator]} reaches out to {self.node_labels[recipient]}'
            # Draw engagement attempts
            if (adj_matrix == self.adj_matrices[frame+1]).all():
                edge_color = "red"
            else:
                edge_color = "green"
            nx.draw_networkx_edges(G, self.pos, edgelist=[self.actions[frame]], style="dotted", edge_color=edge_color, width=2, ax=self.ax, arrows=True, arrowsize=30)
        else:
            subtitle = self.actions[frame]
        self.ax.text(0.5, 0.96, subtitle, transform=self.ax.transAxes,
                        ha='center', fontsize=12, color='gray')
        self.ax.margins(0.15)


        legend_elements = [
            Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="Failed Attempt"),
            Line2D([0], [0], color="green", linestyle="--", linewidth=1.5, label="Successful Attempt"),
            Line2D([0], [0], color="grey", linestyle="-", linewidth=1.5, label="Existing Relationship")
        ]

        # Add legend to plot
        self.ax.legend(handles=legend_elements, loc="lower right", bbox_to_anchor=(1, -0.1))
        


    def animate(self, save=False, return_html=False):
        ani = FuncAnimation(self.fig, self.update, frames=len(self.adj_matrices),
                            interval=self.interval, repeat=True)
        
        if save:
            writer = anim.PillowWriter(fps=0.67)
            ani.save("stakeholder_network.gif", writer=writer)
        if return_html:
            return HTML(ani.to_jshtml())


