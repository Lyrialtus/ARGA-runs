'''Part 2'''

from itertools import combinations
import networkx as nx

from graph import ARCGraph

class Image:
    '''Graph -> (Image) -> Task'''
    # Some abstractions were retired
    # Abstraction order matters for short runs
    abstractions = ["nag", "nbccg", "mcccg", "nbvccg", "nbhccg"]
    abstraction_getters = {
        "nag": "get_no_abstraction_graph",
        "nbccg": "get_non_black_connected_components_graph",
        "mcccg": "get_multicolor_connected_components_graph",
        "nbvccg": "get_non_background_vertical_connected_components_graph",
        "nbhccg": "get_non_background_horizontal_connected_components_graph"}
    multicolor_abstractions = ["nag", "mcccg"]

    def __init__(self, task, grid=None, width=None, height=None, graph=None, name="image"):
        """
        an image represents a 2D-grid of pixels
        the coordinate system follows the convention of 0,0
        being the top left pixel of the image
        :param grid: a grid that represent the image
        :param width: if a grid is not given, determines the width of the graph
        :param height: if a grid is not given, determines the height of the graph
        :param graph: if a networkx graph is given, use it directly as the graph
        """
        self.task = task
        self.all_colors = task.all_colors

        self.name = name
        self.grid = grid
        self.background_color = task.bgc

        if not grid and not graph:
            # create a graph with default color
            self.width = width
            self.height = height
            self.image_size = (height, width)
            self.graph = nx.grid_2d_graph(height, width)
            nx.set_node_attributes(self.graph, 0, "color")
        elif graph:
            self.width = max(node[1] for node in graph.nodes()) + 1
            self.height = max(node[0] for node in graph.nodes()) + 1
            self.image_size = (self.height, self.width)
            self.graph = graph
        else:
            # create a graph with the color in given grid
            self.width = len(grid[0])
            self.height = len(grid)
            self.image_size = (self.height, self.width)
            self.graph = nx.grid_2d_graph(self.height, self.width)
            for r, row in enumerate(grid):
                for c, color in enumerate(row):
                    self.graph.nodes[r, c]["color"] = color

        self.corners = {(0, 0), (0, self.width - 1), (self.height - 1, 0),
                        (self.height - 1, self.width - 1)}

    def add_edges(self, graph, bg=0):
        """
        add edges between the abstracted nodes
        """
        for node_1, node_2 in combinations(graph.nodes, 2):
            nodes_1 = graph.nodes[node_1]["nodes"]
            nodes_2 = graph.nodes[node_2]["nodes"]
            y1, x1 = zip(*nodes_1)
            y2, x2 = zip(*nodes_2)
            c1 = (sum(x1) / len(nodes_1), sum(y1) / len(nodes_1))
            c2 = (sum(x2) / len(nodes_2), sum(y2) / len(nodes_2))

            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                            if self.graph.nodes[n1[0], column_index]["color"] != bg:
                                break
                        else:
                            if abs(c1[0] - c2[0]) > abs(c1[1] - c2[1]):
                                graph.add_edge(node_1, node_2, direction="horizontal")
                                break
                    elif n1[1] == n2[1]:  # two nodes on the same column
                        for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                            if self.graph.nodes[row_index, n1[1]]["color"] != bg:
                                break
                        else:
                            if abs(c1[0] - c2[0]) < abs(c1[1] - c2[1]):
                                graph.add_edge(node_1, node_2, direction="vertical")
                                break
                else:
                    continue
                break

    def get_no_abstraction_graph(self, graph=None):
        """
        return an abstracted graph where a node is defined as:
        the entire graph as one multi-color node
        """
        if not graph:
            graph = self.graph
        nag = nx.Graph()

        sub_nodes = []
        sub_nodes_color = []
        for node in graph.nodes():
            sub_nodes.append(node)
            sub_nodes_color.append(graph.nodes[node]["color"])
        nag.add_node((0, 0), nodes=sub_nodes, color=sub_nodes_color, size=len(sub_nodes), z=0)

        return ARCGraph(nag, self.name, self, "nag")

    def get_non_black_connected_components_graph(self, graph=None):
        """
        return an abstracted graph where a node is defined as:
        a group of adjacent pixels of the same color in the original graph,
        excluding background color
        """
        if not graph:
            graph = self.graph
        nbccg = nx.Graph()

        for color in self.all_colors:
            if color == self.background_color:
                continue
            color_nodes = (node for node, data in graph.nodes(
                data=True) if data.get("color") == color)
            color_subgraph = graph.subgraph(color_nodes)
            color_connected_components = nx.connected_components(color_subgraph)
            for i, component in enumerate(color_connected_components):
                nbccg.add_node((color, i), nodes=list(component),
                               color=color, size=len(component), z=0)

        self.add_edges(nbccg, self.background_color)
        return ARCGraph(nbccg, self.name, self, "nbccg")

    def get_multicolor_connected_components_graph(self, graph=None):
        """
        return an abstracted graph where a node is defined as:
        a group of adjacent pixels of any non-background color in the original graph
        """
        if not graph:
            graph = self.graph
        mcccg = nx.Graph()

        non_background_nodes = [node for node, data in graph.nodes(
            data=True) if data["color"] != self.background_color]
        color_subgraph = graph.subgraph(non_background_nodes)
        multicolor_connected_components = nx.connected_components(color_subgraph)

        for i, component in enumerate(multicolor_connected_components):
            sub_nodes = []
            sub_nodes_color = []
            for node in component:
                sub_nodes.append(node)
                sub_nodes_color.append(graph.nodes[node]["color"])
            mcccg.add_node((len(sub_nodes), i), nodes=sub_nodes,
                           color=sub_nodes_color, size=len(sub_nodes), z=0)

        self.add_edges(mcccg, self.background_color)
        return ARCGraph(mcccg, self.name, self, "mcccg")

    def get_non_background_vertical_connected_components_graph(self, graph=None):
        """
        return an abstracted graph where a node is defined as:
        a group of vertically adjacent pixels of the same color in the original graph,
        excluding background color
        """
        if not graph:
            graph = self.graph
        nbvccg = nx.Graph()

        for color in self.all_colors:
            color_connected_components = []
            if color == self.background_color:
                continue
            for column in range(self.width):
                color_nodes = (node for node, data in graph.nodes(data=True) if
                               node[1] == column and data.get("color") == color)
                color_subgraph = graph.subgraph(color_nodes)
                color_connected_components.extend(list(nx.connected_components(color_subgraph)))
            for i, component in enumerate(color_connected_components):
                nbvccg.add_node((color, i), nodes=component, color=color, size=len(component), z=0)

        self.add_edges(nbvccg, self.background_color)
        return ARCGraph(nbvccg, self.name, self, "nbvccg")

    def get_non_background_horizontal_connected_components_graph(self, graph=None):
        """
        return an abstracted graph where a node is defined as:
        a group of horizontally adjacent pixels of the same color in the original graph,
        excluding background color
        """
        if not graph:
            graph = self.graph
        nbhccg = nx.Graph()

        for color in self.all_colors:
            color_connected_components = []
            if color == self.background_color:
                continue
            for row in range(self.height):
                color_nodes = (node for node, data in graph.nodes(data=True) if
                               node[0] == row and data.get("color") == color)
                color_subgraph = graph.subgraph(color_nodes)
                color_connected_components.extend(list(nx.connected_components(color_subgraph)))
            for i, component in enumerate(color_connected_components):
                nbhccg.add_node((color, i), nodes=component, color=color, size=len(component), z=0)

        self.add_edges(nbhccg, self.background_color)
        return ARCGraph(nbhccg, self.name, self, "nbhccg")
