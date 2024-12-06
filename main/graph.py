'''Part 1'''

import numpy as np

class ARCGraph:
    '''(Graph) -> Image -> Task'''

    filter_ops = ["filter_by_nothing",
                  "filter_by_color",
                  "filter_by_size",
                  "filter_by_degree",
                  "filter_by_neighbor_color"]

    param_binding_ops = ["param_bind_neighbor_by_color",
                         "param_bind_node_by_color",
                         "param_bind_node_by_size",
                         "param_bind_node_by_shape"]

    param_values = {
        'flip_rotation': [('vertical', 0), ('horizontal', 0),
                          (None, 90), ('vertical', 90), ('horizontal', 90),
                          (None, 180), (None, 270)],
        'limit': ['max', 1, 2],
        'move_mode': ['touching', 'phasing', 'piercing', 'blasting'],
        'relative_position': ['source', 'target', 'middle'],
        'image_point': ['top', 'bottom', 'left', 'right',
                        'top_left', 'top_right', 'bottom_left', 'bottom_right'],
        'direction': ['up', 'down', 'left', 'right',
                      'up_left', 'up_right', 'down_left', 'down_right']}

    transformation_ops = {
        "nag": ["flip_rotate"],
        "nbccg": ["update_color", "move", "extend", "flip_rotate",
                  "add_border", "fill_rectangle", "hollow_rectangle",
                  "mirror", "insert"],
        "mcccg": ["swap_colors", "move", "add_border", "fill_rectangle",
                  "hollow_rectangle", "insert"],
        "nbvccg": ["update_color", "move"],
        "nbhccg": ["update_color", "move"]}

    dynamic_parameters = ["color",
                          "direction",
                          "image_point",
                          "mirror_axis"]

    def __init__(self, graph, name, image, abstraction=None):
        '''setup'''
        self.graph = graph
        self.image = image
        self.abstraction = abstraction
        if abstraction is None:
            self.name = name
        elif abstraction in name.split("_"):
            self.name = name
        else:
            self.name = name + "_" + abstraction
        if self.abstraction in image.multicolor_abstractions:
            self.is_multicolor = True
        else:
            self.is_multicolor = False

    # ------------------------------------- filters ------------------------------------------
    # filters take the form of filter(node, params), return true if node satisfies filter
    def filter_by_nothing(self, _):
        '''return true'''
        return True

    def filter_by_color(self, node, color, exclude: bool = False):
        """
        return true if node has given color
        if exclude, return true if node does not have given color
        """
        if self.is_multicolor and isinstance(color, int):
            if not exclude:
                return color in self.graph.nodes[node]["color"]
            return color not in self.graph.nodes[node]["color"]
        if color == "same":
            color = self.graph.nodes[node]["color"]
        if not exclude:
            return self.graph.nodes[node]["color"] == color
        return self.graph.nodes[node]["color"] != color

    def filter_by_size(self, node, size, exclude: bool = False):
        """
        return true if node has size equal to given size
        if exclude, return true if node does not have size equal to given size
        """
        if size == "max":
            size = self.get_attribute_max("size")
        elif size == "min":
            size = self.get_attribute_min("size")

        if size == "odd" and not exclude:
            return self.graph.nodes[node]["size"] % 2 != 0
        if size == "odd" and exclude:
            return self.graph.nodes[node]["size"] % 2 == 0

        if exclude:
            return self.graph.nodes[node]["size"] != size
        return self.graph.nodes[node]["size"] == size

    def filter_by_degree(self, node, degree, exclude: bool = False):
        """
        return true if node has degree equal to given degree
        if exclude, return true if node does not have degree equal to given degree
        """
        if not exclude:
            return self.graph.degree[node] == degree
        return self.graph.degree[node] != degree

    def filter_by_neighbor_color(self, node, color, exclude: bool = False):
        """
        return true if node has a neighbor of a given color
        if exclude, return true if node does not have a neighbor of a given color
        """
        if color == "same":
            color = self.graph.nodes[node]["color"]
        for neighbor in self.graph.neighbors(node):
            if not exclude:
                if self.graph.nodes[neighbor]["color"] == color:
                    return True
            elif exclude:
                if self.graph.nodes[neighbor]["color"] != color:
                    return True
        return False

    # --------------------------------- parameter binding functions --------------------------------
    # parameter binding takes the form of param_binding(node, params),
    # return node_2 if rel(node, node_2) and filter(node_2, params).
    # ex. neighbor of node with color blue

    def param_bind_neighbor_by_color(self, node, color, exclude: bool = False):
        """
        return the neighbor of node satisfying given color filter
        """
        if color == "same":
            color = self.graph.nodes[node]["color"]
        for neighbor in self.graph.neighbors(node):
            if self.filter_by_color(neighbor, color, exclude):
                return neighbor
        return None

    def param_bind_node_by_color(self, node, color, exclude: bool = False):
        """
        return any node satisfying given color filter
        """
        if color == "same":
            color = self.graph.nodes[node]["color"]
        for other_node in self.graph.nodes():
            if other_node != node and self.filter_by_color(other_node, color, exclude):
                return other_node
        return None

    def param_bind_node_by_size(self, size, exclude: bool = False):
        """
        return any node in graph satisfying given size filter
        """
        for node in self.graph.nodes():
            if self.filter_by_size(node, size, exclude):
                return node
        return None

    def param_bind_node_by_shape(self, node):
        """
        return any other node in the graph with the same shape as node
        """
        target_shape = self.get_shape(node)
        for param_bind_node in self.graph.nodes:
            if param_bind_node != node:
                candidate_shape = self.get_shape(param_bind_node)
                if candidate_shape == target_shape:
                    return param_bind_node
        return None

    # ------------------------------------------ transformations -----------------------------------
    def update_color(self, node, color):
        """
        update node color to given color
        """
        if color is None or color == 'same':
            return self
        self.graph.nodes[node]["color"] = color
        return self

    def swap_colors(self, node):
        '''swap if 2 colors in multicolor'''
        colors = self.graph.nodes[node]["color"]
        uni = np.unique(colors)
        if len(uni) == 2:
            for i, color in enumerate(colors):
                if color == uni[0]:
                    self.graph.nodes[node]["color"][i] = uni[1]
                else:
                    self.graph.nodes[node]["color"][i] = uni[0]
        return self

    def move(self, node, direction, limit, move_mode):
        """
        move node in a given direction until limit, another node, or the edge of the image
        """
        if limit == 'max':
            limit = 30
        delta_x = 0
        delta_y = 0
        if direction is None:
            return self
        if direction in ('up', 'up_left', 'up_right'):
            delta_y = -1
        elif direction in ('down', 'down_left', 'down_right'):
            delta_y = 1
        if direction in ('left', 'up_left', 'down_left'):
            delta_x = -1
        elif direction in ('right', 'up_right', 'down_right'):
            delta_x = 1

        for _ in range(limit):
            updated_nodes = []
            for sub_node in self.graph.nodes[node]["nodes"]:
                updated_nodes.append((sub_node[0] + delta_y, sub_node[1] + delta_x))
            if not self.check_inbound(updated_nodes):
                break
            if self.check_collision(node, updated_nodes):
                if move_mode == 'touching':
                    break
                if move_mode == 'phasing':
                    continue
            self.graph.nodes[node]["nodes"] = updated_nodes

        if move_mode == 'piercing':
            self.graph.nodes[node]['z'] -= 1
        if move_mode == 'blasting':
            self.graph.nodes[node]['z'] += 1
        return self

    def extend(self, node, direction, move_mode):
        """
        extend node in a given direction,
        if overlap is true, extend node even if it overlaps with another node
        if overlap is false, stop extending before it overlaps with another node
        """
        # How can I do colors?
        updated_sub_nodes = []
        delta_x = 0
        delta_y = 0
        if direction is None:
            return self
        if direction in ('up', 'up_left', 'up_right'):
            delta_y = -1
        elif direction in ('down', 'down_left', 'down_right'):
            delta_y = 1
        if direction in ('left', 'up_left', 'down_left'):
            delta_x = -1
        elif direction in ('right', 'up_right', 'down_right'):
            delta_x = 1

        for sub_node in self.graph.nodes[node]["nodes"]:
            sub_node_y = sub_node[0]
            sub_node_x = sub_node[1]
            updated_sub_nodes.append((sub_node_y, sub_node_x))

            for _ in range(30):
                sub_node_y += delta_y
                sub_node_x += delta_x
                if not self.check_inbound((sub_node_y, sub_node_x)):
                    break
                if self.check_collision(node, [(sub_node_y, sub_node_x)]):
                    if move_mode == 'touching':
                        break
                    if move_mode == 'phasing':
                        continue
                updated_sub_nodes.append((sub_node_y, sub_node_x))

        if move_mode == 'piercing':
            self.graph.nodes[node]['z'] -= 1
        if move_mode == 'blasting':
            self.graph.nodes[node]['z'] += 1
        self.graph.nodes[node]["nodes"] = list(set(updated_sub_nodes))
        self.graph.nodes[node]["size"] = len(updated_sub_nodes)
        return self

    def flip_rotate(self, node, flip_rotation):
        '''flip/rotate the node around its center point'''
        new_subnodes = []
        if flip_rotation[0] == 'vertical':
            max_y = max(subnode[0] for subnode in self.graph.nodes[node]["nodes"])
            min_y = min(subnode[0] for subnode in self.graph.nodes[node]["nodes"])
            for subnode in self.graph.nodes[node]["nodes"]:
                new_y = max_y - (subnode[0] - min_y)
                new_x = subnode[1]
                new_subnodes.append((new_y, new_x))
            self.graph.nodes[node]["nodes"] = new_subnodes
        elif flip_rotation[0] == 'horizontal':
            max_x = max(subnode[1] for subnode in self.graph.nodes[node]["nodes"])
            min_x = min(subnode[1] for subnode in self.graph.nodes[node]["nodes"])
            for subnode in self.graph.nodes[node]["nodes"]:
                new_y = subnode[0]
                new_x = max_x - (subnode[1] - min_x)
                new_subnodes.append((new_y, new_x))
            self.graph.nodes[node]["nodes"] = new_subnodes

        rotate_times = int(flip_rotation[1] / 90)
        max_y = max(subnode[0] for subnode in self.graph.nodes[node]["nodes"])
        min_y = min(subnode[0] for subnode in self.graph.nodes[node]["nodes"])
        max_x = max(subnode[1] for subnode in self.graph.nodes[node]["nodes"])
        min_x = min(subnode[1] for subnode in self.graph.nodes[node]["nodes"])
        for _ in range(rotate_times):
            new_nodes = []
            for sub_node in self.graph.nodes[node]["nodes"]:
                new_node = (min_y + (sub_node[1] - min_x),
                            min_x + (max_x - min_x - (sub_node[0] - min_y)))
                new_nodes.append(new_node)
            self.graph.nodes[node]["nodes"] = new_nodes
        return self

    def add_border(self, node, border_color):
        """
        add a border with thickness 1 and border_color around the given node
        """
        if border_color == "same":
            border_color = self.graph.nodes[node]["color"]
            if self.is_multicolor:
                border_color = self.graph.nodes[node]["color"][0]
        delta = [-1, 0, 1]
        border_pixels = []
        for sub_node in self.graph.nodes[node]["nodes"]:
            for x in delta:
                for y in delta:
                    border_pixel = (sub_node[0] + y, sub_node[1] + x)
                    if (border_pixel not in border_pixels
                        and not self.check_pixel_occupied(border_pixel)):
                        border_pixels.append(border_pixel)
        if len(border_pixels) == 0:
            return self
        new_node_id = self.generate_node_id(border_color)
        color = [border_color for _ in border_pixels] if self.is_multicolor else border_color
        self.graph.add_node(new_node_id, nodes=border_pixels,
                            color=color, size=len(border_pixels), z=0)
        return self

    def fill_rectangle(self, node, fill_color, overlap: bool):
        """
        fill the rectangle containing the given node with the given color
        if overlap is True, fill the rectangle even if it overlaps with other nodes
        """
        if fill_color == "same":
            fill_color = self.graph.nodes[node]["color"]
            if self.is_multicolor:
                fill_color = self.graph.nodes[node]["color"][0]
        all_x = [sub_node[1] for sub_node in self.graph.nodes[node]["nodes"]]
        all_y = [sub_node[0] for sub_node in self.graph.nodes[node]["nodes"]]
        min_x, min_y, max_x, max_y = min(all_x), min(all_y), max(all_x), max(all_y)
        unfilled_pixels = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                pixel = (y, x)
                if pixel not in self.graph.nodes[node]["nodes"]:
                    if overlap:
                        unfilled_pixels.append(pixel)
                    elif not self.check_pixel_occupied(pixel):
                        unfilled_pixels.append(pixel)
        if len(unfilled_pixels) == 0:
            return self
        new_node_id = self.generate_node_id(fill_color)
        color = [fill_color for _ in unfilled_pixels] if self.is_multicolor else fill_color
        self.graph.add_node(new_node_id, nodes=unfilled_pixels,
                            color=color, size=len(unfilled_pixels), z=0)
        return self

    def hollow_rectangle(self, node, fill_color):
        """
        hollowing the rectangle containing the given node with the given color
        """
        if fill_color == "same":
            fill_color = self.graph.nodes[node]["color"]
            if self.is_multicolor:
                fill_color = self.graph.nodes[node]["color"][0]
        all_y = [n[0] for n in self.graph.nodes[node]["nodes"]]
        all_x = [n[1] for n in self.graph.nodes[node]["nodes"]]
        border_y = [min(all_y), max(all_y)]
        border_x = [min(all_x), max(all_x)]
        non_border_pixels = []
        new_subnodes = []
        for subnode in self.graph.nodes[node]["nodes"]:
            if subnode[0] in border_y or subnode[1] in border_x:
                new_subnodes.append(subnode)
            else:
                non_border_pixels.append(subnode)
        self.graph.nodes[node]["nodes"] = new_subnodes

        if fill_color == self.image.background_color or len(non_border_pixels) == 0:
            return self
        new_node_id = self.generate_node_id(fill_color)
        color = [fill_color for _ in non_border_pixels] if self.is_multicolor else fill_color
        self.graph.add_node(new_node_id, nodes=non_border_pixels,
                            color=color, size=len(non_border_pixels), z=0)
        return self

    def mirror(self, node, mirror_axis):
        """
        mirroring a node with respect to the given axis
        mirror_axis takes the form of (y, x) where one of y, x equals None to
        indicate the other being the axis of mirroring
        """
        if not isinstance(mirror_axis, tuple):
            return self
        if mirror_axis[1] is None and mirror_axis[0] is not None:
            axis = mirror_axis[0]
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_y = axis - (subnode[0] - axis)
                new_x = subnode[1]
                new_subnodes.append((new_y, new_x))
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        elif mirror_axis[0] is None and mirror_axis[1] is not None:
            axis = mirror_axis[1]
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_y = subnode[0]
                new_x = axis - (subnode[1] - axis)
                new_subnodes.append((new_y, new_x))
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        return self

    def insert(self, node, object_id, image_point, relative_position):
        """
        insert some pattern identified by object_id at some location,
        the location is defined as, the relative position between the given node and image_point
        for example, image_point=top, relative_position=middle will insert the pattern between
        the given node and the top of the image
        if object_id is -1, use the pattern given by node
        """
        if image_point is None:
            return self
        node_centroid = self.get_centroid(node)
        if not isinstance(image_point, tuple):
            if image_point == 'top':
                image_point = (0, node_centroid[1])
            elif image_point == 'bottom':
                image_point = (self.image.height - 1, node_centroid[1])
            elif image_point == 'left':
                image_point = (node_centroid[0], 0)
            elif image_point == 'right':
                image_point = (node_centroid[0], self.image.width - 1)
            elif image_point == 'top_left':
                image_point = (0, 0)
            elif image_point == 'top_right':
                image_point = (0, self.image.width - 1)
            elif image_point == 'bottom_left':
                image_point = (self.image.height - 1, 0)
            elif image_point == 'bottom_right':
                image_point = (self.image.height - 1, self.image.width - 1)

        if object_id == -1:
            # special id for dynamic objects, which uses the given nodes as objects
            node_object = self.graph.nodes[node]
        else:
            node_object = self.image.task.static_objects_for_insertion[self.abstraction][object_id]

        target_point = self.get_point_from_relative_position(
            node_centroid, image_point, relative_position)
        if target_point is None:
            return self

        # Different version: respecting grid boundaries
        object_centroid = self.get_centroid(node_object["nodes"])
        subnodes_coords = []
        for subnode in node_object["nodes"]:
            delta_y = subnode[0] - object_centroid[0]
            delta_x = subnode[1] - object_centroid[1]
            subnodes_coords.append((target_point[0] + delta_y, target_point[1] + delta_x))
        if len(subnodes_coords) == 0:
            return self

        new_node_id = self.generate_node_id(node_object["color"])
        self.graph.add_node(new_node_id, nodes=subnodes_coords, color=node_object["color"],
                            size=len(subnodes_coords), z=0)
        return self

    # ------------------------------------- utils ------------------------------------------
    def remove_node(self, node):
        """
        remove a node from the graph
        """
        self.graph.remove_node(node)
        return self

    def get_attribute_max(self, attribute_name):
        """
        get the maximum value of the given attribute
        """
        if len(self.graph.nodes) == 0:
            return None
        return max(data[attribute_name] for node, data in self.graph.nodes(data=True))

    def get_attribute_min(self, attribute_name):
        """
        get the minimum value of the given attribute
        """
        if len(self.graph.nodes) == 0:
            return None
        return min(data[attribute_name] for node, data in self.graph.nodes(data=True))

    def get_color(self, node):
        """
        return the color of the node
        """
        if isinstance(node, list):
            return [self.graph.nodes[node_i]["color"] for node_i in node]
        return self.graph.nodes[node]["color"]

    def check_inbound(self, pixels):
        """
        check if given pixels are all within the image boundary
        """
        if not isinstance(pixels, list):
            pixels = [pixels]
        for pixel in pixels:
            y, x = pixel
            if x < 0 or y < 0 or x >= self.image.width or y >= self.image.height:
                return False
        return True

    def check_collision(self, node_id, pixels_list=None):
        """
        check if given pixels_list collide with other nodes in the graph
        node_id is used to retrieve pixels_list if not given
        node_id is also used so that only collision with other nodes are detected
        """
        if pixels_list is None:
            pixels_set = set(self.graph.nodes[node_id]["nodes"])
        else:
            pixels_set = set(pixels_list)
        for node, data in self.graph.nodes(data=True):
            if len(set(data["nodes"]) & pixels_set) != 0 and node != node_id:
                return True
        return False

    def check_pixel_occupied(self, pixel):
        """
        check if a pixel is occupied by any node in the graph
        """
        for _, data in self.graph.nodes(data=True):
            if pixel in data["nodes"]:
                return True
        return False

    def get_shape(self, node):
        """
        given a node, get the shape of the node
        the shape of the node is defined using its pixels shifted so that the top left is 0,0
        """
        sub_nodes = self.graph.nodes[node]["nodes"]
        if len(sub_nodes) == 0:
            return set()
        min_x = min(sub_node[1] for sub_node in sub_nodes)
        min_y = min(sub_node[0] for sub_node in sub_nodes)
        return {(y - min_y, x - min_x) for y, x in sub_nodes}

    def get_centroid(self, node):
        '''get the centroid of the node pixels'''
        if isinstance(node, tuple):
            pixels = self.graph.nodes[node]["nodes"]
        else:
            pixels = node
        size = len(pixels)
        if size == 0:
            return (0, 0)
        center_y = (sum(n[0] for n in pixels) + size // 2) // size
        center_x = (sum(n[1] for n in pixels) + size // 2) // size
        return (center_y, center_x)

    def get_relative_position(self, node_1, node_2):
        """
        direction of where node 2 is relative to node 1,
        what is the direction of going from 1 to 2
        """
        nodes_1 = self.graph.nodes[node_1]["nodes"]
        nodes_2 = self.graph.nodes[node_2]["nodes"]
        if len(nodes_1) == 0 or len(nodes_2) == 0:
            return None

        y1, x1 = zip(*nodes_1)
        y2, x2 = zip(*nodes_2)
        c1 = (sum(x1) / len(nodes_1), sum(y1) / len(nodes_1))
        c2 = (sum(x2) / len(nodes_2), sum(y2) / len(nodes_2))

        for sub_node_1 in nodes_1:
            for sub_node_2 in nodes_2:
                if sub_node_1[0] == sub_node_2[0]:
                    if abs(c1[0] - c2[0]) < abs(c1[1] - c2[1]):
                        continue
                    if sub_node_1[1] < sub_node_2[1]:
                        return 'right'
                    if sub_node_1[1] > sub_node_2[1]:
                        return 'left'
                elif sub_node_1[1] == sub_node_2[1]:
                    if abs(c1[0] - c2[0]) > abs(c1[1] - c2[1]):
                        continue
                    if sub_node_1[0] < sub_node_2[0]:
                        return 'down'
                    if sub_node_1[0] > sub_node_2[0]:
                        return 'up'
        return None

    def get_mirror_axis(self, node_1, node_2):
        """
        get the axis to mirror node_1 with given node_2
        """
        if not self.graph.has_edge(node_1, node_2):
            return (None, None)
        node_2_centroid = self.get_centroid(node_2)
        if self.graph.edges[node_1, node_2]["direction"] == "vertical":
            return (node_2_centroid[0], None)
        return (None, node_2_centroid[1])

    def get_point_from_relative_position(self, filtered_point, relative_point, relative_position):
        """
        get the point to insert new node given
        filtered_point: the centroid of the filtered node
        relative_point: the centroid of the target node, or static point such as (0,0)
        relative_position: the relative position of the filtered_point to the relative_point
        """
        if relative_position == 'source':
            return filtered_point
        elif relative_position == 'target':
            return relative_point
        elif relative_position == 'middle':
            y = (filtered_point[0] + relative_point[0]) // 2
            x = (filtered_point[1] + relative_point[1]) // 2
            return (y, x)
        return None

    # ------------------------------------------ apply -----------------------------------
    def apply(self, abstraction, filters, filter_params, transformation, transformation_params):
        """
        perform a full operation on the abstracted graph
        1. apply filters to get a list of nodes to transform
        2. apply param binding to the filtered nodes to retrieve parameters for the transformation
        3. apply transformation to the nodes
        """
        # Cleaning (a stable version produces no empty nodes)
        empty = []
        for node, data in self.graph.nodes(data=True):
            if data['size'] == 0:
                empty.append(node)
        for node in empty:
            self.remove_node(node)

        transformed_nodes = {}
        for node in self.graph.nodes():
            if self.apply_filters(node, filters, filter_params):
                params = self.apply_param_binding(node, transformation_params)
                transformed_nodes[node] = params
        for node, params in transformed_nodes.items():
            self.apply_transformation(node, transformation, params)

        if abstraction == 'bug':
            print('lol')

    def apply_filters(self, node, filters, filter_params):
        """
        given filters and a node, return True if node satisfies all filters
        """
        satisfy = True
        for filter_, filter_param in zip(filters, filter_params):
            satisfy = satisfy and getattr(self, filter_)(node, **filter_param)
        return satisfy

    def apply_param_binding(self, node, transformation_params):
        """
        handle dynamic parameters: if a dictionary is passed as a parameter value, this means
        the parameter value needs to be retrieved from the parameter-bound nodes during the search
        example: set param "color" to the color of the neighbor with size 1
        """
        transformation_params_retrieved = {}
        for param_key, param_value in transformation_params[0].items():
            transformation_params_retrieved[param_key] = param_value

            if isinstance(param_value, dict):
                param_bind_function = param_value["filters"][0]
                param_bind_function_params = param_value["filter_params"][0]
                target_node = getattr(self, param_bind_function)(node, **param_bind_function_params)
                if target_node is None:
                    transformation_params_retrieved[param_key] = None
                    continue

                #  retrieve value, ex. color of the neighbor with size 1
                if param_key == "color":
                    target_value = self.get_color(target_node)
                elif param_key == "direction":
                    target_value = self.get_relative_position(node, target_node)
                elif param_key == "image_point":
                    target_value = self.get_centroid(target_node)
                elif param_key == "mirror_axis":
                    target_value = self.get_mirror_axis(node, target_node)
                else:
                    raise ValueError("Unsupported dynamic parameter")

                transformation_params_retrieved[param_key] = target_value

        return transformation_params_retrieved

    def apply_transformation(self, node, transformation, transformation_params):
        """
        apply transformation to a node
        currently only allows one transformation
        """
        getattr(self, transformation[0])(node, **transformation_params)

    # ------------------------------------------ meta utils -----------------------------------
    def copy(self):
        """
        return a copy of this ARCGraph object
        """
        return ARCGraph(self.graph.copy(), self.name, self.image, self.abstraction)

    def generate_node_id(self, color):
        """
        find the next available id for a given color,
        ex: if color=1 and there are already (1,0) and (1,1), return (1,2)
        """
        if isinstance(color, list):  # multi-color cases
            color = color[0]
        max_id = 0
        for node in self.graph.nodes():
            if node[0] == color:
                max_id = max(max_id, node[1])
        return (color, max_id + 1)

    def get_pixel_grid(self):
        '''get the 2D-grid projection of the current graph state'''
        grid = np.full(self.image.image_size, self.image.background_color)
        mc = self.abstraction in self.image.multicolor_abstractions
        points = {}
        for _, data in self.graph.nodes(data=True):
            for i, node in enumerate(data["nodes"]):
                if self.check_inbound(node):
                    color = data["color"][i] if mc else data["color"]
                    if node in points:
                        points[node].append((color, data['z']))
                    else:
                        points[node] = [(color, data['z'])]
        for node, colors in points.items():
            colors.sort(key=lambda x: x[1], reverse=True)
            grid[node] = colors[0][0]
        return grid
