'''Part 3'''

import json
import time
from itertools import product
from inspect import signature
from queue import PriorityQueue
import numpy as np

from graph import ARCGraph
from image import Image
from utils import plot_sample, plot_some

TSP = '../input/arc-prize-2024/arc-agi_training_solutions.json'
ESP = '../input/arc-prize-2024/arc-agi_evaluation_solutions.json'

with open(TSP, 'r', encoding='utf-8') as file:
    truth_t = json.load(file)
with open(ESP, 'r', encoding='utf-8') as file:
    truth_e = json.load(file)
truth = truth_t | truth_e

class PriorityItem:
    """
    storing information of a node in the search tree,
    the sequence of filters and transformations is stored as data,
    instead of the abstracted graphs themselves
    """
    def __init__(self, data, abstraction, priority, depth=0, wide=False):
        self.data = data
        self.abstraction = abstraction
        self.priority = priority
        self.depth = depth
        self.wide = wide

    def __lt__(self, other):
        if self.wide:
            if self.depth == other.depth:
                if self.priority == other.priority:
                    return len(json.dumps(self.data)) < len(json.dumps(other.data))
                return self.priority < other.priority
            return self.depth < other.depth

        if self.priority == other.priority:
            if self.depth == other.depth:
                return len(json.dumps(self.data)) < len(json.dumps(other.data))
            return self.depth < other.depth
        return self.priority < other.priority

class Task:
    '''Graph -> Image -> (Task)'''
    def __init__(self, name, data):
        """
        contains all information related to an ARC task
        """
        self.task_id = name
        self.data = data
        self.test = False
        self.solved = False
        self.all_colors = set()
        self.bgc = 0
        self.start_time = 0

        # input/output images
        self.train_inputs = []
        self.train_outputs = []
        self.test_inputs = []
        self.test_outputs = []

        # abstracted graphs from input/output images:
        # dictionaries of ARCGraphs, where the keys are the abstraction names and
        # values are lists of ARCGraphs with the abstraction name for all inputs/outputs
        self.abstracted_input_graphs = {}
        self.abstracted_output_graphs = {}
        self.abstracted_input_graphs_original = {}
        self.abstracted_output_graphs_original = {}

        # metadata tracking
        self.total_nodes_explored = 0
        self.total_unique_frontier_nodes = 0

        # attributes used for search
        self.shared_frontier = None  # whether to use shared or abstraction frontiers
        self.constrained = None  # whether to do constraint acquisition or not
        self.time_limit = None  # time limit for search
        self.verbose = None  # print search progress
        self.wide = None  # depth vs score priority
        self.abstraction = None  # which type of abstraction the search is currently working with
        self.static_objects_for_insertion = {}  # static objects for the "insert" transformation
        self.object_sizes = {}  # object sizes to use for filters
        self.object_degrees = {}  # object degrees to use for filters
        self.transformation_ops = {}  # a dictionary of transformation operations
        self.frontier_hash = {}  # used for checking if a resulting image is already found
        self.tabu_list = {}  # temporarily disabling expanding frontier for a specific abstraction
        self.tabu_list_waiting = {}  # list of nodes to be added back to frontier
        self.current_best_scores = {}  # tracking the current best score for each abstraction
        self.current_best_score = float("inf")  # the current best score
        self.current_best_apply_call = None  # apply call that produces the current best solution
        self.current_best_abstraction = None  # abstraction that produces the current best solution
        self.frontier = None  # a priority queue of frontier nodes to be expanded

        self.base_filter = {'filters': ['filter_by_nothing'], 'filter_params': [{}]}

        self.load_task(data)

    def load_task(self, data):
        '''load the task'''
        self.all_colors = data['all_colors']
        self.bgc = data['bgc']

        for i, data_pair in enumerate(data['train']):
            self.train_inputs.append(
                Image(self, grid=data_pair['input'], name=f'{self.task_id}_{i+1}_train_in'))
            self.train_outputs.append(
                Image(self, grid=data_pair['output'], name=f'{self.task_id}_{i+1}_train_out'))
        for i, data_pair in enumerate(data['test']):
            self.test_inputs.append(
                Image(self, grid=data_pair['input'], name=f'{self.task_id}_{i+1}_test_in'))
            if self.task_id in truth:
                self.test_outputs.append(
                    Image(self, grid=truth[self.task_id][i], name=f'{self.task_id}_{i+1}_test_out'))
            else:
                self.test = True

    def solve(self, time_limit=100, shared_frontier=True, constrained=True,
              verbose=False, wide=False, stats=None):
        """
        solve for a solution, yes
        :param shared_frontier: whether the search uses a shared frontier between abstractions
        :param time_limit: maximum time allowed for search in seconds
        :param constrained: whether constraint acquisition is used
        """
        self.time_limit = time_limit
        self.shared_frontier = shared_frontier
        self.constrained = constrained
        self.verbose = verbose
        self.wide = wide

        # frontier for search, each item is a PriorityItem object
        if shared_frontier:
            self.frontier = PriorityQueue()
        else:
            self.frontier = {}  # maintain a separate frontier for each abstraction

        print(f"Solving: {self.task_id}")

        self.start_time = time.time()

        # initialize frontier
        stop_search = self.initialize_frontier()

        # main search loop
        while not stop_search:
            if self.shared_frontier:
                stop_search = self.search_shared_frontier()
            else:
                stop_search = self.search_separate_frontier()

        predictions = []
        no_error = False

        # apply to test images
        for i, input_image in enumerate(self.test_inputs):
            solution = [[0, 0], [0, 0]]
            if self.current_best_abstraction is None:
                predictions.append(solution)
                print("No solution at all!")
                continue

            abstracted_graph = getattr(input_image, Image.abstraction_getters[
                self.current_best_abstraction])()
            for call in self.current_best_apply_call:
                abstracted_graph.apply(**call)

            sol_grid = abstracted_graph.get_pixel_grid()

            # check if the solution gets the correct test output
            error = 0
            if not self.test:
                error = np.sum(np.not_equal(sol_grid, self.test_outputs[i].grid))
                if error == 0:
                    no_error = True
                    print('The solution is correct!')
                else:
                    print(f"The best solution predicted {error} "
                          f"out of {sol_grid.size} pixels incorrectly")
                if self.solved or no_error:
                    print('Steps:', len(self.current_best_apply_call),
                          '| Sol:', self.current_best_apply_call)

            if self.solved:
                solution = sol_grid.tolist()
            predictions.append(solution)

        print()
        if stats is not None and self.task_id in stats:
            seconds = time.time() - self.start_time
            stats[self.task_id] += f' {seconds:6.2f} {str(self.solved):5} {no_error}'

        return predictions

    def timeout(self):
        '''time out the task'''
        return time.time() - self.start_time > self.time_limit

    def complete(self, message='none'):
        '''display results'''
        messages = {'solved': 'Solution found!',
                    'timeout': 'Solution not found due to time limit reached!',
                    'empty': 'Solution not found due to empty search space!',
                    'none': 'Solution not found due to unknown reasons!'}
        print(messages[message])
        print(f'Abstraction: {self.current_best_abstraction} | '
              f'Best score: {self.current_best_score} | '
              f'Runtime: {time.time() - self.start_time:g}')
        return True

    def update_score(self, frontier_node):
        '''track stats'''
        if frontier_node.priority < self.current_best_score:
            self.current_best_score = frontier_node.priority
            self.current_best_apply_call = frontier_node.data
            self.current_best_abstraction = self.abstraction
            if frontier_node.priority == 0:
                self.solved = True

    def initialize_frontier(self):
        """
        initialize frontier
        :return: True if a solution is found during initialization
        or time limit has been reached, False otherwise
        """
        if self.verbose:
            print("Initializing frontier")

        for abstraction in Image.abstractions:
            # It's time to stop
            if self.timeout():
                return self.complete('timeout')

            # specify the current abstraction
            self.abstraction = abstraction

            # initialize individual frontiers if abstractions do not share one
            if not self.shared_frontier:
                self.frontier[abstraction] = PriorityQueue()

            # initialize additional attributes used in search
            self.current_best_scores[abstraction] = float("inf")
            self.tabu_list[abstraction] = 0
            self.tabu_list_waiting[abstraction] = []
            self.frontier_hash[abstraction] = set()

            # produce the abstracted graphs for input output images using the current abstraction
            # these are the 'original' abstracted graphs that will not be updated
            self.abstracted_input_graphs_original[abstraction] = [
                getattr(input_image, Image.abstraction_getters[
                    abstraction])() for input_image in self.train_inputs]
            self.abstracted_output_graphs_original[abstraction] = [
                getattr(output_image, Image.abstraction_getters[
                    abstraction])() for output_image in self.train_outputs]

            # keep a list of transformation ops that we modify
            # based on constraint acquisition results
            self.transformation_ops[abstraction] = ARCGraph.transformation_ops[abstraction]

            # constraint acquisition (global)
            if self.constrained:
                self.constraints_acquisition_global()

            # look for static objects to insert
            # if insert transformation is not pruned by constraint acquisition
            self.get_static_inserted_objects(abstraction)

            # get the list of object sizes and degrees
            self.get_static_object_attributes(abstraction)

            # initiate frontier with dummy node and expand it
            # (representing doing nothing to the input image)
            frontier_node = PriorityItem([], abstraction, float("inf"), float("inf"), self.wide)

            sol_node = self.expand_frontier(frontier_node)
            if sol_node is not None:
                self.update_score(sol_node)
                return self.complete('solved')

            if self.shared_frontier:
                if self.frontier.empty():
                    continue
                frontier_node = self.frontier.queue[0]
            else:
                if self.frontier[abstraction].empty():
                    continue
                frontier_node = self.frontier[abstraction].queue[0]

            if frontier_node.priority < self.current_best_scores[abstraction]:
                self.current_best_scores[abstraction] = frontier_node.priority

            self.update_score(frontier_node)

            if self.timeout():
                return self.complete('timeout')

        print('Init complete')
        return False

    def search_shared_frontier(self):
        """
        perform one iteration of search for a solution using a shared frontier
        :return: True if a solution is found or time limit has been reached, False otherwise
        """
        if self.frontier.empty():
            if self.verbose:
                print('Tabu reset!')
            for abstraction in self.tabu_list:
                # put the nodes from the waiting list back into frontier
                for node in self.tabu_list_waiting[abstraction]:
                    self.frontier.put(node)
                self.tabu_list_waiting[abstraction] = []
                self.tabu_list[abstraction] = 0

        if self.frontier.empty():
            return self.complete('empty')

        frontier_node = self.frontier.get(False)

        # if this abstraction is on tabu list, collect the nodes
        if self.tabu_list[frontier_node.abstraction] > 0:
            self.tabu_list_waiting[frontier_node.abstraction].append(frontier_node)
            return False
        # if this abstraction is not on tabu list, but has a worse score than before,
        # explore it and put it on tabu list
        if frontier_node.priority >= self.current_best_scores[frontier_node.abstraction]:
            self.tabu_list[frontier_node.abstraction] = 1
        else:
            self.current_best_scores[frontier_node.abstraction] = frontier_node.priority

        self.abstraction = frontier_node.abstraction
        if self.verbose:
            print(f"Exploring frontier node with score {frontier_node.priority} "
                  f"at depth {frontier_node.depth} "
                  f"with abstraction {self.abstraction}")

        self.update_score(frontier_node)
        sol_node = self.expand_frontier(frontier_node)
        if sol_node is not None:
            self.update_score(sol_node)
            return self.complete('solved')

        if self.timeout():
            return self.complete('timeout')

        return False

    def search_separate_frontier(self):
        """
        perform one iteration of search for a solution using a multiple frontiers
        :return: True if a solution is found or time limit has been reached, False otherwise
        """
        proceed = False

        for abstraction in Image.abstractions:
            self.abstraction = abstraction

            if self.frontier[self.abstraction].empty():
                continue
            # if this abstraction is on tabu list, explore something else
            if self.tabu_list[self.abstraction] > 0:
                self.tabu_list[self.abstraction] = 0
                proceed = True
                continue

            proceed = True
            frontier_node = self.frontier[self.abstraction].get()

            # if this abstraction is not on tabu list, but has a worse score than before,
            # explore it and put it on tabu list
            if frontier_node.priority >= self.current_best_scores[self.abstraction]:
                self.tabu_list[self.abstraction] = 1
            else:
                self.current_best_scores[self.abstraction] = frontier_node.priority

            if self.verbose:
                print(f"Exploring frontier node with score {frontier_node.priority} "
                      f"at depth {len(frontier_node.data)} "
                      f"with abstraction {self.abstraction}")

            self.update_score(frontier_node)
            sol_node = self.expand_frontier(frontier_node)
            if sol_node is not None:
                self.update_score(sol_node)
                return self.complete('solved')

            if self.timeout():
                return self.complete('timeout')

        if not proceed:
            return self.complete('empty')

        return False

    def expand_frontier(self, frontier_node):
        """
        expand one frontier node
        """
        if self.verbose:
            print(f"Expanding frontier node with abstraction {self.abstraction}")
        # up to date abstracted graphs
        self.abstracted_input_graphs[self.abstraction] = []
        for abstracted_input_graph in self.abstracted_input_graphs_original[self.abstraction]:
            abstracted_input = abstracted_input_graph.copy()
            for apply_call in frontier_node.data:
                # apply the transformation to the abstracted graph
                abstracted_input.apply(**apply_call)
            self.abstracted_input_graphs[self.abstraction].append(abstracted_input)

        apply_calls = self.get_candidate_transformations()

        if self.verbose:
            print(f"Number of new candidate nodes: {len(apply_calls)}")
        added_nodes = 0

        for apply_call in apply_calls:
            if self.timeout():
                break
            self.total_nodes_explored += 1
            cumulated_apply_calls = frontier_node.data.copy()
            cumulated_apply_calls.append(apply_call)
            # Apply the new step to the precalculated graphs
            apply_call_score = self.calculate_score(apply_call)

            if apply_call_score == -1:
                continue
            if apply_call_score >= self.current_best_scores[self.abstraction]:
                continue

            added_nodes += 1
            priority_item = PriorityItem(cumulated_apply_calls,
                                         self.abstraction,
                                         apply_call_score,
                                         len(cumulated_apply_calls),
                                         self.wide)
            # Done
            if apply_call_score == 0:
                return priority_item

            if self.abstraction == 'nag':
                continue

            if self.shared_frontier:
                self.frontier.put(priority_item)
            else:
                self.frontier[self.abstraction].put(priority_item)

        if self.verbose:
            print(f"Number of new nodes added to Frontier: {added_nodes}")
        self.total_unique_frontier_nodes += added_nodes
        return None

    def get_candidate_transformations(self):
        """
        generate candidate transformations, return list of full operations candidates
        """
        filters = self.get_candidate_filters()
        all_apply_calls = []
        if self.constrained:
            transformation_ops = self.get_transformations()
        else:
            transformation_ops = self.transformation_ops[self.abstraction]

        # Many-way combinatorial madness!
        # This is the thing to address!
        for transform_op in transformation_ops:
            if self.timeout():
                break
            sig = signature(getattr(ARCGraph, transform_op))
            params = [x for x in sig.parameters.keys() if x not in ('self', 'node')]
            for filter_call in filters:
                if self.timeout():
                    break
                generated_params = self.generate_parameters(sig, filter_call)
                for item in product(*generated_params):
                    param_vals = {}
                    for param, value in zip(params, item):
                        param_vals[param] = value
                    # dont need deep copy here since we are not modifying existing entries
                    apply_call = {"abstraction": self.abstraction}
                    apply_call |= filter_call.copy()
                    apply_call["transformation"] = [transform_op]
                    apply_call["transformation_params"] = [param_vals]
                    all_apply_calls.append(apply_call)
        return all_apply_calls

    def get_candidate_filters(self):
        """
        return list of candidate filters
        """
        if self.abstraction == 'nag':
            return [self.base_filter]

        all_apply_filter_calls = []  # final list of filter calls
        filtered_nodes_all = []  # use this list to avoid filters that return the same set of nodes

        for filter_op in ARCGraph.filter_ops:
            # first, we generate all possible values for each parameter
            sig = signature(getattr(ARCGraph, filter_op))
            params = [x for x in sig.parameters.keys() if x not in ('self', 'node')]
            generated_params = []
            for param in sig.parameters.values():
                param_name = param.name
                if param_name in ("self", "node"):
                    continue

                if param_name == "color":
                    generated_params.append(["same"] + [c for c in self.all_colors])
                elif param_name == "size":
                    generated_params.append(["min", "max", "odd"] +
                        [s for s in self.object_sizes[self.abstraction]])
                elif param_name == "degree":
                    generated_params.append([d for d in self.object_degrees[self.abstraction]])
                elif param.annotation == bool:
                    generated_params.append([True, False])
                elif param_name in ARCGraph.param_values:
                    generated_params.append([p for p in ARCGraph.param_values[param_name]])

            # then, we combine all generated values to get all possible combinations of parameters
            for item in product(*generated_params):
                param_vals = {}
                for param, value in zip(params, item):
                    param_vals[param] = value
                candidate_filter = {"filters": [filter_op],
                                    "filter_params": [param_vals]}

                #  do not include if the filter result in empty set of nodes
                # (this will be the majority of filters)
                filtered_nodes = []
                for abstracted_input_graph in self.abstracted_input_graphs[self.abstraction]:
                    filtered_nodes_i = []
                    for node in abstracted_input_graph.graph.nodes():
                        if abstracted_input_graph.apply_filters(node, **candidate_filter):
                            filtered_nodes_i.append(node)
                    filtered_nodes.extend(filtered_nodes_i)
                filtered_nodes.sort()
                # does not result in empty or duplicate set of nodes
                if len(filtered_nodes) > 0 and filtered_nodes not in filtered_nodes_all:
                    filtered_nodes_all.append(filtered_nodes)
                    all_apply_filter_calls.append(candidate_filter)

        if self.verbose:
            print(f"Applicable filters: {len(all_apply_filter_calls)}")
        return all_apply_filter_calls

    def generate_parameters(self, transform_sig, filter_call):
        """
        given filter nodes and a transformation, generate parameters for the transformation
        example: given filters for red nodes and move,
        return [up, down, left, right, get_relative_position(red nodes, blue neighbors), ...]
        :param filter_call: the specific apply filter call to get the nodes
        :param all_calls: all apply filter calls, this is used to generate the dynamic parameters
        :param transform_sig: signature for a transformation
        :return: parameters to be passed to the transformation
        """
        generated_params = []
        for param in transform_sig.parameters.values():
            param_name = param.name
            if param_name in ("self", "node"):
                continue
            # first we generate the static values
            if param_name in ("color", "fill_color", "border_color"):
                all_possible_values = ["same"] + [c for c in self.all_colors]
            elif param_name == "object_id":
                all_possible_values = [-1] + [i for i in range(len(
                    self.static_objects_for_insertion[self.abstraction]))]
            elif param_name in ARCGraph.param_values:
                all_possible_values = [p for p in ARCGraph.param_values[param_name]]
            elif param.annotation == bool:
                all_possible_values = [True, False]
            else:
                all_possible_values = []

            # then we add dynamic values for parameters
            if param_name in ARCGraph.dynamic_parameters:
                all_possible_values = self.generate_dynamic_parameters(
                    all_possible_values, filter_call)

            generated_params.append(all_possible_values)
        return generated_params

    def generate_dynamic_parameters(self, all_possible_values, filter_call):
        """
        generate the dynamic part
        """
        param_nodes_all = []
        # the filters that define the dynamic parameter values,
        # has their own parameters generated_filter_params
        for param_binding_op in ARCGraph.param_binding_ops:
            sig = signature(getattr(ARCGraph, param_binding_op))
            generated_filter_params = []
            for deep_param in sig.parameters.values():
                deep_param_name = deep_param.name
                if deep_param_name in ("self", "node"):
                    continue
                if deep_param_name == "color":
                    generated_filter_params.append(["same"] + [c for c in self.all_colors])
                elif deep_param_name == "size":
                    generated_filter_params.append(["min", "max", "odd"] +
                        [s for s in self.object_sizes[self.abstraction]])
                elif deep_param_name == "degree":
                    generated_filter_params.append(
                        [d for d in self.object_degrees[self.abstraction]])
                elif deep_param.annotation == bool:
                    generated_filter_params.append([True, False])
                elif deep_param_name in ARCGraph.param_values:
                    generated_filter_params.append(
                        [p for p in ARCGraph.param_values[deep_param_name]])

            for item in product(*generated_filter_params):
                param_vals = {}
                for i, param in enumerate(list(sig.parameters)[2:]):
                    param_vals[sig.parameters[param].name] = item[i]

                param_nodes = []
                for abstracted_input_graph in self.abstracted_input_graphs[self.abstraction]:
                    param_nodes_i = []
                    for node in abstracted_input_graph.graph.nodes():
                        if abstracted_input_graph.apply_filters(node, **filter_call):
                            param_node = getattr(
                                abstracted_input_graph, param_binding_op)(node, **param_vals)
                            if param_node is not None:
                                param_nodes_i.append(param_node)
                    param_nodes.extend(param_nodes_i)
                param_nodes.sort()
                if len(param_nodes) > 0 and param_nodes not in param_nodes_all:
                    param_nodes_all.append(param_nodes)
                    all_possible_values.append({"filters": [param_binding_op],
                                                "filter_params": [param_vals]})
        return all_possible_values

    def get_search_score(self, focused='', index=0):
        '''estimate combinatorial complexity of the task (experimental)'''
        self.time_limit = 300
        self.start_time = time.time()
        all_combos = {}
        for abstraction in Image.abstractions:
            self.abstraction = abstraction
            self.abstracted_input_graphs_original[abstraction] = [
                getattr(input_image, Image.abstraction_getters[
                    abstraction])() for input_image in self.train_inputs]
            self.abstracted_output_graphs_original[abstraction] = [
                getattr(output_image, Image.abstraction_getters[
                    abstraction])() for output_image in self.train_outputs]
            self.abstracted_input_graphs[
                abstraction] = self.abstracted_input_graphs_original[abstraction]
            self.abstracted_output_graphs[
                abstraction] = self.abstracted_output_graphs_original[abstraction]
            self.transformation_ops[abstraction] = ARCGraph.transformation_ops[abstraction]
            self.constraints_acquisition_global()
            self.get_static_inserted_objects(abstraction)
            self.get_static_object_attributes(abstraction)

            combos = self.get_candidate_transformations()
            all_combos[abstraction] = len(combos)

            def plot_6(images, a, call):
                for ti, to in images:
                    graph = getattr(ti, Image.abstraction_getters[a])()
                    graph.apply(**call)
                    solution = graph.get_pixel_grid()
                    diff_io = np.not_equal(ti.grid, to.grid)
                    diff_is = np.not_equal(ti.grid, solution)
                    diff_so = np.not_equal(solution, to.grid)
                    plot_some([ti.grid, diff_io, to.grid, diff_is, solution, diff_so],
                              ['Input', 'Diff IO', 'Output', 'Diff IS', 'Sol', 'Diff SO'])

            if abstraction == focused:
                print(abstraction, index, combos[index])
                print('Train:', len(self.train_inputs))
                plot_6(zip(self.train_inputs, self.train_outputs), abstraction, combos[index])
                print('Test:', len(self.test_inputs))
                plot_6(zip(self.test_inputs, self.test_outputs), abstraction, combos[index])

        return all_combos

    def calculate_score(self, apply_call):
        """
        calculate the total score across all training examples for a given apply call,
        hash the apply call by using it as a key to the cache
        """
        all_recon = []
        for abstracted_input_graph in self.abstracted_input_graphs[self.abstraction]:
            abstracted_input = abstracted_input_graph.copy()
            abstracted_input.apply(**apply_call)
            reconstructed = abstracted_input.get_pixel_grid()
            all_recon.append(reconstructed.tolist())

        hash_string = json.dumps(all_recon, separators=('', ''))
        if hash_string in self.frontier_hash[self.abstraction]:
            return -1
        self.frontier_hash[self.abstraction].add(hash_string)

        score = 0
        for i, output in enumerate(self.train_outputs):
            score += np.sum(np.not_equal(all_recon[i], output.grid))

        return score

    def check_sol(self, sol, mode='all', plot=False):
        '''check solutions independently (experimental)'''
        abstraction = sol[0]['abstraction']
        # Required for some operations
        self.abstracted_input_graphs_original[abstraction] = [
            getattr(input_image, Image.abstraction_getters[
                abstraction])() for input_image in self.train_inputs]
        self.abstracted_output_graphs_original[abstraction] = [
            getattr(output_image, Image.abstraction_getters[
                abstraction])() for output_image in self.train_outputs]
        self.transformation_ops[abstraction] = ARCGraph.transformation_ops[abstraction]
        self.get_static_inserted_objects(abstraction)
        self.get_static_object_attributes(abstraction)

        def check(images):
            for input_image, output_image in images:
                graph = getattr(input_image, Image.abstraction_getters[abstraction])()
                for call in sol:
                    graph.apply(**call)
                solution = graph.get_pixel_grid()

                error = np.sum(np.not_equal(solution, output_image.grid))
                if error == 0:
                    print('The solution is correct!')
                else:
                    print(f'The solution predicted {error} '
                          f'out of {solution.size} pixels incorrectly')
                if plot:
                    plot_sample(input_image.grid, output_image.grid, solution)

        if mode != 'test':
            print('Train:', len(self.train_inputs))
            check(zip(self.train_inputs, self.train_outputs))
        if mode != 'train':
            print('Test:', len(self.test_inputs))
            check(zip(self.test_inputs, self.test_outputs))

    def constraints_acquisition_global(self):
        """
        find the constraints that all nodes in the instance must follow
        """
        no_movements = True
        for i, image in enumerate(self.train_inputs):
            for node, data in image.graph.nodes(data=True):
                if (data["color"] != image.background_color
                    and self.train_outputs[i].graph.nodes[
                        node]["color"] == image.background_color) or (
                            data["color"] == image.background_color
                            and self.train_outputs[i].graph.nodes[
                                node]["color"] != image.background_color):
                    no_movements = False
        no_new_objects = True
        for i, abstracted_output_graph in enumerate(
            self.abstracted_output_graphs_original[self.abstraction]):
            abstracted_input_nodes = self.abstracted_input_graphs_original[
                self.abstraction][i].graph.nodes()
            for abstracted_node, data in abstracted_output_graph.graph.nodes(data=True):
                if abstracted_node not in abstracted_input_nodes:
                    no_new_objects = False
                    break
        if no_movements:
            excluded = ["move", "fill_rectangle", "add_border", "insert"]
            self.transformation_ops[self.abstraction] = [
                t for t in self.transformation_ops[self.abstraction] if t not in excluded]
        elif no_new_objects:
            self.transformation_ops[self.abstraction] = [
                t for t in self.transformation_ops[self.abstraction] if t not in ["insert"]]

    def constraints_acquisition_local(self):
        """
        given an apply_filter_call, find the set of constraints that
        the nodes returned by the apply_filter_call must satisfy.
        these are called local constraints as they apply only to the nodes
        that satisfy the filter
        """
        list_of_rules = ["color_equal", "position_equal", "size_equal"]
        found_constraints = []
        for rule in list_of_rules:
            all_satisfied = True
            for index in range(len(self.train_inputs)):
                satisfied = self.constraints_param_satisfaction(rule, index)
                all_satisfied = all_satisfied and satisfied
            if all_satisfied:
                found_constraints.append(rule)
        return found_constraints

    def constraints_param_satisfaction(self, rule, index):
        """
        given condition and rule, first generate the sequence using the condition
        then transform the sequence into the expected format for the constraint
        :param condition: {'filters': ['filter_nodes_by_color'],
          'filter_params': [{'color': 0, 'exclude': True}]}
        :param rule: "rule_name"
        :param index: training instance index
        """
        input_abs = self.abstracted_input_graphs[self.abstraction][index]
        output_abs = self.abstracted_output_graphs_original[self.abstraction][index]

        input_nodes = []
        for node in input_abs.graph.nodes():
            input_nodes.append(node)

        output_nodes = []
        for node in output_abs.graph.nodes():
            output_nodes.append(node)

        if rule == "color_equal":
            input_sequence = [input_abs.graph.nodes[node]["color"] for node in input_nodes]
            output_sequence = [output_abs.graph.nodes[node]["color"] for node in output_nodes]

        elif rule == "position_equal":
            input_sequence = []
            output_sequence = []
            for node in input_nodes:
                input_sequence.extend(subnode for subnode in input_abs.graph.nodes[node]["nodes"])
            for node in output_nodes:
                output_sequence.extend(subnode for subnode in output_abs.graph.nodes[node]["nodes"])

        elif rule == "size_equal":
            input_sequence = [input_abs.graph.nodes[node]["size"] for node in input_nodes]
            output_sequence = [output_abs.graph.nodes[node]["size"] for node in output_nodes]

        input_sequence.sort()
        output_sequence.sort()

        return input_sequence == output_sequence

    def get_transformations(self):
        """
        given a set of constraints that must be satisfied,
        return a set of transformations that do not violate them
        """
        constraints = self.constraints_acquisition_local()
        transformations = self.transformation_ops[self.abstraction]
        excluded = []
        for constraint in constraints:
            if constraint == "color_equal":
                excluded.extend(["update_color"])
            elif constraint == "position_equal":
                excluded.extend(["move", "extend"])
            elif constraint == "size_equal":
                excluded.extend(["extend"])
        transformations = [t for t in transformations if t not in excluded]
        return transformations

    def get_static_inserted_objects(self, abstraction):
        """
        populate self.static_objects_for_insertion,
        which contains all static objects detected in the images
        """
        self.static_objects_for_insertion[abstraction] = []
        if 'insert' not in self.transformation_ops[abstraction]:
            return
        existing_objects = []
        for i, abstracted_output_graph in enumerate(
            self.abstracted_output_graphs_original[abstraction]):
            abstracted_input_nodes = self.abstracted_input_graphs_original[
                abstraction][i].graph.nodes()
            for abstracted_node, data in abstracted_output_graph.graph.nodes(data=True):
                if abstracted_node not in abstracted_input_nodes:
                    new_object = data.copy()
                    min_x = min(subnode[1] for subnode in new_object["nodes"])
                    min_y = min(subnode[0] for subnode in new_object["nodes"])
                    adjusted_subnodes = []
                    for subnode in new_object["nodes"]:
                        adjusted_subnodes.append((subnode[0] - min_y, subnode[1] - min_x))
                    adjusted_subnodes.sort()
                    if adjusted_subnodes not in existing_objects:
                        existing_objects.append(adjusted_subnodes)
                        self.static_objects_for_insertion[abstraction].append(new_object)

    def get_static_object_attributes(self, abstraction):
        """
        populate self.object_sizes and self.object_degrees,
        which contains all sizes and degrees of existing objects
        """
        self.object_sizes[abstraction] = set()
        self.object_degrees[abstraction] = set()
        for abs_graph in self.abstracted_input_graphs_original[abstraction]:
            for _, size in abs_graph.graph.nodes(data="size"):
                self.object_sizes[abstraction].add(size)
            for _, degree in abs_graph.graph.degree():
                self.object_degrees[abstraction].add(degree)
