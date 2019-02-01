"""
The MIT Licence

Copyright 2018 Paul Sinclair

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Source: https://github.com/pbsinclair42/MCTS

Minor modifications by Islam Elnabarawy for formatting and consistency
"""

import math
import random
import time

from rps3env.classes.game_state import GameState


class TreeNode:
    def __init__(self, state, parent):
        """
        :type state: GameState
        :type parent: TreeNode | None
        """
        self.state = state
        self.isTerminal = state.is_terminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}


class MCTS:
    def __init__(self, time_limit=None, iteration_limit=None, exploration_constant=1 / math.sqrt(2)):
        if time_limit is not None:
            if iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = time_limit
            self.limitType = 'time'
        else:
            if iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iteration_limit
            self.limitType = 'iterations'
        self.explorationConstant = exploration_constant
        self.root = None
        self.global_best_reward = 0

    def search(self, initial_state):
        """
        :type initial_state: GameState
        """
        self.root = TreeNode(initial_state, None)

        if self.limitType == 'time':
            time_limit = time.time() + self.timeLimit / 1000
            while time.time() < time_limit:
                self.execute_round()
        else:
            for i in range(self.searchLimit):
                self.execute_round()

        best_child = self.get_best_child(self.root, 0)
        return self.get_action(self.root, best_child)

    def execute_round(self):
        node = self.select_node(self.root)
        reward = self.rollout(node.state)
        self.backpropagate(node, reward)

    def select_node(self, node):
        """
        :type node: TreeNode
        """
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.get_best_child(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    @staticmethod
    def rollout(state):
        """
        :type state: GameState
        """
        while not state.is_terminal():
            try:
                action = random.choice(state.get_possible_actions())
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(state))
            state = state.take_action(action)
        return state.get_reward()

    @staticmethod
    def expand(node):
        """
        :type node: TreeNode
        """
        actions = node.state.get_possible_actions()
        for action in actions:
            if action not in node.children.keys():
                new_node = TreeNode(node.state.take_action(action), node)
                node.children[action] = new_node
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return new_node

        raise Exception("Should never reach here")

    @staticmethod
    def backpropagate(node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    @staticmethod
    def get_best_child(node, exploration_value):
        best_value = float("-inf")
        best_nodes = []
        for child in node.children.values():
            node_value = child.totalReward / child.numVisits + exploration_value * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)
        return random.choice(best_nodes)

    @staticmethod
    def get_action(root, best_child):
        for action, node in root.children.items():
            if node is best_child:
                return action
