"""
Agent model implementations.

"""


import numpy as np
import pandas as pd
from tree import Tree
from collections import deque
from functions import *


class Base_Robot:
    def set_initial_values(self, initial_position, i, n_agents, forgetting_factor):
        # Best rollout path and score found.
        self.best_score = 0
        self.best_rollout = list()
        # Starting position in the graph.
        self.position = initial_position
        # Number of agents and the robot's index.
        self.index = i
        self.n_agents = n_agents
        self.active_agents = np.array(range(0, n_agents), dtype=int)
        # Forgetting factor value (window size or discounted factor).
        self.forgetting_factor = forgetting_factor
        # Index of the current root node.
        self.root_idx = 0
        # Path histories and probabilities.
        self.distribution = dict()
        for agent in self.active_agents:
            self.distribution[agent] = pd.DataFrame(data={'prob': [1], 'path': [list()], 'tree_idx': [0]})
        # Best respone of the joint policies.
        self.best_response = dict()

    # Update the root node.
    def set_root_idx(self, new_root_idx:int):
        self.root_idx = new_root_idx

    def set_active_agents(self, new_active_agents):
        self.active_agents = new_active_agents

    # Get the state of action.
    def get_current_state(self):
        return self.tree.data.state[self.root_idx]

    # Each agent holds a MCTS tree.
    def set_tree(self, state, action):
        self.tree = Tree(state=state,
                        actions_to_try=[action], 
                        score=0,
                        N=0,
                        last_backdrop_epoch=0, 
                        best_rollout_score=-np.inf, 
                        best_rollout_path=[list()])

    # Restart the search tree.
    def reset_tree(self, state, action):
        self.root_idx = 0
        self.set_tree(state, action)
        self.distribution = dict()
        for agent in self.active_agents:
            self.distribution[agent] = pd.DataFrame(data={'prob': [1], 'path': [list()], 'tree_idx': [0]})

    # Return the actions sequence with highest probability in the probability distribution table.
    def get_actions_sequence(self):
        max_idx = self.distribution[self.index].prob.idxmax()
        return self.distribution[self.index].path[max_idx]

    # Get the immediate action and its index.
    def get_state_action(self):
        # Index of path with highest prob.
        best_path_idx = self.distribution[self.index].prob.idxmax()
        # Tree index of best path.
        max_idx = self.distribution[self.index].tree_idx[best_path_idx]
        # The new root is the index of the first node, the action is the node state.
        new_root_idx = self.tree.get_seq_indice(max_idx, self.root_idx)[1]
        action = self.tree.data.state[new_root_idx]
        return action, new_root_idx

    # Simulate other robot paths based on their probabilities.
    def simulate_other_robots(self):
        self.best_response.clear()
        for i in self.active_agents:
            if i != self.index:
                self.best_response[i] = (random_choice_bias(
                                                self.distribution[i]['path'].copy().tolist(),
                                                self.distribution[i]['prob'].copy().tolist()))

    # Update the probabilities of choosing action sequences.
    def updateDistribution(self, alpha, beta):
        idx = self.distribution[self.index]['tree_idx']
        reward = self.tree.data['score'][idx].copy().tolist()
        prob = self.distribution[self.index]['prob'].copy().tolist()

        overal_expectation = sum(np.multiply(prob, reward))
        log_probability = np.log(prob)
        entropy = sum(np.multiply(prob, log_probability))

        log_gradient = -alpha * ((overal_expectation - reward)/beta + entropy + log_probability)

        self.distribution[self.index].prob = np.exp(log_probability + log_gradient)
        self.distribution[self.index].prob /= sum(self.distribution[self.index].prob)

    # Compress the search tree.
    def compress_tree(self, number_of_components:int):
        # Current state of the root node.
        current_state = self.tree.data.state[self.root_idx]

        # If current state is nan, get the whole tree.
        if np.isnan(current_state):
            sub_tree = self.tree.data
        else:
            # Get the current depth of the root node.
            current_depth = int(self.tree.data.depth[self.root_idx])
            # Get the subtree from the current root node.
            sub_tree = self.tree.data.loc[self.tree.data.apply(lambda row: row["best_rollout_path"][current_depth] == current_state, axis=1)]

        # Number of nodes to be sent.
        immediate_children = sub_tree[sub_tree.index > self.root_idx].copy()
        n = min(number_of_components, immediate_children.shape[0])
        # Sorted the tree.
        sent_children = self.sort_node(immediate_children, n)

        # Tree indices to be sent.
        tree_idx = sent_children.index

        # Redistributed the probabilites based on their scores.
        initial_probability = np.finfo(float).eps + sent_children.score
        initial_probability /= sum(initial_probability)

        # Each compressed tree contains n paths and their probabilities.
        self.distribution[self.index] = pd.DataFrame(data={'prob': initial_probability, 
                                                    'path': self.tree.data.best_rollout_path[tree_idx], 
                                                    'tree_idx': tree_idx}).reset_index(drop=True)

    # Method to sort the nodes to be sent.
    def sort_node(self):
        raise NotImplementedError

    def __str__(self):
        return str(self.tree) + "," + str(self.distribution) + "," + str(self.best_score) + "," + str(self.best_rollout)


class Discounted_Robot(Base_Robot):
    def __init__(self, initial_state, initial_actions, initial_position, i, n_agents, gamma):
        self.set_initial_values(initial_position, i, n_agents, gamma)
        # Each robot holds a MCTS.
        self.set_tree(initial_state, initial_actions)

    def sort_node(self, immediate_children, n):
        # Sorted the tree based on empirical score in descending order.
        return immediate_children.nlargest(n, 'score')


class Attrition_Robot(Base_Robot):
    def __init__(self, initial_state, initial_actions, initial_position, i, n_agents, gamma):
        self.set_initial_values(initial_position, i, n_agents, gamma)
        # Each robot holds a MCTS.
        self.set_tree(initial_state, initial_actions)
        # Keep track of other agents status.
        self.status = dict()
        self.reset_status()

    def sort_node(self, immediate_children, n):
        # Sorted the tree based on number of times node are chosen and score in descending order.
        return immediate_children.nlargest(n, 'score')
    
    def check_status(self, current_iter:int, threshold:int=0):
        # Remove agent info if it has not communicated in a while.
        flag = False
        for i in range(self.n_agents):
            if i != self.index:
                if current_iter - self.status[i] > threshold:
                    self.distribution[i] = pd.DataFrame(data={'prob': [1], 'path': [list()], 'tree_idx': [0]})
                    flag = True
        return flag
    
    def reset_status(self):
        for i in range(self.n_agents):
            self.status[i] = 0
    
    def reset_distribution(self):
        for i in range(self.n_agents):
            if i != self.index:
                self.distribution[i] = pd.DataFrame(data={'prob': [1], 'path': [list()], 'tree_idx': [0]})


class Central_Robot:
    def __init__(self, initial_state, initial_actions, initial_position, i, n_agents):
        # Starting position in the graph.
        self.position = initial_position
        # Number of agents and the robot's index.
        self.index = i
        self.n_agents = n_agents
        # Copied values.
        self.initial_state = initial_state
        self.initial_actions = initial_actions.copy()

    def get_state(self):
        return self.initial_state

    def get_actions(self):
        return self.initial_actions
    
    def set_actions(self, actions):
        self.initial_actions = actions.copy()
