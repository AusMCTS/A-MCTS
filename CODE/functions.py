"""
Helper functions for MCTS.

Reference:
Dec-MCTS: Decentralized planning for multi-robot active perception
[https://doi.org/10.1177/0278364918755924]
"""


import numpy as np
import math
from collections import deque
from tree import Tree
from copy import deepcopy


# Random choice among all available options.
def random_choice(vector:list):
    idx = np.random.randint(low=0, high=len(vector), size=1)[0]
    choice = vector[idx]
    return choice


# Random choice with probabilities.
def random_choice_bias(choices:list, pb:list):
    unnorm_pb = pb
    cuml_pb = np.cumsum(unnorm_pb)
    random_draw = cuml_pb[-1] * np.random.rand()
    idx = np.where(random_draw < cuml_pb)[0][0]
    return choices[idx]


# Backpropagation for decentralised MCTS.
def backprop_decmcts(tree:Tree, tree_idx:int, root_idx:int, current_score:float, current_path:list, gamma:float, current_epoch:int):
    # Node sequences to be backpropagated.
    backtrace = tree.get_seq_indice(tree_idx, root_idx)

    # Update new score and number of time node being selected using the dec-mcst formula.
    old_score = np.array(tree.data.score[backtrace].copy())
    last_backprop = np.array(tree.data.last_backdrop_epoch[backtrace].copy())
    current_n = np.array(tree.data.N[backtrace].copy())

    discount_factor = np.power(gamma, (current_epoch - last_backprop))
    discount_n = np.multiply(discount_factor, current_n)
    new_accumulative_score = np.multiply(discount_n, old_score) + current_score
    new_n = discount_n + 1

    tree.data.loc[backtrace, 'score'] = np.divide(new_accumulative_score, new_n)
    tree.data.loc[backtrace, 'N'] = new_n

    # Update best score, best path and current epoch.
    to_replace = list()
    for i in backtrace:
        if current_score > tree.data.loc[i, 'best_rollout_score']:
            to_replace.append(i)
            tree.data.at[i, 'best_rollout_path'] = current_path
    tree.data.loc[to_replace, 'best_rollout_score'] = current_score
    tree.data.loc[backtrace, 'last_backdrop_epoch'] = current_epoch


# Update the probabilities of choosing action sequences using regret matching.
def updateRegretMatchingDistribution(num_iterations:int, rob, f_payoff):
    # Set up the cumlative regret dict for each agent
    active_robots = list()
    for index in rob.distribution.keys():
        if len(rob.distribution[index].path[0]) > 0:
            active_robots.append(index)
    cumulative_regrets = dict()
    for rob_idx in active_robots:
        cumulative_regrets[rob_idx] = np.zeros(len(rob.distribution[rob_idx].prob))

    # Best joit policy and global payoff.
    best_joint_policy = dict()
    best_payoff = 0

    # Compute the strategies using regret maching
    for _ in range(num_iterations):
        action_sequences = dict()
        # Sample action sequences for every agents
        for index in active_robots:
            action_sequences[index] = random_choice_bias(rob.distribution[index]['path'].copy().tolist(),
                                                    rob.distribution[index]['prob'].copy().tolist())
        # Calculate the global payoff
        actual_payoff = f_payoff(action_sequences)
        if actual_payoff > best_payoff:
            best_payoff = actual_payoff
            best_joint_policy = deepcopy(action_sequences)

        # Update the cumulative regrets for each action sequence of each robot
        for index in active_robots:
            # Copy values for calculation
            what_if_actions = deepcopy(action_sequences)
            other_actions = rob.distribution[index]['path'].copy().tolist()
            # Regrets equal difference between what-if action and actual action
            for i in range(len(other_actions)):
                what_if_actions.update({index: other_actions[i]})
                cumulative_regrets[index][i] += f_payoff(what_if_actions) - actual_payoff
            # Update regret-matching strategy
            pos_cumulative_regrets = np.maximum(0, cumulative_regrets[index])
            if sum(pos_cumulative_regrets) > 0:
                rob.distribution[index].prob = pos_cumulative_regrets / sum(pos_cumulative_regrets)
            else:
                rob.distribution[index].prob = np.full(shape=len(rob.distribution[index].prob), fill_value=1/len(rob.distribution[index].prob))

    return best_joint_policy


"""
Greedy search - add each action sequence that maximises the intermediate payoff M*N.
"""
def greedy_search(rob, f_payoff):
    active_robots = list()
    for index in rob.distribution.keys():
        if len(rob.distribution[index].path[0]) > 0:
            active_robots.append(index)
    # Best joit policy and global payoff.
    best_joint_policy = dict()

    # Loop throughh each robot sequentially and choose the action sequence that maximises the intermediate payoff.
    action_sequences = dict()
    for index in active_robots:
        best_payoff = -1
        for i in range(len(rob.distribution[index].path)):
            action_sequences[index] = rob.distribution[index].path[i].copy()
            intermediate_payoff = f_payoff(action_sequences)
            if intermediate_payoff > best_payoff:
                best_payoff = intermediate_payoff
                best_joint_policy[index] = rob.distribution[index].path[i].copy()
                rob.distribution[index].at[i, 'prob'] = 1
        action_sequences[index] = best_joint_policy[index].copy()

    return best_joint_policy


"""
Exhaustive search through the whole search space M^N.
"""
def exhaustive_search(robots, active_robots, f_payoff):
    # Get the number of agents and the number of action sequences.
    num_agents = len(active_robots)
    num_actions = 0
    for index in active_robots:
        if len(robots[index].distribution[index].prob) > num_actions:
            num_actions = len(robots[index].distribution[index].prob)
    action_sequences = dict()

    num_iterations = pow(num_actions, num_agents)

    # Best joit policy and global payoff.
    best_joint_policy = dict()
    best_payoff = 0
    j = 0

    # Compute every possible combinations.
    for i in range(num_iterations):
        action_sequences.clear()
        for index in active_robots:
            action_idx = math.floor(i / pow(num_actions, num_agents - j - 1) % num_actions)
            if action_idx < len(robots[index].distribution[index].prob):
                action_sequences[index] = robots[index].distribution[index].path[action_idx].copy()
            j = (j + 1) % num_agents

        payoff = f_payoff(action_sequences)
        # Return the best combination.
        if payoff > best_payoff:
            best_payoff = payoff
            best_joint_policy = action_sequences.copy()

    return best_joint_policy

