"""
MCTS abtract implementation.

"""


from functions import random_choice
import numpy as np


# Select best successor nodes.
def select(tree, tree_idx, f_ucb):
    # Indices of available children nodes.
    children_idx = tree.get_child_idx(tree_idx)

    # Choose the child with highest UCB score.
    children_score = np.array(tree.data.score[children_idx])
    children_N = np.array(tree.data.N[children_idx])
    ucb_score = children_score + f_ucb(tree.data.N[tree_idx], children_N)
    max_idx = np.argmax(ucb_score)

    return children_idx[max_idx]


# Expand the tree by randomly choose a successor node.
def expand(tree, tree_idx, f_actions, f_terminal):
    # Randomly choose a node (action) from the available options.
    available_actions = tree.data.actions_to_try[tree_idx]
    next_state = random_choice(available_actions)

    # Remove that option.
    available_actions.remove(next_state)
    tree.data.at[tree_idx, 'actions_to_try'] = available_actions

    total_history = tree.get_state_history(tree_idx)
    total_history.append(next_state)

    # If not terminal, add selected node to the search tree.
    if not f_terminal(total_history):
        tree_idx = tree.add_leaf(tree_idx,
                                state=next_state, 
                                actions_to_try=f_actions(total_history),
                                score=0,
                                N=0)
    return tree_idx


# Roll out to evaluate the selected nodes.
def roll_out(tree, tree_idx, f_actions, f_sampler, f_terminal):
    rollout_history = tree.get_state_history(tree_idx)

    # While not terminated, keep randomly choose an action.
    while True:
        available_rollout_actions = f_actions(rollout_history)
        if len(available_rollout_actions) == 0:
            break
        next_rollout_state = f_sampler(available_rollout_actions)

        # If met terminal condition, stop roll out.
        if f_terminal(rollout_history + [next_rollout_state]):
            break
        else:
            # Else append to rollout history and continue.
            rollout_history.append(next_rollout_state)

    return rollout_history


# Backpropagate to evaluate the selected nodes.
def backprop(tree, tree_idx, current_score, _):
    # Node sequences to be backpropagated.
    backtrace = tree.get_seq_indice(tree_idx)

    # Retrieve current info.
    old_score = np.array(tree.data.score[backtrace])
    n = np.array(tree.data.N[backtrace])

    # Update new score.
    tree.data.at[backtrace, 'score'] = np.divide((np.multiply(old_score, n) + current_score), (n + 1))
    tree.data.at[backtrace, 'N'] = n + 1


# MCTS growTree.
def growTree(tree, tree_idx, f_score, f_actions, f_terminal, f_ucb, f_sampler, f_backprop=backprop):
    # If the node has been fully explored, select the best successor node.
    while not tree.data.actions_to_try[tree_idx] and len(tree.get_child_idx(tree_idx)) != 0:
        tree_idx = select(tree, tree_idx, f_ucb)

    # If reach a non-fully explored node, select a successor node to expand.
    if tree.data.actions_to_try[tree_idx]:
        tree_idx = expand(tree, tree_idx, f_actions, f_terminal)

    # Perform rollout until the end of the episode.
    rollout_history = roll_out(tree, tree_idx, f_actions, f_sampler, f_terminal)

    # Backpropagate to evaluate the selected nodes.
    current_score = f_score(rollout_history)
    f_backprop(tree, tree_idx, current_score, rollout_history)

    return current_score, rollout_history

