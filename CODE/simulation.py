"""
Simulation of Decentralised MCTS on multi-agents task.

"""


import numpy as np
import pandas as pd
import math
import argparse
import csv
import sys
import os
import time
from graph import Graph
from agent import Discounted_Robot, Attrition_Robot
from functions import *
from graph_helper import *
from mcts import growTree
from progressbar import progressbar
from datetime import datetime
from copy import deepcopy


def create_robot(n_agents, G, mode):
    robots = list()
    for i in range(n_agents):
        initial_actions = []
        for j in range(len(G.edges_list)):
            if G.edges_list[j][0] == i:
                initial_actions.append(j)
        if mode == "Dec" or mode == "Global" or mode == "Reset":
            robots.append(Discounted_Robot(np.nan, initial_actions, agents[i], i, n_agents, gamma))
        elif mode == "RM" or mode == "Greedy":
            robots.append(Attrition_Robot(np.nan, initial_actions, agents[i], i, n_agents, gamma))
        else:
            print("Undefined Mode")
            sys.exit()
    return robots, initial_actions


def exporting_results(directory, config, trial, rollout_path, reward_per_round, time_per_round, files_name):
    np.savetxt("{}/{}-performance.csv".format(directory, files_name), reward_per_round, delimiter=",")
    np.savetxt("{}/{}-time.csv".format(directory, files_name), time_per_round, delimiter=",")
    np.savetxt("{}/{}-planning_score.csv".format(directory, files_name), score_per_iter, delimiter=",")
    with open("{}/{}-rollout-C{}-T{}.csv".format(directory, files_name, config, trial+1), "w", newline='') as f:
        write = csv.writer(f)
        for path in rollout_path:
            write.writerow([path])


# Joint path histories of all robots.
def f_joint(rob_index:int, edge_history:list, edge_history_other_robots:dict):
    edge_history_other_robots.update({rob_index: edge_history})
    return edge_history_other_robots


# Get reward mask of the edge between the current node to all immediate successor nodes.
def evaluate_immediate_actions(G:Graph, available_actions:list):
    pb = list()
    for i in range(len(available_actions)):
        pb.append(1 + sum(G.evaluate_edge_reward(available_actions[i])))
    return pb


def planning(f_reward, f_actions, f_terminal, f_ucb, f_sampler, planning_time:int, robots:list, active_robots:list, N_components:int, N_comms_every:int):
    # Planning start.
    start = time.time()

    # Clear info of others from previous cycle.
    if adapt_type == "Greedy":
        for active_index in active_robots:
            robots[active_index].reset_distribution()
    elif adapt_type == "RM":
        for active_index in active_robots:
            robots[active_index].reset_status()

    # for current_iter in progressbar(range(planning_time), redirect_stdout=True):
    for current_iter in range(planning_time):
        joint_rollout = dict()

        # Compress the tree into product distribution.
        if ((current_iter+1) % N_comms_every) == 0:
            for active_index in active_robots:
                robots[active_index].compress_tree(N_components)

            if adapt_type == "RM" or adapt_type == "Greedy":
                # Communication to other robots.
                for active_index in active_robots:
                    for other_index in active_robots:
                        if other_index != active_index:
                            if np.random.rand() <= comm_rate:
                                robots[other_index].distribution[active_index] = robots[active_index].distribution[active_index].copy()
                                if adapt_type == "RM":
                                    robots[other_index].status[active_index] = current_iter

                # Find the join policy.
                for active_index in active_robots:
                    robots[active_index].best_response.clear()
                    f_payoff = lambda chosen_actions: f_reward(chosen_actions)
                    if adapt_type == "RM":
                        _ = robots[active_index].check_status(current_iter, threshold*N_comms_every)
                        robots[active_index].best_response = updateRegretMatchingDistribution(100, robots[active_index], f_payoff)
                    elif adapt_type == "Greedy":
                        robots[active_index].best_response = greedy_search(robots[active_index], f_payoff)

                # Synchronise the join policy.
                if adapt_type == "RM":
                    synchronise_policy = dict()
                    best_score = 0
                    # Find the best join policy computed by each agent.
                    for active_index in active_robots:
                        score = f_reward(robots[active_index].best_response)
                        if score > best_score:
                            synchronise_policy = deepcopy(robots[active_index].best_response)
                            best_score = score
                    # Synchronise the join policy among agents.
                    for active_index in active_robots:
                        robots[active_index].best_response = deepcopy(synchronise_policy)

        for active_index in active_robots:
            # Simulate other robots actions.
            if adapt_type == "Dec" or adapt_type == "Global" or adapt_type == "Reset":
                robots[active_index].simulate_other_robots()

            # Utility is the joint reward minus other robots' reward.
            if adapt_type == "Dec" or adapt_type == "Reset":
                other_robots_reward = f_reward(robots[active_index].best_response)
                f_score = lambda edge_history: f_reward(f_joint(active_index, edge_history, robots[active_index].best_response)) - other_robots_reward
            # Global utility.
            else:
                f_score = lambda edge_history: f_reward(f_joint(active_index, edge_history, robots[active_index].best_response))

            # Re-initialise backprop to take the current iteration and gamma.
            f_backprop = lambda tree, tree_idx, current_score, current_rollout: backprop_decmcts(tree, tree_idx, robots[active_index].root_idx, current_score, current_rollout, gamma, current_iter)

            # Grow the tree search.
            _, joint_rollout[active_index] = growTree(robots[active_index].tree, robots[active_index].root_idx, f_score, f_actions, f_terminal, f_ucb, f_sampler, f_backprop)

            # Update the distribution and communication to other robots.
            if adapt_type == "Dec" or adapt_type == "Global" or adapt_type == "Reset":
                robots[active_index].updateDistribution(alpha, gamma_sched(current_iter))
                for other_index in active_robots:
                    if other_index != active_index:
                        if np.random.rand() <= comm_rate:
                            robots[other_index].distribution[active_index] = robots[active_index].distribution[active_index].copy()

        # Keep track of planning score.
        if args.save:
            score_per_iter[current_iter+planning_time*action_order][N_trial*config + trial] = f_reward(joint_rollout)
    # Planning ends.
    end = time.time()

    return robots, end - start


# Find and return the immediate action and tree index with highest probabilites.
def execute_action_update_state(robots:list, active_robots:list, joint_execution_path:dict):
    for active_index in active_robots:
        action, new_root_idx  = robots[active_index].get_state_action()
        joint_excecution_path[active_index].append(action)
        # Set the new root.
        robots[active_index].set_root_idx(new_root_idx)
    return joint_execution_path


if __name__ == '__main__':

    # Parsing the input options.
    parser = argparse.ArgumentParser(description="Simulate Dec-MCTS on multi-agent tasks")
    parser.add_argument("-s", "--save", help="Save performance", action='store_true', default=False)
    parser.add_argument("-a", "--adapt", help="Type of adaptation to environment changes", required=True, choices=['Dec', 'Global', 'Reset', 'RM', 'Greedy'])
    parser.add_argument("-f", "--folder", help="Folder name")
    parser.add_argument("-v", "--verbose", help="Print details", action='store_true', default=False)
    parser.add_argument("-p", "--params", help="Parameter testing", nargs="+")
    args = parser.parse_args()

    # System parameters.
    xL = -2                                           # min of the x-axis.
    xH = 2                                            # max of the x-axis.
    yL = -2                                           # min of the y-axis.
    yH = 2                                            # max of the y-axis.
    obsMask = np.array([[0]])                         # the obstacles mask.

    # Checking input arguments.
    adapt_type = args.adapt

    # Fixed parameters.
    N_configs = 10
    N_trial = 2                                       # number of trial experiments per configurations.
    reward_radius = 0.05                              # reward disk radius.
    N_comms_every = 50                                # period of tree compression.
    c_p = 0.4                                         # Exploration parameter
    gamma = 0.9                                       # discounting factor for d-ucb.
    alpha = 1.0                                       # step size for variational update.
    gamma_sched = lambda N_iter: max(math.pow(0.95, N_iter), 0.001) # cooling schedule for gamma.

    # Shared variational parameters.
    N_components = 10                                 # number of components to be communicated.
    max_actions = 9                                   # action budget.
    planning_time = 500                               # planning time budget for each action.
    fail_rate = 1                                     # Amount of failure occurences.
    fail_intensity = 0.5                              # Percentage of agents per failure.
    comm_rate = 1.0                                   # Inter-agent communication success rate.
    n_rewards = 200                                   # number of rewards.
    threshold = 0                                     # Tolerance threshold between agent loss and comm loss.

    if args.params != None:
        param_name = args.params[0]
        param_value = args.params[1]
        if param_name == "action":
            max_actions = int(param_value)
        elif param_name == "planning":
            planning_time = int(param_value)
        elif param_name == "rate":
            fail_rate = int(param_value)
        elif param_name == "intensity":
            fail_intensity = float(param_value)
        elif param_name == "comp":
            N_components = int(param_value)
        elif param_name == "comm":
            comm_rate = float(param_value)
        elif param_name == "reward":
            n_rewards = int(param_value)
        elif param_name == "threshold":
            threshold = int(param_value)

    try:
        if args.save:
            if args.folder != None:
                directory = os.path.join("../Data/", args.folder)
            else:
                now = datetime.now()
                directory = os.path.join("../Data/", now.strftime("%Y-%m-%d-%H-%M"))
            if not os.path.isdir(directory):
                try:
                    os.mkdir(directory)
                except:
                    pass
            # Reward/rollout score and planning time matrix.
            reward_per_action = np.array(np.zeros([max_actions, N_trial*N_configs]))
            time_per_action = np.array(np.zeros([max_actions, N_trial*N_configs]))
            # File names.
            files_name = "{}".format(adapt_type)
            # Planning score.
            score_per_iter = np.array(np.zeros([max_actions*planning_time, N_trial*N_configs]))

        # Initialise a seed for reproductivity.
        rng_rate = np.random.default_rng(12345)
        rng_intenisty = np.random.default_rng(12345)
        rng_num = np.random.default_rng(12345)

        # Start simulations.
        for config in range(N_configs):
            # Generate graph environment.
            G = Graph(xL, xH, yL, yH, reward_radius, obsMask)
            path = "../Data/Config_{}".format(config)
            G, agents, rewards, nodes, n_agents, n_nodes, _ = import_graph(G, path)
            if rewards.shape[0] != n_rewards:
                rewards = modify_number_rewards(G, n_rewards, rewards, seed=config)
            G.reset_reward(n_rewards)
            G.add_reward(rewards)
            n_agents = 20
            if args.params != None:
                param_name = args.params[0]
                param_value = args.params[1]
                if param_name == "agent":
                    n_agents = int(param_value)

            # Global objective function.
            f_reward = lambda edge_history: sum(G.evaluate_traj_reward(edge_history))/n_rewards
            # Available functions are listed of successor nodes.
            f_actions = lambda edge_history: G.find_edge(edge_history[-1])
            # Terminal condition is when the travelling budget expires.
            f_terminal = lambda edge_history: len(edge_history) > max_actions+1
            # UCB functions.
            f_ucb = lambda Np, Nc: 2 * c_p * np.sqrt(np.divide(np.log(Np), Nc))
            # Pick next action based on its immediate reward (greedy random).
            f_sampler = lambda available_actions: random_choice_bias(available_actions, evaluate_immediate_actions(G, available_actions))

            if args.verbose:
                print("Config {}/{}".format(config+1, N_configs))
                print("Generating the graph environment with {} action budget, {} planning iterations, {} fail times, and {} fail intensity.".format(max_actions, planning_time, fail_rate, fail_intensity))

            for trial in range(N_trial):
                print("Trial {}/{}: Starting with {} {} agents".format(trial+1, N_trial, adapt_type, n_agents))
                if args.save:
                    rollout_path = list()

                # Create robots.
                robots, initial_actions = create_robot(n_agents, G, adapt_type)
                active_robots = np.array(range(0, n_agents), dtype=int)
                joint_excecution_path = dict()
                for idx in active_robots:
                    joint_excecution_path[idx] = list()

                # When churns occur.
                churn_at = np.sort(rng_rate.choice(range(max_actions), size=fail_rate, replace=False))
                if args.params != None:
                    if args.params[0] == "churn_at":
                        churn_at = np.array(args.params[1:], dtype=int)
                        fail_rate = len(churn_at)
                # Number of agents fail each time.
                total_remove = math.ceil(n_agents*fail_intensity)
                max_per_remove = math.ceil(n_agents*fail_intensity/fail_rate)
                min_per_remove = math.floor(n_agents*fail_intensity/fail_rate)
                if fail_rate == 1:
                    num_to_remove = rng_num.integers(min_per_remove, max_per_remove, size=fail_rate-1, endpoint=True)
                    num_to_remove = np.append(num_to_remove, total_remove-np.sum(num_to_remove))
                else:
                    num_to_remove = rng_num.integers(min_per_remove, max_per_remove, size=fail_rate, endpoint=True)

                # Randomly remove agents.
                for action_order in range(max_actions):
                    if action_order in churn_at:
                    # if action_order+1 == math.ceil(max_actions/2):
                        idx_to_remove = rng_intenisty.choice(range(len(active_robots)), size=num_to_remove[0], replace=False)
                        if args.verbose:
                            print("Remove {} agents {} at action {}/{}".format(num_to_remove[0], active_robots[idx_to_remove], action_order+1, max_actions))
                        num_to_remove = np.delete(num_to_remove, 0)
                        for idx in idx_to_remove:
                            joint_excecution_path[active_robots[idx]] = list()
                        active_robots = np.delete(active_robots, idx_to_remove)

                        if adapt_type == "Reset" and action_order != 0:
                            for idx in active_robots:
                                robots[idx].set_active_agents(deepcopy(active_robots))
                                robots[idx].reset_tree(np.nan, f_actions(joint_excecution_path[idx]))

                    # Each agent increamentally grows its own search tree using a simulated model of the graph.
                    robots, elapsed_time = planning(f_reward, f_actions, f_terminal, f_ucb, f_sampler, planning_time, robots, active_robots, N_components, N_comms_every)

                    # The action sequence or tree branch with highest probabilities is executed.
                    joint_excecution_path = execute_action_update_state(robots, active_robots, joint_excecution_path)

                    # Evaluate the executed the joint action sequences with the real graph.
                    joint_excecution_score = f_reward(joint_excecution_path)

                    # Save values for analytics.
                    if args.save:
                        reward_per_action[action_order][N_trial*config + trial] = joint_excecution_score
                        time_per_action[action_order][N_trial*config + trial] = elapsed_time
                        rollout_path.append(deepcopy(joint_excecution_path))
                    if args.verbose:
                        print("Reward: {} - Budget Used: {}/{}".format(joint_excecution_score, action_order+1, max_actions))

                if args.save:
                    exporting_results(directory, config, trial, rollout_path, reward_per_action, time_per_action, files_name)

    except KeyboardInterrupt:
        print("Simulation discarding. Exporting results so far.")
        if args.save:
            exporting_results(directory, config, trial, rollout_path, reward_per_action, time_per_action, files_name)
