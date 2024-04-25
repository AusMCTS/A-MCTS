"""
Simulation of Centralised MCTS on multi-agents task.

"""


import numpy as np
import argparse
import csv
import os
from progressbar import progressbar
from graph import Graph
from tree import Tree
from agent import Central_Robot
from functions import *
from graph_helper import *
from mcts import growTree
from datetime import datetime
from copy import deepcopy


def create_robot(n_agents:int, G:Graph):
    robots = list()

    for i in range(n_agents):
        initial_actions = []
        # Get available actions at the staring positions.
        for j in range(len(G.edges_list)):
            if G.edges_list[j][0] == i:
                initial_actions.append(j)
        robots.append(Central_Robot(np.nan, deepcopy(initial_actions), agents, i, n_agents))

    return robots


def exporting_results(directory, config, trial, rollout_path, reward_per_round, files_name):
    np.savetxt("{}/{}-performance.csv".format(directory, files_name), reward_per_round, delimiter=",")
    np.savetxt("{}/{}-planning_score.csv".format(directory, files_name), score_per_iter, delimiter=",")
    with open("{}/{}-rollout-C{}-T{}.csv".format(directory, files_name, config, trial+1), "w", newline='') as f:
        write = csv.writer(f)
        for path in rollout_path:
            write.writerow([path])


# Joint path histories of all robots.
def f_joint(edge_history:list, active_robots):
    edge_history_robots = dict()
    for i in active_robots:
        edge_history_robots[i] = list()
    for i in range(1, len(edge_history)):
        edge_history_robots[active_robots[(i - 1) % len(active_robots)]].append(edge_history[i])
    return edge_history_robots


# Terminal condition is when the travelling budget of the agent expires.
def f_terminal(edge_history:list):
    edge_history_robots = f_joint(edge_history, active_robots)
    robot_history = edge_history_robots[active_robots[-1]]
    return True if len(robot_history) > remaining_budget else False


# Get reward mask of the edge between the current node to all immediate successor nodes.
def evaluate_immediate_actions(G:Graph, available_actions:list):
    pb = list()
    for i in range(len(available_actions)):
        pb.append(1 + sum(G.evaluate_edge_reward(available_actions[i])))
    return pb


if __name__ == '__main__':

    # Parsing the input options.
    parser = argparse.ArgumentParser(description="Simulate Cen-MCTS on multi-agent tasks")
    parser.add_argument("-s", "--save", help="Save performance", action='store_true', default=False)
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

    # Fixed parameters.
    N_trial = 1                                       # number of trial experiments per configurations.
    N_configs = 1                                    # number of configurations to test.
    reward_radius = 0.05                              # reward disk radius.
    c_p = 0.4                                         # Exploration parameter                                

    # Shared variational parameters.
    max_actions = 9                                   # action budget.
    planning_time = 500                               # planning time budget for each action.
    fail_rate = 1                                     # Amount of failure occurences.
    fail_intensity = 0.5                              # Percentage of agents per failure.
    n_rewards = 200                                   # number of rewards.

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
        elif param_name == "reward":
            n_rewards = int(param_value)

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
            reward_per_round = np.array(np.zeros([max_actions, N_configs*N_trial]))
            # File names.
            files_name = "Central"
            # Planning score.
            score_per_iter = np.array(np.zeros([planning_time, N_configs*N_trial]))

        # Initialise a seed for reproductivity.
        rng_rate = np.random.default_rng(12345)
        rng_intenisty = np.random.default_rng(12345)
        rng_num = np.random.default_rng(12345)

        for config in range(N_configs):
            # Generate graph environment.
            G = Graph(xL, xH, yL, yH, reward_radius, obsMask)
            path = "../Data/Config_{}".format(config)
            G, agents, rewards, nodes, _, n_nodes, _ = import_graph(G, path)
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
            f_actions = lambda edge_history: robots[active_robots[len(edge_history)-1]].get_actions() if len(edge_history) <= len(active_robots) else G.find_edge(edge_history[-len(active_robots)])
            # UCB functions.
            f_ucb = lambda Np, Nc: 2 * c_p * np.sqrt(np.divide(np.log(Np), Nc))
            # Pick next action based on its immediate reward (greedy random).
            f_sampler = lambda available_actions: random_choice_bias(available_actions, evaluate_immediate_actions(G, available_actions))
            # Global objective function.
            f_score = lambda edge_history: sum(G.evaluate_traj_reward(f_joint(edge_history, active_robots)))/n_rewards

            if args.verbose:
                print("Config {}/{}".format(config+1, N_configs))
                print("Generating the graph environment with {} action budget, {} planning iterations, {} fail times, and {} fail intensity.".format(max_actions, planning_time, fail_rate, fail_intensity))

            # Start simulation.
            for trial in range(N_trial):
                print("Trial {}/{}".format(trial+1, N_trial))
                if args.save:
                    rollout_path = list()

                # Create robots.
                robots = create_robot(n_agents, G)
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

                remaining_budget = max_actions
                for action_order in range(max_actions):
                    # Randomly remove agents.
                    if action_order in churn_at:
                        idx_to_remove = rng_intenisty.choice(range(len(active_robots)), size=num_to_remove[0], replace=False)
                        if args.verbose:
                            print("Remove {} agents {} at action {}/{}".format(num_to_remove[0], active_robots[idx_to_remove], action_order+1, max_actions))
                        num_to_remove = np.delete(num_to_remove, 0)
                        for idx in idx_to_remove:
                            joint_excecution_path[active_robots[idx]] = list()
                        active_robots = np.delete(active_robots, idx_to_remove)

                    # Single tree for every agents.
                    tree = Tree(state=[robots[active_robots[0]].get_state()],
                                actions_to_try=[robots[active_robots[0]].get_actions()],
                                score=0,
                                N=0,
                                best_rollout_score=-np.inf, 
                                best_rollout_path=[list()])

                    # Planning start.
                    for current_iter in progressbar(range(planning_time), redirect_stdout=True):
                    # for current_iter in range(planning_time):
                        # Grow the tree search.
                        rollout_score, rollout_history = growTree(tree, 0, f_score, f_actions, f_terminal, f_ucb, f_sampler)

                        if rollout_score > tree.data.at[0, 'best_rollout_score']:
                            tree.data.at[0, 'best_rollout_score'] = rollout_score
                            tree.data.at[0, 'best_rollout_path'] = f_joint(rollout_history, active_robots)

                        # Keep track of planning score.
                        if args.save:
                            score_per_iter[current_iter][N_trial*config + trial] = rollout_score

                    # Evaluate the executed the joint action sequences with the real graph.
                    for active_index in active_robots:
                        action = tree.data.at[0, 'best_rollout_path'][active_index][0]
                        joint_excecution_path[active_index].append(action)
                        # Set the new actions.
                        initial_actions = G.find_edge(action)
                        robots[active_index].set_actions(initial_actions)

                    remaining_budget -= 1
                    acc_collected_rewards = G.evaluate_traj_reward(joint_excecution_path)
                    joint_excecution_score = sum(acc_collected_rewards)/n_rewards

                    # Save values for analytics.
                    if args.save:
                        reward_per_round[action_order][N_trial*config + trial] = joint_excecution_score
                        rollout_path.append(joint_excecution_path)
                    if args.verbose:
                        print("Score: {}/{}".format(joint_excecution_score, action_order+1))

                    if args.save:
                        exporting_results(directory, config, trial, rollout_path, reward_per_round, files_name)

    except KeyboardInterrupt:
        print("Simulation discarding. Exporting results so far.")
        if args.save:
            exporting_results(directory, config, trial, rollout_path, reward_per_round, files_name)
