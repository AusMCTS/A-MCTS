"""
Graph environment constructors.

"""


from matplotlib import pyplot as plt
from prm import minDistance
from random import randint
import numpy as np


def get_colour_code(n_agents):
    color = []
    n = n_agents
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    return color


class Graph:
    def __init__(self, xL, xH, yL, yH, radius, obsMask=None):
        self.xL = xL
        self.xH = xH
        self.yL = yL
        self.yH = yH
        self.radius = radius
        if obsMask is not None:
            self.obsMask = obsMask
        else:
            self.obsMask = np.array([[0,0],[0,0]])
        self.nodes = dict()         # Nodes' latitude and longtitude.
        self.edges = dict()         # Edges' adjacency matrix.
        self.edges_list = list()
        self.rewards = dict()       # Edges' rewards mask.
        self.node_counter = 0

    def add_node(self, x_coor, y_coor):
        self.nodes[self.node_counter] = np.array([x_coor, y_coor])
        self.node_counter += 1

    def add_edge(self, start, end, cost, n_rewards):
        # Undirected graph.
        if start not in self.edges.keys():
            self.edges[start] = dict()
            self.rewards[start] = dict()
        self.edges_list.append((start, end))

        # Length cost.
        self.edges[start][end] = cost
        # Rewards logical mask.
        self.rewards[start][end] = np.zeros(n_rewards).astype(int)
    
    def reset_reward(self, n_rewards):
        for i in self.edges.keys():
            for j in self.edges[i].keys():
                self.rewards[i][j] = np.zeros(n_rewards).astype(int)

    def add_reward(self, rewards):
        # An edge covers a reward if it lies within a fixed radius from that reward.
        for i in self.edges.keys():
            for j in self.edges[i].keys():
                for k, reward in enumerate(rewards):
                    if np.all(~np.isnan(reward)):
                        minDist = minDistance(self.nodes[i], self.nodes[j], reward)
                        self.rewards[i][j][k] = (minDist <= self.radius)

    def find_edge(self, ref):
        idx = []
        for i in range(len(self.edges_list)):
            if self.edges_list[i][0] == self.edges_list[int(ref)][1]:
                idx.append(i)
        return idx

    def find_neighbours(self, ref):
        # Return reachable nodes from the reference node.
        return [key for key in self.edges[ref].keys()]

    def find_node(self, coordinates):
        # Find node based on its coordinates.
        for idx, coor in self.nodes.items():
            if (coor == np.array(coordinates)).all():
                return idx

    def find_coordinates(self, node_idx):
        # Find node coordinates based on its index.
        return self.nodes[node_idx]

    def evaluate_edge_cost(self, idx):
        # Get edge cost.
        start = self.edges_list[int(idx)][0]
        end = self.edges_list[int(idx)][1]
        return self.edges[start][end]

    def evaluate_edge_reward(self, idx):
        # Get edge reward mask.
        start = self.edges_list[int(idx)][0]
        end = self.edges_list[int(idx)][1]
        return self.rewards[start][end]

    def evaluate_traj_cost(self, edge_history):
        # Get length cost of the whole trajectories.
        cost = 0
        if type(edge_history) is dict:
            for key in edge_history.keys():
                if len(edge_history[key]) > 0:
                    for i in range(len(edge_history[key])):
                        if ~np.isnan(edge_history[key][i]):
                            cost += self.evaluate_edge_cost(edge_history[key][i])
        else:
            for i in range(len(edge_history)):
                if ~np.isnan(edge_history[i]):
                    cost += self.evaluate_edge_cost(edge_history[i])
        return cost

    def evaluate_traj_reward(self, edge_history):
        # Get reward of the whole trajectories.
        reward = [0]
        if type(edge_history) is dict:
            for key in edge_history.keys():
                if len(edge_history[key]) > 0:
                    for i in range(len(edge_history[key])):
                        if ~np.isnan(edge_history[key][i]):
                            reward |= self.evaluate_edge_reward(edge_history[key][i])
        else:
            for i in range(len(edge_history)-1):
                if ~np.isnan(edge_history[i]):
                    reward |= self.evaluate_edge_reward(edge_history[i])

        return reward

    def draw_nodes(self):
        # Initialise the plot.
        if not plt.fignum_exists(1):
            # plt.figure(figsize=(7, 7))
            fig, ax = plt.subplots(figsize=(7,7))
            plt.xticks(np.arange(self.xL, self.xH+0.2, step=0.2))
            plt.yticks(np.arange(self.yL, self.yH+0.2, step=0.2))
            plt.xlim(self.xL, self.xH)
            plt.ylim(self.yL, self.yH)

        for node in self.nodes.values():
            plt.scatter(node[0], node[1], s=5, fc='blue')
        # for i, txt in enumerate([key for key in self.nodes.keys()]):
        #     plt.annotate(txt, (self.nodes[i][0], self.nodes[i][1]))
        plt.draw()

    def draw_agents(self, agents):
        # Initialise the plot.
        if not plt.fignum_exists(1):
            # plt.figure(figsize=(7, 7))
            fig, ax = plt.subplots(figsize=(7,7))
            plt.xticks(np.arange(self.xL, self.xH+0.2, step=0.2))
            plt.yticks(np.arange(self.yL, self.yH+0.2, step=0.2))
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.xlim(self.xL, self.xH)
            plt.ylim(self.yL, self.yH)

        for agent in agents:
            plt.scatter(agent[0], agent[1], marker="*", s=250, fc='orange')
        plt.draw()

    def draw_rewards(self, rewards):
        # Initialise the plot.
        if not plt.fignum_exists(1):
            #plt.figure(figsize=(7, 7))
            fig, ax = plt.subplots(figsize=(7,7))
            plt.xticks(np.arange(self.xL, self.xH+0.2, step=0.2))
            plt.yticks(np.arange(self.yL, self.yH+0.2, step=0.2))
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.xlim(self.xL, self.xH)
            plt.ylim(self.yL, self.yH)
        i = 0
        for reward in rewards:
            circle = plt.Circle((reward[0], reward[1]), self.radius, fc='none', ec='lime')
            plt.gca().add_patch(circle)
            plt.scatter(reward[0], reward[1], s=5, fc='red')
            # plt.annotate(i, (reward[0], reward[1]))
            i += 1
        plt.draw()

    def draw_edges(self):
        # Initialise the plot.
        if not plt.fignum_exists(1):
            # plt.figure(figsize=(7, 7))
            fig, ax = plt.subplots(figsize=(7,7))
            plt.xticks(np.arange(self.xL, self.xH+0.2, step=0.2))
            plt.yticks(np.arange(self.yL, self.yH+0.2, step=0.2))
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.xlim(self.xL-0.05, self.xH+0.05)
            plt.ylim(self.yL-0.05, self.yH+0.05)

        # Plot the generated nodes, edges.
        drawn_edges = list()
        for i in self.edges.keys():
            for j in self.edges[i].keys():
                if tuple((j, i)) not in drawn_edges:
                    drawn_edges.append(tuple((i, j)))
                    x_index = [self.nodes[i][0], self.nodes[j][0]]
                    y_index = [self.nodes[i][1], self.nodes[j][1]]
                    plt.plot(x_index, y_index, 'navajowhite', linewidth=0.2)
        plt.draw()

    def draw_obs(self):
        # Size of the obstacles mask.
        nX = self.obsMask.shape[1]
        nY = self.obsMask.shape[0]
        # Size of the graph environment.
        width = self.xH - self.xL
        height = self.yH - self.yL
        # Size of each rectangle obstacle block.
        obs_width = width/nX
        obs_height = height/nY
        # Indices of obstacle blocks.
        obs_x = np.where(self.obsMask == 1)[1].tolist()
        obs_y = np.where(self.obsMask == 1)[0].tolist()

        # Initialise the plot.
        if not plt.fignum_exists(1):
            # plt.figure(figsize=(7, 7))
            fig, ax = plt.subplots(figsize=(7,7))
            plt.xticks(np.arange(self.xL, self.xH+0.2, step=0.2))
            plt.yticks(np.arange(self.yL, self.yH+0.2, step=0.2))
            plt.xlim(self.xL, self.xH)
            plt.ylim(self.yL, self.yH)

        # Plot the obstacle blocks.
        for i in range(len(obs_x)):
            # Convert the indices to coordinates.
            x_index = self.xL + obs_x[i]*obs_width
            y_index = self.yH - (obs_y[i]+1)*obs_height
            rectangle = plt.Rectangle((x_index, y_index), obs_width, obs_height, fc='grey', ec='black')
            plt.gca().add_patch(rectangle)

    def draw_results(self, best_joint_path, covered_rewards, uncovered_rewards, prefix=None, title=None, video=False):
        # Initialise the plot.
        if not plt.fignum_exists(1):
            # fig = plt.figure(figsize=(7, 7))
            fig, ax = plt.subplots(figsize=(7,7))
            plt.xticks(np.arange(self.xL, self.xH+0.2, step=0.2))
            plt.yticks(np.arange(self.yL, self.yH+0.2, step=0.2))
            plt.xlim(self.xL, self.xH)
            plt.ylim(self.yL, self.yH)
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        else:
            fig = plt.gcf()
            plt.cla()
            # fig, ax = plt.subplots(figsize=(7,7))
            plt.xticks(np.arange(self.xL, self.xH+0.2, step=0.2))
            plt.yticks(np.arange(self.yL, self.yH+0.2, step=0.2))
            # Turn off tick labels
            # ax.set_yticklabels([])
            # ax.set_xticklabels([])
            plt.xlim(self.xL, self.xH)
            plt.ylim(self.yL, self.yH)
        if title != None:
            plt.title("{}".format(title))

        self.draw_obs()
        # self.draw_edges()
        # self.draw_nodes()

        # Plot the covered (lime) and uncovered rewards (red).
        for reward in covered_rewards:
            circle = plt.Circle((reward[0], reward[1]), self.radius, fc='none', ec='lime')
            plt.gca().add_patch(circle)

        for reward in uncovered_rewards:
            circle = plt.Circle((reward[0], reward[1]), self.radius, fc='none', ec='red')
            plt.gca().add_patch(circle)

        # Plot the best paths by each agent.
        colour_code = get_colour_code(len(best_joint_path))
        for key in best_joint_path.keys():
            for i in range(len(best_joint_path[key])):
                if ~np.isnan(best_joint_path[key][i]):
                    start_node = self.edges_list[int(best_joint_path[key][i])][0]
                    end_node = self.edges_list[int(best_joint_path[key][i])][1]
                    start_node_coor = self.find_coordinates(start_node)
                    end_node_coor = self.find_coordinates(end_node)
                    x_index = [start_node_coor[0], end_node_coor[0]]
                    y_index = [start_node_coor[1], end_node_coor[1]]
                    plt.plot(x_index, y_index, colour_code[key], linewidth=0.7)
            dx = (end_node_coor[0] - start_node_coor[0])/100
            dy = (end_node_coor[1] - start_node_coor[1])/100
            plt.arrow(end_node_coor[0], end_node_coor[1], dx, dy, shape='full', lw=2, length_includes_head=True, head_width=.02, color=colour_code[key])

        plt.draw()
        if video:
            fig.savefig("../Data/gif-{}/{}.png".format(prefix, title))
        else:
            plt.pause(0.01)
