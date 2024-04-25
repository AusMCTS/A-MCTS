"""
Tree data structure for MCTS implementation.

"""


import numpy as np
import pandas as pd


class Tree:
    def __init__(self, **kwargs):
        # **kwargs allows __init__ to take multiple arguments in form of dict.
        self.leaf_t = dict()
        self.leaf_t['parent'] = np.nan
        self.leaf_t['depth'] = np.nan
        for key, val in kwargs.items():
            self.leaf_t[key] = val

        # Use dataframe to present each tree.
        self.data = pd.DataFrame.from_dict(self.leaf_t)
        self.data.at[0, 'parent'] = 0
        self.data.at[0, 'depth'] = 0
        self.children = {0: list()}         # List of successor leaves.

    def get_child_idx(self, idx):
        return self.children[idx]

    def add_leaf(self, idx_parent, **kwargs):
        # Add a successor leaf.
        leaf_added = self.leaf_t
        leaf_added['parent'] = idx_parent
        leaf_added['depth'] = self.data.at[idx_parent, 'depth'] + 1
        for key, val in kwargs.items():
            leaf_added[key] = val

        self.data = pd.concat([self.data, pd.DataFrame([leaf_added])], ignore_index=True)
        idx_added = self.data.shape[0] - 1
        self.children[idx_added] = list()
        self.children[idx_parent].append(idx_added)

        return idx_added

    def get_leaf(self, idx):
        return self.data.iloc[[idx]]

    def get_seq_indice(self, idx, root_idx=0):
        # Get a trajectory sequence walking backward to root.
        indice_seq = list()
        while True:
            indice_seq.insert(0, int(idx))
            if idx == root_idx:
                break
            idx = self.data.parent[idx]
        return indice_seq

    def get_state_history(self, idx, root_idx=0):
        return self.data.state[self.get_seq_indice(idx, root_idx)].tolist()
