# class import
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import distance
import math

# kd-tree KNN
class KDTreeKnn():
    def __init__(self):
        self.index = None
        self.index_built = False
        return
        
    def datafeed(self, X, Z):
        rownum, colnum = X.shape
        self.datasize = rownum
        self.spatial_dim = colnum
        
        if(rownum != Z.shape[0]):
            print("Number of input data doesn't match")
            return

        self.attributeset = Z
        self.attribute_dim = self.attributeset.shape[1]
        self.index = KDTree(X)
        self.index_built = True
    
    # Direct search without considering attributes
    # Return indices of neighbors
    def search_direct(self, q, k=10):
        s_amount = min(k, self.datasize)
        return self.index.query([q], k=s_amount, return_distance=False)[0]
    
    # Search for a wilder group, then generate two sorts and used the average rank
    # Return indices of neighbors
    def search_greedy(self, q, k=10, search_magnitude=2, metric="cosine", set_dist="mean"):
        s_amount = math.floor(k * search_magnitude)
        s_amount = min(s_amount, self.datasize)
        fun_setdist = self._find_mindist if set_dist == "mean" else self._find_mindist
        
        candidates = list(self.index.query([q], k=s_amount, return_distance=False)[0])
        answer_attrs = np.empty((0, self.attribute_dim))
        answer_set = []
        while True:
            # Choose candidate id
            if len(answer_set) == 0:
                selected_index = 0
                selected_id = candidates[0]
            else:
                # list_index-rank on query-dist
                rank_spatial = np.arange(1, len(candidates)+1)
                # list_index-rank on diversity
                setdists = np.array([fun_setdist(self.attributeset[c], answer_attrs) for c in candidates])
                rank_sdist = np.argsort(setdists*(-1)) + 1 # Further one will get higher rank
                aggr_rank = (rank_spatial + rank_sdist)/2
                selected_index = np.argmin(aggr_rank) # Choose the one with highest rank (min one)
                selected_id = candidates[selected_index]
            # Add to answer set, and pop out that candidate
            answer_set.append(selected_id)
            answer_attrs = np.vstack([answer_attrs, self.attributeset[selected_id]])
            candidates = candidates[:selected_index] + candidates[selected_index+1:]
            if len(candidates) == k:
                break
        return candidates
                            
    def _find_mindist(self, q, S, metric="euclidean"):
        q_dists = distance.cdist([q], S, metric=metric)
        return q_dists.min()

    def _find_meandist(self, q, S, metric="euclidean"):
        q_dists = distance.cdist([q], S, metric=metric)
        return q_dists.mean()