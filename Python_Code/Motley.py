# class import
import numpy as np
from rtree import index
from sklearn.neighbors import KDTree
from scipy.spatial import distance
from itertools import combinations
import math
from random import shuffle

class Motley():
    def __init__(self, threshold=0.2, alpha=0.1, idx_method="rtree", metric_rtree="euclidean"):
        self.threshold = threshold
        self.alpha = alpha
        self.idx_method = idx_method if idx_method == "kdtree" else "rtree"
        # self.metric_rtree = cosine if metric_rtree == "cosine" or metric_rtree == "euclidean" else euclidean
        
    # X means data point in the spatial space
    # Z means corresponding attribute representation
    def datafeed(self, X, Z):
        # rownum: number of points, colnum: spatial dimension
        rownum, colnum = X.shape
        self.datasize = rownum
        self.spatial_dim = colnum
        
        if(rownum != Z.shape[0]):
            print("Number of input data doesn't match")
            return

        self.attributeset = Z
        self.attribute_dim = self.attributeset.shape[1]
        num_attrs = self.attributeset.shape[1]
        a = self.alpha
        
        # Build up index (kd-tree / R-tree)
        if self.idx_method == "kdtree":
            # Note: kd-tree doesn't support additional object linkage
            self.index = KDTree(X)
        else:
            p = index.Property()
            p.dimension = colnum
            self.index = index.Index(properties=p)
            for idx, row in enumerate(X):
                # (1) Index based on row number
                # (2) Store point for the bounding box
                # (3) Store attribute representation as inner object
                self.index.insert(idx, np.append(row, row), Z[idx])
                
        # Weights used for computing MinDiv
        # Number of weights depends on dimension of attribute space.
        self.weight = np.fromfunction(
            lambda self, x: ((a**(x))*(1-a)/(1-a**num_attrs))
            , (1, num_attrs))
    
    # Query a point and find its diversed neighbors
    # aggress is set for next round's search, if k neighbors are not found
    # max_iter is set to avoid whole-document scanning
    def search(self, qs_space, k=10, aggress=5, approach="greedy", max_iter=5):
        # Initial search: nearest (k * aggress) points
        s_amount = int(k * aggress)
        filtered, num_iter = 0, 0 # Neighbors found / Iteration already run
        
        # Initial result contains zero row, so the nearest neighbor is guaranteed
        # to be in the result set.
        res = np.empty((0, self.attributeset.shape[1]))
        ret = []
        
        if self.idx_method == "kdtree":
            # TODO: should stop if s_amount > size of dataset
            while (len(res) < k) and (num_iter < max_iter) and filtered < self.datasize:
                # [filtered:] - Exclude those already exaimed
                s_amount = min(s_amount, self.datasize)
                q_ans = self.index.query([qs_space], k=s_amount, return_distance=False)[0][filtered:]
                # the query returns a list of indices, get point attributes from self.attrs
                for cand in q_ans:
                    # Add if pass the diversity test
                    if self.diversity_check_greedy(res, self.attributeset[cand]):
                        ret.append(cand)
                        res = np.vstack([res, self.attributeset[cand]])
                    if len(res) == k:
                        break
                filtered += len(q_ans)
                # Start the next round
                if len(res) != k:
                    num_iter += 1
                    s_amount = int(s_amount * aggress)
        else: # R-tree
            while (len(res) < k) and (num_iter < max_iter) and filtered < self.datasize:
                q_ans = [x for x in self.index.nearest(np.append(qs_space, qs_space), s_amount, objects=True)][filtered:]
                for cand in q_ans:
                    tmp_attr = cand.object
                    if self.diversity_check_greedy(res, tmp_attr):
                        ret.append(cand.id)
                        res = np.vstack([res, tmp_attr])
                    if len(res) == k:
                        break
                filtered += len(q_ans)
                if len(res) != k:
                    num_iter += 1
                    s_amount = int(s_amount * aggress)
        return ret
    
    def search_buffered(self, qs_space, k=10, buf_size="auto", aggress=5, approach="greedy", max_iter=5):
        leaders = {}
        sdist_newcomer = 0.0
        s_amount = math.ceil(k * aggress)
        filtered, num_iter = 0, 0
        # Suggested buffer size = K in reference paper.
        buf_sz = buf_size if type(buf_size) == int and buf_size > 0 else k
        
        # TODO: should stop if s_amount > size of dataset
        while (len(leaders) < k) and (num_iter < max_iter) and filtered < self.datasize:
            
            print("leader size: %d" % len(leaders))
            print("buffer size: %d" % sum([len(leaders[i]["follower_buffer"]) for i in leaders]))
            
            # [filtered:] - Exclude those already exaimed
            s_amount = min(s_amount, self.datasize)
            if self.idx_method == "kdtree":
                q_ans = self.index.query([qs_space], k=s_amount, return_distance=False)[0][filtered:]
            else: # r-tree
                q_ans = [x for x in self.index.nearest(np.append(qs_space, qs_space), s_amount, objects=True)][filtered:]
            print("Iteration %d" % (num_iter))
            print("should scan through %d data..." % len(q_ans))
            # the query returns a list of indices, get point attributes from self.attrs
            for cand in q_ans:
                # TODO: not yet done!
                group_l = []            # Leaders near to N
                new_leaders_backup = [] # Store new leaders selected
                followers_backup = []   # Store remain followers whose leader is removed
                
                if self.idx_method == "kdtree":
                    cand_entry = {
                            "id": cand,
                            "vector_s": self.index.data[cand],
                            "vector_a": self.attributeset[cand]
                        }
                else:
                    cand_entry = {
                            "id": cand.id,
                            # bbox: (x1, y1, ..., x2, y2, ...)
                            "vector_s": np.array(cand.bbox[:self.spatial_dim]),
                            "vector_a": cand.object
                        }
                sdist_newcomer = distance.euclidean(cand_entry["vector_s"], qs_space)
                cand_entry["dist_to_q"] = sdist_newcomer
                
                # (0) Find out leaders "near" to N
                for leader_idx in leaders:
                    leader_point = leaders[leader_idx]
                    if not self.diversity_pair(leader_point["vector_a"], cand_entry["vector_a"]):
                        group_l.append(leader_point)
                
                # (1) If group L is empty, filter buffer based on the newcomer
                #     Then set newcomer as a leader
                if len(group_l) == 0:
                    for leader_idx in leaders:
                        leader_point = leaders[leader_idx]
                        # Clear candidates near to the newcomer
                        leader_point["follower_buffer"] = [
                            x for x in leader_point["follower_buffer"]
                            if self.diversity_pair(x["vector_a"], cand_entry["vector_a"])
                        ]
                    cand_entry["follower_buffer"] = []
                    leaders[cand_entry["id"]] = cand_entry

                # (2) From current buffers, find the buffer that can be new leaders
                for leader_idx in leaders:
                    # Update spatial radius of follower domain first
                    leader_point = leaders[leader_idx]
                    self.update_follower_range(leader_point)

                    # Find followers that can become new leaders
                    diff_newcomer_leaderdomain = sdist_newcomer - leader_point["domain_radius"]
                    group_s = [
                        x for x in leader_point["follower_buffer"]
                        if x["dist_to_q"] < diff_newcomer_leaderdomain
                    ]
                    diversed_group_s = self.find_max_diversed_group(group_s)
                    # If a group > 2 points found, replace the leader with this group
                    if len(diversed_group_s) >= 2:
                        group_nons_backup = [
                            x for x in leader_point["follower_buffer"]
                            if x["dist_to_q"] >= diff_newcomer_leaderdomain
                        ]
                        del leader_point
                        # Note: It is unsafe to insert dictionary during iteration
                        # so, other than deletions, all update will be done at once
                        # after iteration.
                        followers_backup = followers_backup + group_nons_backup
                        new_leaders_backup = new_leaders_backup + diversed_group_s
                # (3) If near to only one leader, assign it as follower
                if len(group_l) == 1:                
                    lead_tmp = leaders[group_l[0]["id"]]
                    if len(lead_tmp["follower_buffer"]) < buf_sz:
                        lead_tmp["follower_buffer"].append(cand_entry)
                
                # (4) Reorganize leaders and followers
                # Try to assign followers to one of the leader
                # If can't, depose it
                # Note: There's no specific policy provided in the paper.
                #       Because inner-buffer search for diversed points is done when
                #       trying to swap leader, we decided to use a more simpler approach.
                #       In other words, choose the query-nearest leader which is not diversed
                #       in attribute space.
                for f in followers_backup:
                    leadidx_shuffled = [x for x in leaders.keys()]
                    shuffle(leadidx_shuffled)
                    for leader_idx in leadidx_shuffled:
                        leader_point = leaders[leader_idx]
                        if (not self.diversity_pair(f["vector_a"], leader_point["vector_a"])
                            and len(leader_point["follower_buffer"]) < buf_sz):
                            if len(leader_point["follower_buffer"]) == 0:
                                leader_point["follower_buffer"].append(f)
                            elif self.diversity_check_greedy(np.array([x["vector_a"] for x in leader_point["follower_buffer"]]), f["vector_a"]):
                                leader_point["follower_buffer"].append(f)

                # To avoid points being assigned to members of new leaders,
                # new leaders are inserted later
                for newlead in new_leaders_backup:
                    newlead["follower_buffer"] = []
                    leaders[newlead["id"]] = newlead
                    
                # print("go through one point...")
                if len(leaders) == k:
                    print("enough!")
                    break
            filtered += len(q_ans)
            # If not enough, start the next round
            if len(leaders) < k:
                num_iter += 1
                s_amount = math.ceil(s_amount * aggress)

        return [l_id for l_id in leaders] # Just return ids
    
    # Finding the radius of a leader's domain
    # Actually it's defined by the furthest point distance within its followers
    def update_follower_range(self, l):
        if len(l["follower_buffer"]) == 0:
            l["domain_radius"] = 0.0
            return
        f_vectors = np.array([x["vector_s"] for x in l["follower_buffer"]])
        l["domain_radius"] = distance.cdist([l["vector_s"]], f_vectors, "euclidean").max()
    
    
    # Find largest diversed group from followers with size >= 2
    # If not found, an empty tuple is returned
    def find_max_diversed_group(self, F):
        for i in range(2, len(F)+1):
            for j in combinations(F, i):
                if self.diversity_check_follower_pairwise(j):
                    return list(j)
        return []
    
    
    def diversity_check_follower_pairwise(self, F):
        for i in combinations(F, 2):
            if not self.diversity_pair(i[0]["vector_a"], i[1]["vector_a"]):
                return False
        return True
                
    
    def diversity_check_greedy(self, X, q):
        size_data, _ = X.shape

        for i in range(size_data):
            if not self.diversity_pair(X[i], q):
                return False
        return True
    
    def diversity_pair(self, p1, p2):
        diff_sorted = np.sort(np.absolute(p1 - p2))
        divdist_tmp = diff_sorted * self.weight

        return divdist_tmp.sum() > self.threshold
# End of class 'Motley'
