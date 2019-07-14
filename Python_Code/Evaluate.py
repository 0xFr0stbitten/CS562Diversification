from scipy.spatial import distance

# q_space: query in spatial space, S: returned neighbors
# -space: spatial vector; -attr: attribute vector
def evaluation(q_space, S_space, S_attr, metric_s="euclidean", metric_a="cosine"):
    # max_dist
    q_dists = distance.cdist([q_space], S_space, metric=metric_s)
    avg_query_dist = q_dists.mean()
    max_dist = q_dists.max()
    p_dists = distance.pdist(S_attr, metric=metric_a)
    avg_pairwise_dist = p_dists.mean()
    # Farthest point from query in spatial space / Average distance in spatial space
    # / Average pairwise distance between answers in attribute space
    return max_dist, avg_query_dist, avg_pairwise_dist 