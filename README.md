## CS562 2018 Spring Final Project -- Diversification Among Search Results in k-Nearest Neighbor Search Problem

### Abstract

To solve the problem of returning too many homogeneous results in any type of search, several algorithms have been created for generation a diversed set of results while maintaining closeness to query. Specifically, this paper will deal with several content-based diversity for refining <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-Nearest-Neighbor (<img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN) search, and compare performance of these algorithms on real case dataset.

### Introduction and Motivation

Often when people search on the web, they prefer a more heterogeneous result in web search. In some situations, search result that only contain homogeneous information may not prove to be all that useful, or satisfactory to users. For example, when we use web application such as Yelp to look for food restuarants nearby, we may expect a diverse set of restaurants in this area, instead of restaurants that all offer food of the same style. Users of such applications have been found to prefer some sort of diversity, and that applies to other types of recommendation system. To increase user satisfaction, algorithms should be implemented to return a "diverse" set of results from a data set. This is known as the diversification problem.

There is no specific definition for diversity within a dataset. Definitions of diversity include content-based, novelty-based, and coverage-based diversity. Content-based diversity is defined based on similarity (or dissimilarity) of items. For novelty-based and coverage-based diversity, documents are labeled with some pre-defined information (usually called "concept" or "information nugget") for each document, while the previous one seeks to maximize information gain for each incoming document, and the later tries to maximize number of covered information for the returned set. In some research, several definitions are combined to form a hybrid model, and require additional learning to obtain proper weightings for each model.

Search for diverse nearest-neighbor is an instance of the p-dispersion problem, which is to choose <img src="/tex/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.270567249999992pt height=14.15524440000002pt/> out of <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> points so minimum distance between any pairs of chosen points is maximized; and this kind of problem is generally NP-hard. In real applications like online suggestion system it's crutial to keep the respond time be short enough, so most  research to this problem take approximated result to increase efficiency.

It is important to solve the diversification problem because diversity not only satisfies users by providing broader result from ambiguity queries in recommendation system, it also allows users to find out additional information related to their requirement and thus help narrow down subsequent queries. By solving the diversification problem and increase variety to the answer, we are one step closer to having optimal search results.

### Problem Definition

The problem we are trying to solve is to add diversity onto solution for original <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-Nearest-Neighbor (<img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN) search problem, which means to provide diversity within the result for a query according to given definition of diversity, and maintain enough relevance to the query in the meantime. In this project, we focus on content-based definition for diversity, implement algorithm based on these definitions, and compare performance between different approaches.

### Method/Approach

In general <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN search, elements in a dataset are represented in a n-dimension feature space, along with metric function that defines distance in this space. And some data structures, such as R-tree or kd-tree can be applied to help speeding up search process. For content-based definition, diversity can be either defined in the same space used for <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN search, or defined in another space that is independent to, or share some of the dimentions with the original one. In the later case, distance in the diversity space can be defined with other metrics, and sometimes even allows us to reduce complexity in <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN search, since search in the diversity space can be reduced to subset from preliminary result. Also, in real cases such as map-based recommendation system, spatial information can be separated from other attributes of items or points, and it's more reasonable because goal of these applications is to provide result that's both near to query position and with enough diversity in other features, compared to just combine all aspects together.

As for heuristics used for speed-up and approximation, these strategies can be simply categorized into greedy heuristics and swap heuristics. Greedy heuristics set up some criteria to decide whether an input should be included in the answer set from the input, and keep testing the next nearest candidate until reach the required answer size. Swap heuristics usually initialize the result with original k-NN result, and trying to replace some points with other candidate points which can increase diversity in each iteration.

In this project, we implemented algorithms that tries to find a diversed set of results within a dataset; in the case of this paper, diversity is defined by the content-based definition. One of the algorithms we used is *MOTLEY* algorithm, which can produce near-optimal solution to the <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-Nearest Diverse Neighbor problem.

MOTLEY is a content-based, greedy-heuristic <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN search algorithm aims for increasing result diversity. It separates the search space into point-attribute space and diversity-attribute space. This algorithm first generates initial result from point-attribute space, and then decide inclusion of intermediate result in diversity-attribute space. MOTLEY also uses R-tree structure, and utilize properties of this structure for further performance improvement. The orignal paper introduces two heuristics for retrieving approximated result, simple-greedy and buffered-greedy. In the following analysis, we will focus on the former heuristic.

MOTLEY defines a binary function to define whether two points are diverse to each other or not. First it computes <img src="/tex/669f1729652bf907c884848a1b80698c.svg?invert_in_darkmode&sanitize=true" align=middle width=61.65833354999999pt height=22.465723500000017pt/> to obtain diversity distance between two points, then set a threshold, called <img src="/tex/ab4f65ae1eb3ac186af343653ac84b43.svg?invert_in_darkmode&sanitize=true" align=middle width=66.64085339999998pt height=22.465723500000017pt/>, to decide truth of the diversity between them. The <img src="/tex/669f1729652bf907c884848a1b80698c.svg?invert_in_darkmode&sanitize=true" align=middle width=61.65833354999999pt height=22.465723500000017pt/> function is defined as follows:

<p align="center"><img src="/tex/6d0ef4d9cdd82282bd4de729364d00e6.svg?invert_in_darkmode&sanitize=true" align=middle width=270.87623475pt height=50.04352485pt/></p>

Which can be obtained by weighted sum on sorted difference of 1D distance over all dimensions in the diversity-attribute space. The threshold <img src="/tex/ab4f65ae1eb3ac186af343653ac84b43.svg?invert_in_darkmode&sanitize=true" align=middle width=66.64085339999998pt height=22.465723500000017pt/> is set between <img src="/tex/e88c070a4a52572ef1d5792a341c0900.svg?invert_in_darkmode&sanitize=true" align=middle width=32.87674994999999pt height=24.65753399999998pt/>, and it's equivalent to general <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN search in point-attribute space if threshold is set to 0. MOTLEY requires all points in the final output are having <img src="/tex/669f1729652bf907c884848a1b80698c.svg?invert_in_darkmode&sanitize=true" align=middle width=61.65833354999999pt height=22.465723500000017pt/> value larger than <img src="/tex/ab4f65ae1eb3ac186af343653ac84b43.svg?invert_in_darkmode&sanitize=true" align=middle width=66.64085339999998pt height=22.465723500000017pt/> with all other points in the set.

In addition to MOTLEY algorithm, we will also present another greedy heuristic algorithm for solving the diversification problem. The simple greedy heuristic algorithm first retrieves a candidate set <img src="/tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/>, with size larger than <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> by normal <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN search, and initialize the answer set <img src="/tex/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode&sanitize=true" align=middle width=11.027402099999989pt height=22.465723500000017pt/> with the top-nearest point in <img src="/tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/>. Then rank all the rest points in <img src="/tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/> based on (1) Distance to the query (relevance), and (2) Distance to current answer set <img src="/tex/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode&sanitize=true" align=middle width=11.027402099999989pt height=22.465723500000017pt/> (diversity), choose the top-rank item and move it to answer set <img src="/tex/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode&sanitize=true" align=middle width=11.027402099999989pt height=22.465723500000017pt/>. This is repeated until <img src="/tex/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode&sanitize=true" align=middle width=11.027402099999989pt height=22.465723500000017pt/> reaches the required answer size <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>. The distance from candidates to answer set can be defined by using (1) Distance to the nearest point in <img src="/tex/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode&sanitize=true" align=middle width=11.027402099999989pt height=22.465723500000017pt/>, or (2)Average distance among points in <img src="/tex/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode&sanitize=true" align=middle width=11.027402099999989pt height=22.465723500000017pt/>:

<p align="center"><img src="/tex/f48df2034c80aad9562160f2866f983c.svg?invert_in_darkmode&sanitize=true" align=middle width=471.57476519999994pt height=44.6002293pt/></p>

To compare the performance, we performed experiments on the real world dataset used in original MOTLEY paper. *Forest Cover* is a dataset contains geographical information of around 580,000 tuples. In our experiment, we take horizontal
and vertical distance to nearest water as point-attributes, and using elevation, aspect and slope for diversity-attributes. We conducted two rounds of experiments on this dataset. In the first round, both <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN and diversity search are based on the whole space (combination of point-attribute and diversity-attribute space) for each algorithm. In the second round, initial <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN search is conducted on the point-attribute space, and then select candidates based on distance in diversity-attribute space.

Note that for the experiments, data structure used in <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN search stage for MOTLEY algorithm is replaced by kd-tree. However, we also provide R-tree structure for <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN search in our code for those who are interested in difference on
performance.

### Results

For all experiments, we tried to find out the 10 nearest-neighbors as answer. For <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN search, we use the default settings in the library. For simple-greedy algorithm, we tried both minimum-distance and mean-distance as point-set distance measure, and run experiments with different candidate size (20, 30, and 50 points). For MOTLEY algorithm, we only modify the diversity threshold (<img src="/tex/ab4f65ae1eb3ac186af343653ac84b43.svg?invert_in_darkmode&sanitize=true" align=middle width=66.64085339999998pt height=22.465723500000017pt/>) of 0.01, 0.03, 0.05. For weighting in <img src="/tex/669f1729652bf907c884848a1b80698c.svg?invert_in_darkmode&sanitize=true" align=middle width=61.65833354999999pt height=22.465723500000017pt/> computation, we use the function defined in the paper, with decay factor a fixed to 0.1:

<p align="center"><img src="/tex/6cb62aa6e1c477e14255a4792a6fd631.svg?invert_in_darkmode&sanitize=true" align=middle width=240.92260664999998pt height=37.345933349999996pt/></p>

To evaluate the performance, we compute the following metrics from the answer set:

1. Distance between farthest point to the query in point-attribute space.
2. Average distance between points in the answer set to the query in point-attribute space.
3. Average pairwise diversity of the answer set in diversity-attribute space.

Note that in the real world cases, especially in map applications, (2) and (3) are meaningful to user and service provider, because we want to obtain the largest variety from the result within the smallest overall range.

From figures (1)~(4), we can find the difference between searching in the whole space and doing different search in separated spaces. Running <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN on the whole space will generate answer with lower diversity in a wilder result range, while spliting space generate a more compact result, with higher answer diversity along with smaller distance from the query.

We can also notice that MOTLEY generate result with much higher diversity, only
with the cost of moderate increase on average distance from query point.

![](/img/Report_P1.png) 
![](/img/Report_P2.png) 
![](/img/Report_P3.png) 
![](/img/Report_P4.png) 

### Conclusion and Future Work

From above experiences, we’ve found that both separating original space into spatial (point-attribute) space and diversity space, and usage of MOTLEY algorithm, can refine result of diversed <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-NN search. One thing worth mentioning is the selection of threshold in MOTLEY algorithm, since it can easily lead to full scan on the whole database, without some further improvement such as additional pruning procedure on R-tree traversal. The paper suggests to set the threshold lower than 0.2 empirically. However, in our case full scan on the dataset occur in most cases when threshold is higher than 0.05. Due to time constraint and library limitation we’re not able to do research on this issue. We also implemented the buffered-greedy algorithm mentioned in MOTLEY paper, however, the running speed is much slower comparing to simple-greedy approach, and full scan problem mentioned above is more severe on this approach, so its experiment result is not included here.

### References

1. Drosou, Marina, and Evaggelia Pitoura. "Search result diversification." SIGMOD record 39.1 (2010): 41-47. [Link](http://www.cs.uoi.gr/~pitoura/distribution/sr10.pdf)
2. Jain, Anoop, Parag Sarda, and Jayant R. Haritsa. "Providing diversity in k-nearest neighbor query results." Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, Berlin, Heidelberg, 2004. [[1]](http://dsl.cds.iisc.ac.in/pub/motley.pdf) [[2]](https://arxiv.org/pdf/cs/0310028.pdf)
3. Forest Cover dataset [Link](https://archive.ics.uci.edu/ml/datasets/covertype)
