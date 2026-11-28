= Theory and Paper Analysis

== Paper Selection and Motivation <sec:paper-choice>
// Owner: M1
// TODO(M1): Summarize why Ester et al. (KDD-96) meets §1.1.1 criteria:
// peer-reviewed venue (KDD-96), explains clustering via density definitions,
// includes effectiveness + efficiency experiments.

We study @ester1996dbscan, which introduces DBSCAN to address three requirements for clustering large spatial databases: minimal domain knowledge for parameter selection, discovery of clusters of arbitrary shape, and efficiency on large datasets.

== Core Method and Mathematical Formulation <sec:core-method>
// Owner: M5 (definitions, lemmas, algorithm, complexity) & M4 (course links)
// TODO(M5): Formalize:
// - Eps-neighborhood: N_Eps(p) = { q in D | dist(p, q) <= Eps }
// - Core point condition: |N_Eps(p)| >= MinPts
// - Directly density-reachable, density-reachable, density-connected
// - Cluster & Noise definitions, Lemmas 1 and 2 with intuition
// - Time & space complexity: O(n log n) with spatial index vs O(n^2) brute force
// - Comparison with CLARANS (Ng & Han)

=== Connection to Course Fundamentals
// Owner: M4

*Unsupervised learning and cluster representation.* DBSCAN, $k$-means, and $k$-medoids all infer groups from unlabeled observations.
Both prototype-based methods take $k$ as input and assign each observation to its nearest representative.
In $k$-means, the representative is the cluster mean and the objective is to minimize the sum of squared distances to those means.
In $k$-medoids, each representative is an observed point and the objective is to minimize the total dissimilarity to the chosen medoids.
DBSCAN uses density connectivity among observations, so the number of clusters emerges from the data for a given choice of `Eps` and `MinPts`.
This makes its cluster representation nonparametric in the sense that it has no fixed set of prototypes.
For example, DBSCAN can recover two separated crescent-shaped bands by following chains of nearby core points along each band.
A nearest-prototype partition can cut across the same bands @ester1996dbscan.

*Distance and feature representation.* As in nearest-neighbor methods, distances depend on how observations are represented.
With Euclidean distance, a feature with a much wider numerical range can dominate the distance and determine which points fall within `Eps`.
Standardizing features can make their scales comparable when they are intended to have similar influence.
In high-dimensional data with many irrelevant features, near and far distances can become similar relative to their magnitude @beyer1999nearest.
When this happens, a small change in `Eps` can substantially change neighborhood counts and which points qualify as core points.

*Hyperparameters and cluster granularity.* At fixed `MinPts`, a small `Eps` can fragment a cluster or leave points labeled as noise.
A larger radius can join distinct groups through a chain of core points.
At fixed `Eps`, increasing `MinPts` raises the local density requirement and can remove sparse clusters.
Decreasing it can let sparse bridges connect groups.
This resembles the model-complexity trade-off discussed in CSC14005.
A clustering can be too coarse to preserve meaningful groups or too sensitive to local fluctuations to recover a stable structure.
A single global setting can therefore struggle when dense and sparse clusters coexist @ester1996dbscan.

== Technical Evaluation and Critique <sec:evaluation>
// Owner: M1
// TODO(M1): Provide:
// - >= 2 technical strengths backed by paper/reproduction evidence
// - >= 2 testable limitations / implicit assumptions (e.g., single global Eps, high-dim distance concentration)
// - >= 1 improvement / alternative directly linked to a stated limitation (e.g., OPTICS/HDBSCAN)
// - Team's critical perspective on practical effectiveness
