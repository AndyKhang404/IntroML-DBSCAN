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
// TODO(M4): Connect DBSCAN to CSC14005 foundations:
// - Unsupervised learning & non-parametric vs parametric clustering (k-means/k-medoids)
// - Distance metrics & curse of dimensionality
// - Hyperparameter trade-offs (Eps, MinPts) and bias-variance / under- vs over-segmentation analogy

== Technical Evaluation and Critique <sec:evaluation>
// Owner: M1
// TODO(M1): Provide:
// - >= 2 technical strengths backed by paper/reproduction evidence
// - >= 2 testable limitations / implicit assumptions (e.g., single global Eps, high-dim distance concentration)
// - >= 1 improvement / alternative directly linked to a stated limitation (e.g., OPTICS/HDBSCAN)
// - Team's critical perspective on practical effectiveness
