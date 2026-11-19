= Practical Reproduction

== Experimental Setup and Scope <sec:setup>
// Owner: M3
// TODO(M3):
// - State chosen scope: Full reproduction (DB1-3 vs CLARANS, sorted k-dist, runtime scaling)
// - Explain substitution of SEQUOIA 2000 with an equivalent 2D spatial point dataset
// - Document datasets, preprocessing, hyperparameters, hardware, software versions, and fixed seed (42)

== Technical Code Analysis <sec:code-analysis>
// Owner: M2
// TODO(M2):
// - Map Source/src/dbscan_repro/dbscan.py components to paper definitions & Algorithm 1
// - Highlight implementation details omitted by the paper:
//   (1) Border point assignment depends on point traversal order
//   (2) MinPts counts the point itself (p in N_Eps(p))
//   (3) Spatial index choice (KD-tree vs R*-tree)
// - Explain design rationale and what happens if components (e.g., KD-tree, core check) are removed/modified
// - Report validation equivalence against scikit-learn DBSCAN

== Results Comparison and Analysis <sec:results>
// Owner: M3
// TODO(M3):
// - Place reproduced results (tables, scatter plots, runtime scaling curves) alongside paper claims
// - Report relative speedup / differences and explain deviations (1996 vs modern hardware, synthetic DB recreation, spatial index type)
