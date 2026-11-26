= Practical Reproduction

== Experimental Setup and Scope <sec:setup>
// Owner: M3
// TODO(M3):
// - State chosen scope: Full reproduction (DB1-3 vs CLARANS, sorted k-dist, runtime scaling)
// - Explain substitution of SEQUOIA 2000 with an equivalent 2D spatial point dataset
// - Document datasets, preprocessing, hyperparameters, hardware, software versions, and fixed seed (42)

== Technical Code Analysis <sec:code-analysis>
// Owner: M2

The from-scratch implementation lives in `Source/src/dbscan_repro/dbscan.py`
(clustering) and `Source/src/dbscan_repro/kdist.py` (parameter heuristic).
Neither file depends on `scikit-learn`; only the KD-tree used for neighborhood
queries comes from `scipy.spatial`, which the course specification allows as an
auxiliary standard component.

=== Mapping to the Paper

#table(
  columns: (auto, 1fr, 1.7fr),
  align: (left, left, left),
  inset: 4pt,
  [*Paper element*], [*Formal object*], [*Implementation*],
  [Definition 1],
    [$N_epsilon (p) = {q in D | "dist"(p, q) <= epsilon}$],
    [`_make_region_query` returns the indices of all points within `eps`, the point itself included],
  [Definition 2],
    [Core point: $|N_epsilon (p)| >= "MinPts"$],
    [`_expand_cluster` compares the neighborhood size with `min_pts` and records `is_core`],
  [Definitions 3--5],
    [Density-reachability, density-connectivity, cluster],
    [Breadth-first expansion of the queue: every core point reached from the seed contributes its neighborhood; Lemma 2 justifies starting from any core point],
  [Algorithm 1, outer loop],
    [Sequential scan over $D$],
    [`fit` scans samples in index order and opens a new cluster only when an `UNCLASSIFIED` point is a core point],
  [`ExpandCluster`],
    [Seed expansion],
    [`_expand_cluster` marks noise, labels the seeds, and processes the queue],
  [Section 4.2],
    [Sorted $k$-dist graph],
    [`compute_sorted_k_distances`, `suggest_eps_from_noise_ratio`, `detect_kdist_valley_index`],
)

The label bookkeeping mirrors the paper: `UNCLASSIFIED = -2` for points not yet
visited, `NOISE_LABEL = -1` for points already proven non-core, and `0, 1, ...`
for clusters. `core_sample_indices_` is stored in ascending order, the same
convention as `sklearn.cluster.DBSCAN` [@pedregosa2011scikit], which makes the
equivalence tests straightforward. Following the paper's recommendation for
two-dimensional data, the defaults are `min_pts = 4` and `k = 4`.

=== Implementation Details Omitted by the Paper

*Self-inclusion and the $k$-dist convention.* Definition 1 of @ester1996dbscan
includes $p$ itself in $N_epsilon (p)$ because $"dist"(p, p) = 0 <= epsilon$;
the code therefore compares $|N_epsilon (p)|$ directly with `min_pts` without
adding one. Section 4.2, in contrast, defines $k$-dist from the $k$-th nearest
neighbor, so a ball of radius $k$-dist($p$) contains $k + 1$ points. The
implementation keeps this asymmetry: `compute_sorted_k_distances` asks the
KD-tree for $k + 1$ neighbors and keeps column $k$, that is, it excludes the
point itself. With `min_pts = 4`, an `eps` equal to the 4-dist value yields
five points in the neighborhood, which satisfies the core condition.

*Border-point ownership.* Section 4.1 states that when a point belongs to two
nearby clusters it "will be assigned to the cluster discovered first". The
pseudocode, however, starts `ExpandCluster` with an unconditional
`changeClIds(seeds, ClId)`, which would overwrite a border point already
claimed by an earlier cluster. The implementation follows the prose: it assigns
the new cluster only to seeds whose label is `UNCLASSIFIED` or `NOISE`, never to
a point already owned by another cluster. Traversal order still decides the
owner when the clusters are discovered in a different order, because the first
core point encountered opens the cluster that reaches the shared border point
first. This behaviour is pinned down by
`test_border_point_assignment_depends_on_scan_order`.

*NOISE points are relabeled but not re-queued.* As explained in Section 4.1, a
point previously marked `NOISE` is known not to be a core point. When such a
point appears in the neighborhood of a core point, the implementation changes
its label to the current cluster but does not append it to the queue, avoiding
a redundant region query.

*Queue semantics.* The paper's `seeds` is a set. The implementation uses a FIFO
queue and labels a point at insertion time, so an unclassified point can enter
the queue at most once; no separate "visited" set is required.

*Spatial index.* The paper assumes an R\*-tree [@beckmann1990rstar] inside a
spatial database system. This project substitutes an in-memory
`scipy.spatial.KDTree`, which answers the same exact radius query. The DBSCAN
logic itself is written from scratch.

*Numerical conventions.* Brute-force queries compare squared Euclidean distances
with $epsilon^2$, avoiding a square root per pair, while the KD-tree uses
Euclidean distances. Both comparisons are inclusive ($<=$), matching
Definition 1.

=== Design Rationale and Ablation Analysis

*Brute force versus KD-tree.* Both strategies implement the same
$epsilon$-neighborhood, so they return identical labels and core points
(`test_brute_and_kdtree_are_identical`). The brute-force variant evaluates all
$n$ distances for every query, giving $O(n^2)$ time; the KD-tree prunes the
search and gives $O(log n)$ average query time and $O(n log n)$ overall for
low-dimensional data, reproducing the complexity argument of Section 4.1.
Removing the index therefore changes efficiency, not the result.

*Core-point condition.* The check $|N_epsilon (p)| >= "MinPts"$ is the only
place where density is tested. If it were removed, for example by treating
every point with at least one neighbor as a seed, DBSCAN would degenerate into
single-linkage clustering with threshold `eps`: chains of border points would
merge distinct clusters, reproducing the chaining effect that the density
notion is designed to avoid. Conversely, raising `min_pts` without changing
`eps` shrinks the set of core points and turns sparse cluster regions into
noise.

*Early noise marking.* A point whose neighborhood is too small is marked
`NOISE` immediately and may be reclaimed later as a border point of a cluster.
This is what allows the algorithm to distinguish "not yet visited"
(`UNCLASSIFIED`) from "visited and known non-core" (`NOISE`) and to skip the
latter when expanding a cluster.

*Removing core-sample bookkeeping.* Cluster labels would remain correct without
the `is_core` array, but the implementation could no longer expose
`core_sample_indices_`, which is needed to compare the core/border structure
with the paper and with scikit-learn.

=== Validation against scikit-learn

The test suite in `Source/tests/test_dbscan.py` compares the implementation
with `sklearn.cluster.DBSCAN` [@pedregosa2011scikit] on four synthetic datasets
(Gaussian blobs, two moons, concentric circles, and uniform noise) with several
`(eps, min_pts)` settings and both neighborhood strategies. For every run it
asserts exact equality of the label vector and of `core_sample_indices_`, not
merely a matching partition. The suite also covers edge cases (all-noise data,
`min_pts = 1`, duplicate points, invalid input) and the order dependence of
shared border points described above. `Source/tests/test_smoke.py` keeps a
lightweight end-to-end check for CI.

The agreement is expected rather than accidental: both implementations include
the point itself in the neighborhood, discover clusters in dataset order, and
keep the first cluster that claims a border point. The remaining difference is
the spatial index (KD-tree versus scikit-learn's own tree/brute strategies),
which does not affect the result.

For the $k$-dist heuristic, `test_k_dist_excludes_the_point_itself` checks the
neighbor convention on a unit square, and
`test_k_dist_sorted_descending_with_known_values` verifies the analytical
values on a one-dimensional grid. The two threshold helpers are unit-tested
separately; they automate the visual choice described in Section 4.2 for the
reproduction experiment.

== Results Comparison and Analysis <sec:results>
// Owner: M3
// TODO(M3):
// - Place reproduced results (tables, scatter plots, runtime scaling curves) alongside paper claims
// - Report relative speedup / differences and explain deviations (1996 vs modern hardware, synthetic DB recreation, spatial index type)
