= Practical Reproduction

== Experimental Setup and Scope <sec:setup>
// Owner: M3

We carry out the full reproduction scope: compare DBSCAN with the CLARANS
medoid baseline on three synthetic spatial datasets, inspect sorted $k$-dist
curves for choosing $"Eps"$, and measure runtime scaling at the five sample
counts used in the paper's SEQUOIA 2000 experiment. The paper's original
database and point coordinates are not available in this repository. DB1--3
are therefore deterministic, figure-inspired synthetic analogues, not exact
copies of the original samples: DB1 contains four separated groups with
unequal spread and uniform background noise; DB2 contains two noisy,
interlocking arc clusters; DB3 contains two concentric ring clusters with
uniform noise. Ground-truth noise points are labeled $-1$ for evaluation.

The substitute scaling data are bounded 2D point clouds with twelve or fewer
Gaussian population centers and a dispersed uniform component. This preserves
the spatial, low-dimensional setting relevant to DBSCAN's neighborhood index;
it does not reproduce the SEQUOIA 2000 dataset or the paper's hardware. All
coordinates are used directly without additional preprocessing. Generators are
deterministic with seed 42. Effectiveness runs use 1,000 points, Euclidean
distance, and $"MinPts"=4$, with per-dataset $"Eps"$ values of 0.03, 0.025, and
0.04 for DB1--3, respectively. The $k$-dist experiment uses $k=4$. These
settings are recorded in `Source/experiments/configs/default.yaml`. CLARANS
uses the known number of
non-noise groups for each synthetic dataset, one randomized local search, and
at most 40 unsuccessful neighbors per local search. Runtime runs use
$"MinPts"=4$, $"Eps"=0.04$, the configured sample counts
`[1252, 2503, 3910, 5213, 6256]`, and three repetitions per method. CLARANS
runtime uses $k=8$, one local search, and a 40-neighbor limit so the randomized
baseline remains tractable at the largest size.

Each experiment exports CSV/PNG results and a JSON metadata record containing
the seed, full YAML configuration, Python and dependency versions, operating
system, and processor identifier. Runtime comparisons are interpreted as
within-machine scaling trends and ratios; they are not compared as absolute
seconds with the 1996 measurements. The recorded environment is Windows 11
(build 10.0.26200), with an Intel64 Family 6 Model 170 processor identifier,
Python 3.12.15, NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1, and Matplotlib
3.11.2. The full per-run environment details are preserved in the experiment
metadata JSON files.

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

The side-by-side point plots in `Source/results/effectiveness_db1.png`,
`effectiveness_db2.png`, and `effectiveness_db3.png` compare ground truth,
DBSCAN, and CLARANS. The CSV summary records Adjusted Rand Index (ARI),
normalized mutual information (NMI), predicted cluster count excluding noise,
and predicted noise ratio. ARI and NMI evaluate the full partition, treating
the noise label as a predicted group; cluster count and noise ratio are also
reported separately so that a high partition score does not hide a different
noise decision. The table below gives the measured comparison for
seed 42; the exact values are exported in `Source/results/effectiveness.csv`.

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: center,
  inset: 4pt,
  [*Dataset*], [*Method*], [*ARI*], [*NMI*], [*Clusters*], [*Noise*],
  [DB1], [DBSCAN], [0.924], [0.895], [4], [10.0%],
  [DB1], [CLARANS], [0.860], [0.846], [4], [0.0%],
  [DB2], [DBSCAN], [0.977], [0.937], [2], [3.7%],
  [DB2], [CLARANS], [0.281], [0.226], [2], [0.0%],
  [DB3], [DBSCAN], [0.941], [0.892], [2], [5.2%],
  [DB3], [CLARANS], [-0.000], [0.001], [2], [0.0%],
)

DBSCAN matches the intended group counts in all three analogues and obtains
the highest ARI/NMI on each. The advantage is modest on compact DB1 and large
on the non-convex DB2 arcs and DB3 rings, where a medoid partition cannot
follow the curved cluster boundaries. Predicted noise is also closer to the
ground truth for DBSCAN; CLARANS assigns every sample to a cluster by design.

The $k$-dist figures in `Source/results/kdist_db1.png` through
`kdist_db3.png` plot each point's distance to its fourth nearest neighbor in
descending order. The vertical and horizontal markers show the automated
valley estimate; the numeric indices and suggested $"Eps"$ values are saved in
`Source/results/kdist_summary.csv`. As in the paper, the curve is a visual aid
for selecting a density threshold, not a guarantee that one global $"Eps"$
recovers every cluster when the local densities differ. The detected values
are 0.0305, 0.0247, and 0.0291 for DB1--3. The effectiveness run uses 0.03
and 0.025 for DB1 and DB2; for DB3, 0.04 separates the two rings despite the
automated valley's lower estimate, illustrating the need to inspect the curve
and resulting spatial partition together.

The runtime table and log-scale curve are exported as
`Source/results/runtime_scaling.csv` and `runtime_scaling.png`. Reported means
and standard deviations summarize three runs on the same generated point set
for each size. Brute-force DBSCAN scans all points for each neighborhood
query, while the KD-tree can prune queries in this two-dimensional setting;
the observed speedup is the brute-force mean divided by the KD-tree mean and
is meaningful only for this implementation and machine. CLARANS has a
different objective and randomized search budget, so its timings are a
baseline trend rather than an algorithmically identical workload.

#table(
  columns: (auto, auto, auto, auto),
  align: center,
  inset: 4pt,
  [*Points*], [*DBSCAN brute (s)*], [*DBSCAN KD-tree (s)*], [*CLARANS (s)*],
  [1,252], [0.043], [0.049], [0.006],
  [2,503], [0.118], [0.101], [0.018],
  [3,910], [0.268], [0.176], [0.023],
  [5,213], [0.370], [0.308], [0.053],
  [6,256], [0.899], [0.488], [0.031],
)

At 1,252 points, the KD-tree is slightly slower in this run (0.049 s versus
0.043 s), where tree construction and query overhead outweigh pruning. At
6,256 points it is about 1.84 times faster than brute force (0.899 s / 0.488
s). Across the endpoints, brute-force time grows about 21.1 times, compared
with about 10.0 times for the KD-tree and 5.5 times for the bounded CLARANS
search. This is consistent with the KD-tree advantage emerging as the input
grows, while the measured factors depend on the generated point density and
implementation details.

These experiments reproduce the paper's comparisons in structure, but not
its original input coordinates or 1996 machine. Synthetic geometry, current
hardware, Python/SciPy implementations, and our explicitly bounded CLARANS
search budget all affect the scores and absolute times. Consequently, we
interpret the figures through clustering behavior and scaling ratios rather
than claiming exact numerical replication. Run-specific results and machine
metadata are stored alongside the generated artifacts.
