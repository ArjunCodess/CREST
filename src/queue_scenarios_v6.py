"""Onset seeds, scenario expected-IoU decoding, graph cuts and proxy diagnostics.

Seed scores are a categorical distribution. Marginal queue probabilities are a
different object. Footprints use directed hop distance. A lexical link index is
not a distance. The single-seed model does not invent one true seed when several
links are queued.
"""
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from .data import family, read_csv
from .evaluation import proxy_queue_iou
from .latent_queue_v6 import QUEUE_RATIO, kalman_filter, queue_v6_features
from .queue_events import decode_queue, visible_event
from .queue_onset import decode_onset


TEMPERATURES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MIN_FOOTPRINTS = 8
MARGINAL_GAP = 0.05


def binary_adjacency(root, panel, link_ids):
    """Unweighted directed mainline adjacency. Connectors outside link_ids are dropped."""
    path = Path(root) / 'corridors' / panel / 'network' / 'lwr_mainline_topology.csv'
    index = {str(link): i for i, link in enumerate(link_ids)}
    adjacency = np.zeros((len(link_ids), len(link_ids)))
    for row in read_csv(path).itertuples(index=False):
        dest = index.get(str(row.link_id))
        if dest is None:
            continue
        incoming = str(getattr(row, 'incoming_link_ids', '') or '')
        outgoing = str(getattr(row, 'outgoing_link_ids', '') or '')
        for source in incoming.split(';'):
            src = index.get(source)
            if src is not None and src != dest:
                adjacency[src, dest] = 1
        for nxt in outgoing.split(';'):
            other = index.get(nxt)
            if other is not None and other != dest:
                adjacency[dest, other] = 1
    return adjacency


def undirected_pairs(adjacency):
    """Each directed edge contributes one undirected cut edge, not one per direction."""
    adjacency = np.asarray(adjacency) > 0
    pairs = set()
    for i, j in zip(*np.nonzero(adjacency)):
        if i != j:
            pairs.add((int(min(i, j)), int(max(i, j))))
    return sorted(pairs)


def directed_offsets(adjacency, max_hop=32):
    """Signed hop offset[seed, link]. Positive means the link is downstream of the seed.

    A finite value requires a unique directed path direction. Cycles become missing.
    The numeric link index is never added or subtracted.
    """
    adjacency = np.asarray(adjacency) > 0
    n = adjacency.shape[0]
    downstream = [np.flatnonzero(adjacency[i]) for i in range(n)]
    upstream = [np.flatnonzero(adjacency[:, i]) for i in range(n)]
    offset = np.full((n, n), np.nan)
    for seed in range(n):
        offset[seed, seed] = 0
        dist = {seed: 0}
        queue = deque([seed])
        while queue:
            node = queue.popleft()
            for nxt in downstream[node]:
                nxt = int(nxt)
                if nxt not in dist:
                    dist[nxt] = dist[node] + 1
                    queue.append(nxt)
                    if dist[nxt] <= max_hop:
                        offset[seed, nxt] = dist[nxt]
        dist_up = {seed: 0}
        queue = deque([seed])
        while queue:
            node = queue.popleft()
            for prev in upstream[node]:
                prev = int(prev)
                if prev not in dist_up:
                    dist_up[prev] = dist_up[node] + 1
                    queue.append(prev)
                    if dist_up[prev] <= max_hop:
                        current = offset[seed, prev]
                        if np.isfinite(current) and current != 0:
                            offset[seed, prev] = np.nan
                        elif not np.isfinite(current):
                            offset[seed, prev] = -dist_up[prev]
    return offset


def weak_components(mask, adjacency):
    nodes = [int(i) for i in np.flatnonzero(np.asarray(mask, bool))]
    parent = {node: node for node in nodes}
    members = set(parent)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    adjacency = np.asarray(adjacency) > 0
    for node in nodes:
        neighbors = set(int(j) for j in np.flatnonzero(adjacency[node]))
        neighbors.update(int(j) for j in np.flatnonzero(adjacency[:, node]))
        for other in neighbors:
            if other in members:
                parent[find(other)] = find(node)
    groups = {}
    for node in nodes:
        groups.setdefault(find(node), []).append(node)
    return list(groups.values())


def downstream_fronts(component, adjacency):
    adjacency = np.asarray(adjacency) > 0
    members = set(component)
    fronts = []
    for node in component:
        if not any(int(j) in members for j in np.flatnonzero(adjacency[node])):
            fronts.append(node)
    return fronts


def footprint_summary(masks, adjacency, max_hop=32, min_components=MIN_FOOTPRINTS):
    """Empirical inclusion by directed offset, allowing several queues in one window.

    A component with zero or several downstream fronts is an unsupported mapping
    and is not given a synthetic unique seed. Fewer than `min_components`
    supported components is an explicit skip, not a fitted footprint.
    """
    adjacency = np.asarray(adjacency)
    offsets = directed_offsets(adjacency, max_hop=max_hop)
    included = {}
    seen = {}
    unsupported = 0
    unmapped_links = 0
    disconnected = 0
    supported = 0
    sizes = []
    for mask in masks:
        mask = np.asarray(mask, bool)
        sizes.append(int(mask.sum()))
        components = weak_components(mask, adjacency)
        if len(components) > 1:
            disconnected += 1
        for component in components:
            fronts = downstream_fronts(component, adjacency)
            if len(fronts) != 1:
                unsupported += 1
                continue
            front = fronts[0]
            supported += 1
            reachable = np.isfinite(offsets[front])
            for link in np.flatnonzero(reachable):
                distance = int(offsets[front, link])
                if abs(distance) > max_hop:
                    continue
                seen[distance] = seen.get(distance, 0) + 1
                if mask[link]:
                    included[distance] = included.get(distance, 0) + 1
            for link in component:
                if not np.isfinite(offsets[front, link]):
                    unmapped_links += 1
    status = 'empirical' if supported >= min_components else 'skipped_small_sample'
    distances = sorted(seen)
    rate = {d: included.get(d, 0) / seen[d] for d in distances}
    return {
        'status': status,
        'robust': status == 'empirical',
        'supported_components': supported,
        'unsupported_components': unsupported,
        'positive_links_unmapped': unmapped_links,
        'disconnected_windows': disconnected,
        'footprint_sizes': sizes,
        'inclusion_by_offset': {str(k): rate[k] for k in distances},
        'offset_counts': {str(k): seen[k] for k in distances},
        'coordinate': 'downstream front of a weakly connected positive component; not a claimed unique generative seed',
        'note': 'Disconnected windows can contain several queues. A single categorical seed does not represent them.',
    }


def match_marginals(seed_probability, conditional, marginal, steps=25, tolerance=MARGINAL_GAP):
    """Scale P(link | seed) so the mixture tracks the marginals. Report the L1 gap.

    Structural zeros can make an exact match impossible. That is a changed joint
    model, not a silent recalibration.
    """
    pi = np.asarray(seed_probability, float)
    pi = pi / pi.sum()
    fitted = np.clip(np.asarray(conditional, float), 0, 1)
    marginal = np.asarray(marginal, float)
    for _ in range(steps):
        mix = pi @ fitted
        scale = np.ones_like(mix)
        ok = mix > 1e-12
        scale[ok] = marginal[ok] / mix[ok]
        fitted = np.clip(fitted * scale, 0, 1)
    mix = pi @ fitted
    gap = float(np.abs(mix - marginal).sum())
    return fitted, gap, bool(gap <= tolerance)


def conditional_from_offsets(offsets, inclusion, links):
    """P(link | seed) from a directed-offset inclusion table. Missing offsets stay 0."""
    n = len(links) if not isinstance(links, int) else links
    table = {int(k): float(v) for k, v in inclusion.items()}
    conditional = np.zeros((n, n))
    for seed in range(n):
        for link in range(n):
            distance = offsets[seed, link]
            if np.isfinite(distance) and int(distance) in table:
                conditional[seed, link] = table[int(distance)]
    return conditional


def sample_scenarios(probability, n_scenarios, seed):
    probability = np.clip(np.asarray(probability, float).reshape(-1), 0, 1)
    rng = np.random.default_rng(seed)
    return rng.random((n_scenarios, probability.size)) < probability


def exact_expected_iou(probability, prediction):
    """Enumerate independent Bernoulli outcomes. Limited to 12 links."""
    probability = np.clip(np.asarray(probability, float).reshape(-1), 0, 1)
    prediction = np.asarray(prediction, bool).reshape(-1)
    n = probability.size
    if n > 12:
        raise ValueError('Exact expected IoU is enumerated only for at most 12 links')
    if prediction.size != n:
        raise ValueError('Prediction width does not match the probability vector')
    total = 0.0
    for mask in range(1 << n):
        prob = 1.0
        intersection = 0
        union = 0
        for i in range(n):
            bit = (mask >> i) & 1
            prob *= probability[i] if bit else (1.0 - probability[i])
            if bit and prediction[i]:
                intersection += 1
            if bit or prediction[i]:
                union += 1
        total += prob * (1.0 if union == 0 else intersection / union)
    return float(total)


def best_exact_set(probability):
    """Smallest-mask tie break among nonempty sets. Limited to 8 links."""
    probability = np.clip(np.asarray(probability, float).reshape(-1), 0, 1)
    n = probability.size
    if n > 8:
        raise ValueError('Exact set search supports at most 8 links')
    best_value, best_mask = -1.0, 1
    for mask in range(1, 1 << n):
        prediction = np.array([(mask >> i) & 1 for i in range(n)], dtype=bool)
        value = exact_expected_iou(probability, prediction)
        if value > best_value + 1e-15 or (abs(value - best_value) <= 1e-15 and mask < best_mask):
            best_value, best_mask = value, mask
    prediction = np.array([(best_mask >> i) & 1 for i in range(n)], dtype=bool)
    return prediction, best_value


def greedy_expected_iou(probability, n_scenarios=500, seed=2026, max_passes=None):
    """Greedy add/drop from a nonempty start. Scenarios use a deterministic seed.

    The objective is the Monte Carlo mean of intersection/union under independent
    Bernoulli draws of the supplied marginals. It is not the expected-intersection
    over expected-union surrogate.
    """
    probability = np.clip(np.asarray(probability, float).reshape(-1), 0, 1)
    n = probability.size
    if n == 0:
        raise ValueError('Expected-IoU decoding received no cells')
    scenarios = sample_scenarios(probability, n_scenarios, seed)
    return optimize_scenario_iou(scenarios, probability, max_passes)


def optimize_scenario_iou(scenarios, probability, max_passes=None):
    """Greedy nonempty set under an explicitly supplied scenario distribution."""
    probability = np.asarray(probability, float).reshape(-1)
    scenarios = np.asarray(scenarios, bool)
    n = probability.size
    start = int(np.flatnonzero(probability == probability.max())[0])
    chosen = np.zeros(n, dtype=bool)
    chosen[start] = True
    passes = n + 2 if max_passes is None else max_passes
    for _ in range(passes):
        current, intersection, union = _scenario_iou(scenarios, chosen)
        best = None
        off = np.flatnonzero(~chosen)
        if len(off):
            added = (intersection[:, None] + scenarios[:, off]) / (union[:, None] + ~scenarios[:, off])
            scores = added.mean(axis=0)
            pick = int(np.argmax(scores))
            best = _prefer(best, float(scores[pick]), int(off[pick]), 'add')
        on = np.flatnonzero(chosen)
        if len(on) > 1:
            dropped = (intersection[:, None] - scenarios[:, on]) / (union[:, None] - (~scenarios[:, on]))
            scores = dropped.mean(axis=0)
            pick = int(np.argmax(scores))
            best = _prefer(best, float(scores[pick]), int(on[pick]), 'drop')
        if best is None or best[0] <= current + 1e-9:
            break
        chosen[best[1]] = best[2] == 'add'
    return chosen


def seed_footprint_decode(seed_probability, marginal, offsets, footprint, n_scenarios=500, seed=606):
    """Sample a seed and directed footprint, reporting any changed marginals."""
    report = joint_marginal_report(seed_probability, marginal, offsets, footprint)
    if report['joint'] == 'independent_fallback':
        return greedy_expected_iou(marginal, n_scenarios, seed), report
    inclusion = {int(k): v for k, v in footprint['inclusion_by_offset'].items()}
    conditional = conditional_from_offsets(offsets, inclusion, len(marginal))
    conditional, _, _ = match_marginals(seed_probability, conditional, marginal)
    rng = np.random.default_rng(seed)
    pi = np.asarray(seed_probability, float)
    pi = pi/pi.sum()
    seeds = rng.choice(len(pi), size=n_scenarios, p=pi)
    scenarios = rng.random((n_scenarios, len(pi))) < conditional[seeds]
    return optimize_scenario_iou(scenarios, marginal), report


def _prefer(best, score, index, kind):
    if best is None or score > best[0] + 1e-15 or (abs(score - best[0]) <= 1e-15 and index < best[1]):
        return (score, index, kind)
    return best


def _scenario_iou(scenarios, chosen):
    intersection = np.logical_and(scenarios, chosen).sum(axis=1).astype(float)
    union = np.logical_or(scenarios, chosen).sum(axis=1).astype(float)
    return float((intersection / union).mean()), intersection, union


def scenario_decode(probability, condition, n_scenarios=500, seed=2026):
    """Decode one window. Onset returns are empty on T+5..T+25 even if those scores are not."""
    probability = np.asarray(probability, float)
    if condition == 'queue_onset':
        if probability.ndim != 2 or probability.shape[0] != 6:
            raise ValueError('Onset probabilities must have shape (6, links)')
        chosen = greedy_expected_iou(probability[-1], n_scenarios, seed)
        output = np.zeros(probability.shape, dtype=bool)
        output[-1] = chosen
        return output
    if probability.ndim != 2:
        raise ValueError('Ongoing probabilities must have shape (horizons, links)')
    flat = greedy_expected_iou(probability.ravel(), n_scenarios, seed)
    return flat.reshape(probability.shape)


def surrogate_decode(probability, condition):
    if condition == 'queue_onset':
        return decode_onset(probability, 'expected_iou')
    return decode_queue(probability, 'expected_iou')


def apply_decoder(probability, condition, policy, n_scenarios=500, seed=2026):
    if policy == 'scenario':
        return scenario_decode(probability, condition, n_scenarios, seed)
    if condition == 'queue_onset':
        return decode_onset(probability, policy)
    return decode_queue(probability, policy)


def decoder_pair(probability, truth, valid, condition, n_scenarios=500, seed=2026):
    """Surrogate versus scenario decoder on one shared probability array."""
    surrogate = surrogate_decode(probability, condition)
    scenario = scenario_decode(probability, condition, n_scenarios, seed)
    return {
        'surrogate_iou': proxy_queue_iou(truth, surrogate, valid),
        'scenario_iou': proxy_queue_iou(truth, scenario, valid),
        'sets_differ': bool(np.any(surrogate != scenario)),
        'identical_probabilities': True,
        'scenario_joint': 'independent_bernoulli_marginals',
    }


def scaled_costs(probability, threshold, beta, scale):
    unary = np.rint((float(threshold) - np.asarray(probability, float).reshape(-1)) * scale).astype(np.int64)
    beta_i = int(np.rint(float(beta) * scale))
    if beta_i < 0 or scale <= 0:
        raise ValueError('Graph-cut scale and beta must be nonnegative')
    return unary, beta_i


def integer_energy(labels, unary, pairs, beta_i):
    labels = np.asarray(labels, bool)
    energy = int(np.sum(np.asarray(unary, np.int64) * labels))
    for i, j in pairs:
        if bool(labels[i]) != bool(labels[j]):
            energy += int(beta_i)
    return energy


def brute_min_energy(unary, pairs, beta_i):
    unary = np.asarray(unary, np.int64)
    n = unary.size
    if n > 12:
        raise ValueError('Energy enumeration supports at most 12 nodes')
    best_energy, best_mask = None, 0
    for mask in range(1 << n):
        labels = np.array([(mask >> i) & 1 for i in range(n)], dtype=bool)
        energy = integer_energy(labels, unary, pairs, beta_i)
        if best_energy is None or energy < best_energy or (energy == best_energy and mask < best_mask):
            best_energy, best_mask = energy, mask
    labels = np.array([(best_mask >> i) & 1 for i in range(n)], dtype=bool)
    return labels, best_energy


def graph_cut(probability, pairs, threshold=0.5, beta=0.25, scale=1_000_000):
    """Minimize sum (threshold-p)*x + beta * |x-x'| with an integer min cut.

    Unary (threshold-p) is negative when p exceeds the threshold, which favors
    label 1. The source side of the min cut is label 1. A negative unary -c
    becomes a source-to-node capacity c: putting the node on the sink side
    (label 0) pays c. A positive unary c becomes a node-to-sink capacity c.
    scipy.maximum_flow stores a skew-symmetric integer flow; residual capacity
    is capacity minus that flow. Source-side nodes are those reachable from the
    source on strictly positive residual edges.
    """
    unary, beta_i = scaled_costs(probability, threshold, beta, scale)
    n = unary.size
    source, sink = n, n + 1
    capacity = {}

    def add(a, b, weight):
        weight = int(weight)
        if weight < 0:
            raise ValueError('Min-cut capacities must be nonnegative')
        if weight:
            capacity[(a, b)] = capacity.get((a, b), 0) + weight

    for node, cost in enumerate(unary):
        if cost < 0:
            add(source, node, -int(cost))
        elif cost > 0:
            add(node, sink, int(cost))
    for i, j in pairs:
        add(int(i), int(j), beta_i)
        add(int(j), int(i), beta_i)
    labels = np.zeros(n, dtype=bool)
    if not capacity:
        return labels
    rows, cols, data = zip(*((a, b, w) for (a, b), w in capacity.items()))
    graph = csr_matrix((np.asarray(data, np.int64), (rows, cols)), shape=(n + 2, n + 2))
    flow = maximum_flow(graph, source, sink).flow
    residual = (graph - flow).toarray()
    reachable = np.zeros(n + 2, dtype=bool)
    stack = [source]
    reachable[source] = True
    while stack:
        node = stack.pop()
        for nxt in np.flatnonzero(residual[node] > 0):
            if not reachable[nxt]:
                reachable[nxt] = True
                stack.append(int(nxt))
    labels[:] = reachable[:n]
    return labels


def space_time_pairs(adjacency, horizons, links):
    spatial = undirected_pairs(adjacency)
    pairs = []
    for horizon in range(horizons):
        base = horizon * links
        pairs.extend((base + i, base + j) for i, j in spatial)
    for horizon in range(horizons - 1):
        base = horizon * links
        pairs.extend((base + i, base + links + i) for i in range(links))
    return pairs


def graph_cut_queue(probability, adjacency, threshold=0.5, beta=0.25, scale=1_000_000):
    probability = np.asarray(probability, float)
    if probability.ndim == 1:
        probability = probability[None]
    horizons, links = probability.shape
    labels = graph_cut(probability.ravel(), space_time_pairs(adjacency, horizons, links), threshold, beta, scale)
    return labels.reshape(horizons, links)


def isolation_share(prediction, adjacency):
    """A positive cell is isolated when no spatial or adjacent-horizon positive supports it."""
    prediction = np.asarray(prediction, bool)
    if prediction.ndim == 1:
        prediction = prediction[None]
    undirected = (np.asarray(adjacency) + np.asarray(adjacency).T) > 0
    np.fill_diagonal(undirected, False)
    horizons, links = prediction.shape
    support = np.zeros_like(prediction)
    for horizon in range(horizons):
        support[horizon] = undirected @ prediction[horizon].astype(float) > 0
    if horizons > 1:
        support[1:] |= prediction[:-1]
        support[:-1] |= prediction[1:]
    positive = prediction
    isolated = positive & ~support
    count = int(positive.sum())
    return {
        'positives': count,
        'isolated': int(isolated.sum()),
        'isolated_share': None if count == 0 else float(isolated.sum() / count),
    }


def flip_rates(raw, denoised, valid):
    """Per-horizon flip share and union flip rate on scored cells.

    The union rate uses cells where either label is queued, so a large free-flow
    agreement cannot shrink the disagreement.
    """
    raw = np.asarray(raw, bool)
    denoised = np.asarray(denoised, bool)
    valid = np.asarray(valid, bool)
    rows = []
    for horizon in range(raw.shape[0]):
        mask = valid[horizon]
        flip = (raw[horizon] != denoised[horizon]) & mask
        union = ((raw[horizon] | denoised[horizon]) & mask)
        rows.append({
            'horizon': horizon + 1,
            'scored_cells': int(mask.sum()),
            'flips': int(flip.sum()),
            'flip_share': float(flip.sum() / mask.sum()) if mask.any() else None,
            'union_cells': int(union.sum()),
            'union_flip_rate': float(flip.sum() / union.sum()) if union.any() else None,
        })
    return rows


def persist_latest(normalized, threshold=QUEUE_RATIO):
    latest = np.zeros(normalized.shape[1], dtype=bool)
    for row in np.asarray(normalized, float):
        seen = np.isfinite(row)
        latest[seen] = row[seen] <= threshold
    return np.repeat(latest[None], 6, axis=0)


def history_queue_report(normalized, filtered_level, threshold=QUEUE_RATIO):
    """Crossings inside the released T-60..T-5 rows. Not a false-positive estimate.

    The official onset condition is defined on latent history. This function only
    describes the 12 released input rows of one window.
    """
    series = np.asarray(normalized, float)
    level = np.asarray(filtered_level, float)
    if len(series) != 13 or len(level) != 13:
        raise ValueError('History diagnostics expect the 13 released rows')
    history = series[:12]
    filtered = level[:12, :, 0] if level.ndim == 3 else level[:12]
    raw_queue = np.isfinite(history) & (history <= threshold)
    filtered_queue = filtered <= threshold
    crossings = int(np.sum((filtered[1:] - threshold) * (filtered[:-1] - threshold) < 0))
    return {
        'released_history_raw_queue': bool(raw_queue.any()),
        'released_history_filtered_queue': bool(filtered_queue.any()),
        'filtered_threshold_crossings': crossings,
        'raw_filtered_disagreement': bool(raw_queue.any()) != bool(filtered_queue.any()),
        'origin_row_raw_queue': bool(np.any(np.isfinite(series[12]) & (series[12] <= threshold))),
        'not_a_false_positive_estimate': True,
        'inputs': 'released T-60..T window only',
    }


def missed_false_by_horizon(truth, prediction, valid):
    rows = []
    truth = np.asarray(truth, bool)
    prediction = np.asarray(prediction, bool)
    valid = np.asarray(valid, bool)
    for horizon in range(truth.shape[0]):
        mask = valid[horizon]
        missed = truth[horizon] & ~prediction[horizon] & mask
        false = ~truth[horizon] & prediction[horizon] & mask
        rows.append({
            'horizon': horizon + 1,
            'missed': int(missed.sum()),
            'false': int(false.sum()),
            'truth_positive': int((truth[horizon] & mask).sum()),
            'pred_positive': int((prediction[horizon] & mask).sum()),
        })
    return rows


def size_oracle(marginal, truth, valid):
    """Top-k by marginal probability, with k the true positive count. Not deployable."""
    truth = np.asarray(truth, bool)
    valid = np.asarray(valid, bool)
    positive = truth & valid
    k = int(positive.sum())
    scores = np.asarray(marginal, float).copy()
    scores[~valid] = -np.inf
    order = np.argsort(-scores, kind='mergesort')
    prediction = np.zeros_like(truth)
    if k:
        prediction[order[:k]] = True
    return {
        'k': k,
        'iou': proxy_queue_iou(truth, prediction, valid),
        'unique_seed_defined': k == 1,
        'limitation': 'Knows the true positive count. Undefined as a unique seed when the count is not 1.',
    }


def set_mass_nll(seed_probability, positive):
    """Negative log mass on the whole positive set. Not a unique-seed likelihood."""
    mass = float(np.asarray(seed_probability, float)[np.asarray(positive, bool)].sum())
    return float(-np.log(max(mass, 1e-12)))


def topk_set_hit(seed_probability, positive, k):
    positive = np.asarray(positive, bool)
    order = np.argsort(-np.asarray(seed_probability, float), kind='mergesort')[:k]
    return bool(positive[order].any()) if positive.any() else False


def softmax_temperature(scores, temperature, valid):
    scores = np.asarray(scores, float)
    valid = np.asarray(valid, bool)
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    if not valid.any():
        return np.zeros(scores.shape)
    shifted = np.full(scores.shape, -np.inf)
    shifted[valid] = scores[valid] / float(temperature)
    shifted[valid] -= np.max(shifted[valid])
    weights = np.exp(shifted)
    weights[~valid] = 0
    total = weights.sum()
    if total <= 0:
        output = np.zeros(scores.shape)
        output[valid] = 1 / valid.sum()
        return output
    return weights / total


def aggregate_macro(rows, key):
    """Equal weight per condition, then per direction (panel), then per family."""
    if not rows:
        return {'overall': None, 'family': {}, 'panel': {}, 'panel_condition': {}}
    buckets = {}
    for row in rows:
        buckets.setdefault((row['panel'], row['condition']), []).append(row[key])
    panel_condition = {k: float(np.mean(v)) for k, v in buckets.items()}
    panels = {}
    for (panel, _), score in panel_condition.items():
        panels.setdefault(panel, []).append(score)
    panel_score = {panel: float(np.mean(v)) for panel, v in panels.items()}
    family_of = {row['panel']: row['family'] for row in rows}
    families = {}
    for panel, score in panel_score.items():
        families.setdefault(family_of[panel], []).append(score)
    family_score = {name: float(np.mean(v)) for name, v in families.items()}
    overall = float(np.mean(list(family_score.values())))
    return {'overall': overall, 'family': family_score, 'panel': panel_score,
            'panel_condition': {f'{a}|{b}': v for (a, b), v in panel_condition.items()}}


def paired_day_block_bootstrap(rows, candidate, baseline, n_boot=400, seed=606):
    """Paired day bootstrap stratified by panel and condition.

    Both scores ride the same draw. The result is not a standard error of an
    absolute queue score and is not the uncertainty of the fixed public windows.
    """
    point = aggregate_macro(rows, candidate)['overall']
    base = aggregate_macro(rows, baseline)['overall']
    if point is None or base is None:
        return {'delta_point': None, 'delta_p05': None, 'delta_p95': None, 'n_boot': 0, 'n_windows': 0}
    by_panel = {}
    for index, row in enumerate(rows):
        by_panel.setdefault((row['panel'], row['condition']), {}).setdefault(row['day'], []).append(index)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for draw in range(n_boot):
        chosen = []
        for days in by_panel.values():
            keys = list(days)
            picks = rng.choice(len(keys), size=len(keys), replace=True)
            for pick in picks:
                chosen.extend(days[keys[int(pick)]])
        sample = [rows[i] for i in chosen]
        deltas[draw] = aggregate_macro(sample, candidate)['overall'] - aggregate_macro(sample, baseline)['overall']
    return {
        'delta_point': float(point - base),
        'delta_p05': float(np.quantile(deltas, 0.05)),
        'delta_p95': float(np.quantile(deltas, 0.95)),
        'n_boot': int(n_boot),
        'n_windows': len(rows),
        'weights': 'equal conditions, equal panels within family, equal families; fixed across replicates',
        'resampling': 'days within panel/condition strata; cross-condition day dependence is not retained',
        'interpretation': (
            'Percentiles of the paired day-block candidate-minus-baseline difference on this '
            'proxy sample. Not a standard error of an absolute score, and not uncertainty of '
            'the deterministic public 8-corridor by 10-window value.'
        ),
    }


def window_sample_spread(values):
    values = np.asarray(list(values), float)
    if not len(values):
        return {'n': 0, 'mean': None, 'sd': None}
    return {
        'n': int(len(values)),
        'mean': float(values.mean()),
        'sd': None if len(values) < 2 else float(values.std(ddof=1)),
    }


class OnsetSeedModel:
    """Grouped ranker. Multiple positive links stay multiple positive labels."""

    def __init__(self, policy='origin_observed', rounds=40, min_groups=8, folds=5, seed=606, min_child_samples=10):
        self.policy = policy
        self.rounds = rounds
        self.min_groups = min_groups
        self.folds = folds
        self.seed = seed
        self.min_child_samples = min_child_samples
        self.status = 'unfit'
        self.prior_status = 'unfit'
        self.temperature = 1.0
        self.prior = None
        self.model = None
        self.footprint = None
        self.max_positives_in_group = 0
        self.skipped_groups = 0
        self.fitted_groups = 0

    def fit(self, events, up, down, params, noise, adjacency):
        events = [event for event in events if event.condition == 'queue_onset']
        events = sorted(events, key=lambda event: event.origin)
        masks = [(event.truth[-1] & event.valid[-1]) for event in events]
        self.footprint = footprint_summary(masks, adjacency, min_components=self.min_groups)
        n = len(params)
        if len(events) < self.min_groups:
            self.status = 'skipped_small_sample'
            self.prior_status = 'skipped_small_sample'
            self.prior = np.full(n, 1 / n)
            return self
        fold = np.zeros(len(events), dtype=int)
        blocks = np.array_split(np.arange(len(events)), min(self.folds, len(events)))
        for index, block in enumerate(blocks):
            fold[block] = index
        truths = [event.truth[-1] & event.valid[-1] for event in events]
        valids = [event.valid[-1] for event in events]
        self.prior = _frequency(truths, valids)
        self.prior_status = 'out_of_fold_training_full_sample_at_evaluation'
        oof = []
        for index, event in enumerate(events):
            keep = np.flatnonzero(fold != fold[index])
            if len(keep) == 0:
                oof.append(self.prior.copy())
            else:
                oof.append(_frequency([truths[i] for i in keep], [valids[i] for i in keep]))
        xx, yy, groups = [], [], []
        self.max_positives_in_group = 0
        self.skipped_groups = 0
        for event, prior in zip(events, oof):
            history, auxiliary = visible_event(event, self.policy)
            features = queue_v6_features(history, auxiliary, event.times, up, down, params, noise)[-n:]
            features = np.concatenate([features, prior[:, None].astype(np.float32)], axis=1)
            valid = event.valid[-1]
            label = (event.truth[-1] & valid).astype(int)
            features, label = features[valid], label[valid]
            positives = int(label.sum())
            if positives == 0 or positives == len(label):
                self.skipped_groups += 1
                continue
            self.max_positives_in_group = max(self.max_positives_in_group, positives)
            xx.append(features)
            yy.append(label)
            groups.append(len(label))
        self.fitted_groups = len(groups)
        if self.fitted_groups < self.min_groups:
            self.status = 'skipped_small_sample'
            return self
        self.model = LGBMRanker(objective='rank_xendcg', n_estimators=self.rounds, learning_rate=.05,
            num_leaves=15, min_child_samples=self.min_child_samples, label_gain=[0, 1],
            colsample_bytree=.9, n_jobs=1, verbosity=-1, random_state=self.seed)
        self.model.fit(np.concatenate(xx), np.concatenate(yy), group=groups)
        self.status = 'fitted'
        return self

    def seed_distribution(self, history, auxiliary, times, up, down, params, noise, valid, temperature=None):
        n = len(valid)
        temperature = self.temperature if temperature is None else temperature
        if self.status != 'fitted':
            probability = np.zeros(n)
            if np.asarray(valid, bool).any():
                probability[np.asarray(valid, bool)] = 1 / np.asarray(valid, bool).sum()
            return probability
        features = queue_v6_features(history, auxiliary, times, up, down, params, noise)[-n:]
        features = np.concatenate([features, self.prior[:, None].astype(np.float32)], axis=1)
        scores = np.asarray(self.model.booster_.predict(features, num_threads=1), float)
        return softmax_temperature(scores, temperature, valid)

    def probabilities(self, event, up, down, params, noise):
        """(6, links) categorical seed distribution. The first five horizons stay zero."""
        history, auxiliary = visible_event(event, self.policy)
        output = np.zeros((6, len(params)))
        output[-1] = self.seed_distribution(
            history, auxiliary, event.times, up, down, params, noise, event.valid[-1])
        return output

    def select_temperature(self, events, up, down, params, noise):
        """October events only. Later months raise rather than entering the grid."""
        if self.status != 'fitted':
            return {'status': self.status, 'temperature': None, 'robust': False}
        for event in events:
            stamp = pd.Timestamp(event.times[-1])
            if stamp.month != 10:
                raise ValueError('Temperature is selected on October only')
        records = []
        for temperature in TEMPERATURES:
            nlls, hit1, hit3, multi = [], [], [], 0
            for event in events:
                if event.condition != 'queue_onset':
                    continue
                history, auxiliary = visible_event(event, self.policy)
                probability = self.seed_distribution(
                    history, auxiliary, event.times, up, down, params, noise, event.valid[-1], temperature)
                positive = event.truth[-1] & event.valid[-1]
                if int(positive.sum()) > 1:
                    multi += 1
                nlls.append(set_mass_nll(probability, positive))
                hit1.append(topk_set_hit(probability, positive, 1))
                hit3.append(topk_set_hit(probability, positive, 3))
            if not nlls:
                continue
            records.append({
                'temperature': temperature,
                'set_mass_nll': float(np.mean(nlls)),
                'top1_set_hit': float(np.mean(hit1)),
                'top3_set_hit': float(np.mean(hit3)),
                'windows': len(nlls),
                'multiple_positive_windows': multi,
            })
        if not records:
            return {'status': 'no_october_onset_windows', 'temperature': None, 'robust': False}
        best = min(records, key=lambda row: (row['set_mass_nll'], row['temperature']))
        self.temperature = best['temperature']
        return {
            'status': 'selected_on_october_set_mass_nll',
            'temperature': self.temperature,
            'grid': records,
            'unique_seed_nll': None,
            'robust': True,
            'note': 'set_mass_nll is the log loss of the positive set, not a single invented seed.',
        }


def _frequency(truths, valids):
    if not truths:
        raise ValueError('Frequency prior received no windows')
    pos = np.sum([np.asarray(t, float) for t in truths], axis=0)
    den = np.sum([np.asarray(v, float) for v in valids], axis=0)
    return (pos + 0.1) / (den + 1)


def joint_marginal_report(seed_probability, marginal, offsets, footprint):
    """Say whether a seed-conditional footprint reproduces the marginal vector."""
    marginal = np.asarray(marginal, float)
    pi = np.asarray(seed_probability, float)
    if footprint is None or footprint.get('status') != 'empirical':
        reason = None if footprint is None else footprint.get('status')
        return {
            'marginals_reproduced': True,
            'l1_gap': 0.0,
            'joint': 'independent_fallback',
            'reason': reason or 'no_footprint',
            'note': 'No footprint was fitted. Scenario decoding stays on the independent marginals.',
        }
    inclusion = {int(k): v for k, v in footprint['inclusion_by_offset'].items()}
    conditional = conditional_from_offsets(offsets, inclusion, len(marginal))
    _, gap, matched = match_marginals(pi, conditional, marginal)
    return {
        'marginals_reproduced': matched,
        'l1_gap': gap,
        'joint': 'seed_plus_directed_footprint' if matched else 'changed_joint_model',
        'reason': None if matched else 'mixture could not reproduce the marginals within tolerance',
        'note': 'Primary scenario-versus-surrogate comparison always uses the independent marginals.',
    }


def panel_identity(panel):
    return {'panel': panel, 'family': family(panel), 'direction': panel.rsplit('_', 1)[-1]}
