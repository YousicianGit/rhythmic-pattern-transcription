import logging

import mir_eval
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def process_downbeats(
    downbeat_times: NDArray[np.float64],
    *,
    max_previous_distance: int = 3,
    tie_tolerance: float = 0.0,
    inlier_tolerance: float = 0.1,
    strides: tuple[int, ...] = (1, 2, 4),
) -> NDArray[np.float64]:
    half_median_downbeats = enforce_half_median_bar_length(
        downbeat_times,
        max_previous_distance=max_previous_distance,
        tie_tolerance=tie_tolerance,
    )
    downbeat_times = refine_metrical_level(
        half_median_downbeats,
        reference_onsets=downbeat_times,
        window_tolerance=inlier_tolerance,
        strides=strides,
    )
    return downbeat_times


def enforce_half_median_bar_length(
    downbeat_times: NDArray[np.float64],
    *,
    max_previous_distance: int = 3,
    tie_tolerance: float = 0.0,
) -> NDArray[np.float64]:
    """
    Clean up a sequence of estimated downbeat times, to make it consistent and remove noisy data points.
    Using the Viterbi algorithm, we try to find a downbeat sequence that 1) has 0.5 * median(diff(downbeat_times))
    bar lengths as closely as possibly and 2) matches as many of the input downbeat times as possible.
    """
    times = np.asarray(downbeat_times, dtype=np.float64, copy=True)
    n = times.size
    if n == 0:
        return times

    start_i = n % 2
    diffs = np.diff(times[start_i:])
    if diffs.size == 0:
        logger.warning("Need at least 3 downbeats for processing; returning input unchanged.")
        return times

    half_median_bar_length = 0.5 * np.median(diffs).item()
    if half_median_bar_length <= 0.0:
        logger.warning("Non-positive half median bar length computed; returning input unchanged.")
        return times

    max_cost = 1e3
    # Dynamic Programming [DP] arrays:
    # - dp_cost[i] = min total cost to end at i over any path length.
    # - dp_len[i]  = number of nodes in that best path (for tie-break: prefer shorter).
    # - back_prev[i] = predecessor index for backtracking.
    dp_cost = np.full(n, max_cost, dtype=np.float64)
    dp_len = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    back_prev = np.full(n, -1, dtype=np.int32)

    # DP base case
    dp_cost[0] = 0.0
    dp_len[0] = 1
    back_prev[0] = -1

    # Forward pass
    for current in range(1, n):
        best_cost = max_cost
        best_len = np.iinfo(np.int32).max
        best_prev = -1

        # Try all predecessors within max_previous_distance
        max_d = min(max_previous_distance, current)
        for d in range(1, max_d + 1):
            prev = current - d
            base = dp_cost[prev]

            interval = times[current] - times[prev]
            num_cycles = int(np.maximum(1, np.round(interval / half_median_bar_length)))
            error = interval - num_cycles * half_median_bar_length

            # First and last downbeat are sometimes wrong. As we require including them, they may distort also
            # neighbouring downbeats, unless we practically disregard the lengths of the first and the last bar.
            cost_multiplier = 1.0
            if prev == 0 or current == n - 1:
                if d == 1:
                    cost_multiplier = 0.01
                elif d == 2:
                    cost_multiplier = 0.5
                else:
                    cost_multiplier = 0.67

            candidate_cost = base + cost_multiplier * error**2
            candidate_len = dp_len[prev] + 1

            # Select best predecessor
            if candidate_cost < best_cost - tie_tolerance:
                best_cost, best_len, best_prev = candidate_cost, candidate_len, prev
            elif abs(candidate_cost - best_cost) <= tie_tolerance and candidate_len < best_len:
                # Tie on cost: prefer fewer steps
                best_cost, best_len, best_prev = candidate_cost, candidate_len, prev

        dp_cost[current] = best_cost
        dp_len[current] = best_len
        back_prev[current] = best_prev

    # Backtrack pass
    index = n - 1
    downbeats = []
    while index > 0:
        prev = back_prev[index]
        if prev < 0:
            raise ValueError(f"Unexpected best previous distance indices {prev} at index {index}, should be >= 0")
        interval = times[index] - times[prev]
        num_cycles = int(np.maximum(1, np.round(interval / half_median_bar_length)))
        for cycle in range(num_cycles):
            downbeats.append(times[index] - cycle * interval / num_cycles)
        index = prev

    downbeats.append(times[0].item())  # first downbeat must be included
    return np.asarray(downbeats)[::-1]  # reverse the order to get back to ascending time


def refine_metrical_level(
    estimated_onsets: NDArray[np.float64],
    *,
    reference_onsets: NDArray[np.float64],
    window_tolerance: float = 0.05,
    strides: tuple[int, ...] = (1, 2, 4),
) -> NDArray[np.float64]:
    """
    Refine metrical level of candidate onsets by comparing to reference onsets.
    This is done by trying different downsampling factors (and a beginning onsets)
    and choosing the one that maximizes the F-measure calculated by
    comparing the `candidate_onsets` against the given input `reference_onsets`.

    Such procedure can help with fixing these two common problems:
    1. Wrong metrical level: when candidate onsets are kind of correct but either upsampled or downsampled.
    2. "Pi-phase" errors: the estimated onsets are exactly 180° wrong,
        so that they are at the middle of the reference onsets
    """
    strides = tuple(sorted(set(strides)))
    if strides[0] < 1:
        raise ValueError("strides must be positive integers")

    candidate = np.asarray(estimated_onsets, dtype=np.float64, copy=True)
    reference = np.asarray(reference_onsets, dtype=np.float64, copy=True)

    if candidate.size == 0:
        return candidate

    first_onset = candidate[0]
    last_onset = candidate[-1]

    best_f1 = -np.inf
    best = candidate
    for stride in strides:
        for phase in range(stride):
            onsets = candidate[phase % stride :: stride]
            if onsets.size == 0:
                continue

            f1 = mir_eval.onset.evaluate(
                reference_onsets=reference,
                estimated_onsets=onsets,
                window=window_tolerance,
            )["F-measure"]
            if f1 >= best_f1:
                best_f1 = f1
                best = onsets

    # ensure that the original temporal range is covered
    if best[0] > first_onset:
        best = np.concatenate(([first_onset], best))
    if best[-1] < last_onset:
        best = np.concatenate((best, [last_onset]))

    return best
