# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 16:22:37 2025

@author: sande
"""

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mcap.reader import make_reader


TOP_EPISODE = "/rl/episode"
TOP_METRICS = "/rl/metrics"
TOP_TRANSITION = "/rl/transition"


def _safe_float(x: Any):
    try:
        return float(x)
    except Exception:
        return None


def read_mcap(
    mcap_path: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    episodes = []
    metrics = []
    transitions = []

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages(topics=[TOP_EPISODE, TOP_METRICS, TOP_TRANSITION]):
            try:
                d = json.loads(message.data)
            except Exception:
                continue

            if channel.topic == TOP_EPISODE:
                episodes.append(d)
            elif channel.topic == TOP_METRICS:
                metrics.append(d)
            elif channel.topic == TOP_TRANSITION:
                transitions.append(d)

    # Sort by step (robust for plotting)
    episodes.sort(key=lambda r: r.get("step", -1))
    metrics.sort(key=lambda r: r.get("step", -1))
    transitions.sort(key=lambda r: r.get("step", -1))
    return episodes, metrics, transitions


def plot_training_curves(episodes: List[Dict[str, Any]], metrics: List[Dict[str, Any]]) -> None:
    # Episode return vs step
    if len(episodes) > 0:
        steps = [int(r["step"]) for r in episodes if "step" in r]
        rets = [float(r["episode_return"]) for r in episodes if "episode_return" in r]
        if len(steps) > 0 and len(rets) > 0:
            plt.figure()
            plt.plot(steps, rets)
            plt.title("Episode return vs environment steps")
            plt.xlabel("step")
            plt.ylabel("episode_return")

    # Metrics: collect available numeric keys
    if len(metrics) > 0:
        # Build series by key
        series = defaultdict(list)
        step_series = []
        for r in metrics:
            if "step" not in r:
                continue
            s = int(r["step"])
            step_series.append(s)
            for k, v in r.items():
                if k == "step":
                    continue
                fv = _safe_float(v)
                if fv is not None:
                    series[k].append((s, fv))

        # Plot a small set of common RL metrics if present; else plot any train/* keys found.
        preferred = [
            "rollout/ep_rew_mean",
            "train/actor_loss",
            "train/critic_loss",
            "train/ent_coef",
            "train/ent_coef_loss",
            "train/n_updates",
            "time/fps",
        ]

        keys_to_plot = [k for k in preferred if k in series]
        if len(keys_to_plot) == 0:
            keys_to_plot = sorted([k for k in series.keys() if k.startswith(("train/", "rollout/", "time/"))])[:6]

        for k in keys_to_plot:
            pts = series[k]
            pts.sort(key=lambda t: t[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.figure()
            plt.plot(xs, ys)
            plt.title(k)
            plt.xlabel("step")
            plt.ylabel(k)


def plot_state_trajectory(
    transitions: List[Dict[str, Any]],
    x_idx: int,
    y_idx: int,
    use_next_state: bool = False,
) -> None:
    if len(transitions) == 0:
        return

    key = "next_state" if use_next_state else "state"
    xs, ys, steps = [], [], []

    for r in transitions:
        if key not in r:
            continue
        st = r[key]
        if st is None:
            continue
        try:
            st = np.asarray(st, dtype=float)
        except Exception:
            continue

        # handle potential shape (dim,) or (1, dim)
        if st.ndim == 2 and st.shape[0] == 1:
            st = st[0]

        if st.ndim != 1:
            continue
        if max(x_idx, y_idx) >= st.shape[0]:
            continue

        xs.append(st[x_idx])
        ys.append(st[y_idx])
        steps.append(int(r.get("step", -1)))

    if len(xs) == 0:
        return

    plt.figure()
    plt.plot(xs, ys)
    plt.axis("equal")
    plt.title(f"State trajectory from {TOP_TRANSITION} ({key}[{x_idx}], {key}[{y_idx}])")
    plt.xlabel(f"{key}[{x_idx}]")
    plt.ylabel(f"{key}[{y_idx}]")


def plot_state_components_vs_step(
    transitions: List[Dict[str, Any]],
    dims_to_plot: List[int],
    use_next_state: bool = False,
) -> None:
    if len(transitions) == 0:
        return

    key = "next_state" if use_next_state else "state"
    steps = []
    comps = {d: [] for d in dims_to_plot}

    for r in transitions:
        if "step" not in r or key not in r:
            continue
        try:
            st = np.asarray(r[key], dtype=float)
        except Exception:
            continue

        if st.ndim == 2 and st.shape[0] == 1:
            st = st[0]
        if st.ndim != 1:
            continue

        s = int(r["step"])
        steps.append(s)
        for d in dims_to_plot:
            comps[d].append(st[d] if d < st.shape[0] else np.nan)

    if len(steps) == 0:
        return

    # sort by step
    order = np.argsort(np.asarray(steps))
    steps = np.asarray(steps)[order]
    plt.figure()
    for d in dims_to_plot:
        y = np.asarray(comps[d], dtype=float)[order]
        plt.plot(steps, y, label=f"{key}[{d}]")
    plt.title(f"Selected state components vs step ({key})")
    plt.xlabel("step")
    plt.ylabel("state value")
    plt.legend()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", default="sac_training_full.mcap", help="Path to MCAP file (e.g., sac_training_full.mcap)")
    ap.add_argument("--x_idx", type=int, default=0, help="State dimension index for x-axis (trajectory plot)")
    ap.add_argument("--y_idx", type=int, default=1, help="State dimension index for y-axis (trajectory plot)")
    ap.add_argument("--use_next_state", action="store_true", help="Plot next_state instead of state")
    ap.add_argument("--state_dims", type=str, default="0,1,2", help="Comma-separated state dims to plot vs step")
    args = ap.parse_args()

    episodes, metrics, transitions = read_mcap(args.mcap)

    print(f"Loaded from {args.mcap}")
    print(f"  episodes:     {len(episodes)} messages")
    print(f"  metrics:      {len(metrics)} messages")
    print(f"  transitions:  {len(transitions)} messages")

    plot_training_curves(episodes, metrics)

    # Trajectory in (x_idx, y_idx)
    plot_state_trajectory(
        transitions,
        x_idx=args.x_idx,
        y_idx=args.y_idx,
        use_next_state=args.use_next_state,
    )

    # Selected components vs step
    dims = []
    for tok in args.state_dims.split(","):
        tok = tok.strip()
        if tok != "":
            dims.append(int(tok))
    plot_state_components_vs_step(
        transitions,
        dims_to_plot=dims,
        use_next_state=args.use_next_state,
    )

    plt.show()


if __name__ == "__main__":
    main()
