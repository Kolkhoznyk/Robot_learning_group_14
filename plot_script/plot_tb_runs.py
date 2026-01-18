#!/usr/bin/env python3
import argparse
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


def find_event_file(logdir: str) -> str:
    candidates = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file found under: {logdir}")
    # take latest by mtime
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def load_scalars(logdir: str, tags: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return dict tag -> (steps, values)."""
    event_file = find_event_file(logdir)
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={
            event_accumulator.SCALARS: 0,
        },
    )
    ea.Reload()

    out = {}
    available = set(ea.Tags().get("scalars", []))
    for tag in tags:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.int64)
        vals = np.array([e.value for e in events], dtype=np.float64)
        out[tag] = (steps, vals)
    return out


def group_name_from_path(path: str) -> str:
    """Heuristic: strip '_seedXX...' suffix if present."""
    base = os.path.basename(os.path.dirname(os.path.normpath(path)))
    base = re.sub(r"_seed\d+.*$", "", base)
    return base


def interp_to_grid(steps: np.ndarray, vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # Need strictly increasing for np.interp
    order = np.argsort(steps)
    x = steps[order]
    y = vals[order]
    # Remove duplicate steps
    uniq_x, idx = np.unique(x, return_index=True)
    uniq_y = y[idx]
    return np.interp(grid, uniq_x, uniq_y)


def aggregate_curves(curves: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray):
    ys = []
    for (s, v) in curves:
        ys.append(interp_to_grid(s, v, grid))
    Y = np.stack(ys, axis=0)  # (n_runs, n_grid)
    mean = Y.mean(axis=0)
    sem = Y.std(axis=0, ddof=1) / np.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros_like(mean)
    ci95 = 1.96 * sem
    return mean, ci95


def plot_mean_with_ci(ax, grid, mean, ci, label):
    ax.plot(grid, mean, label=label)
    ax.fill_between(grid, mean - ci, mean + ci, alpha=0.2)

def nearest_interp_on_grid(steps, vals, grid):
    """Interpolate curve onto provided grid. Returns np.ndarray same len as grid."""
    return interp_to_grid(steps, vals, grid)

def aggregate_group_tag(g_runs, tag):
    """Return (grid, mean, ci) for a group for a given tag, or None if missing."""
    curves = []
    all_steps = []
    for _, data in g_runs:
        if tag not in data:
            continue
        s, v = data[tag]
        curves.append((s, v))
        all_steps.append(s)
    if not curves:
        return None
    grid = np.unique(np.concatenate(all_steps))
    mean, ci = aggregate_curves(curves, grid)
    return grid, mean, ci

def align_two_series(grid_a, y_a, grid_b, y_b):
    """Align two mean curves by interpolating both onto a common grid (intersection)."""
    # safest: use intersection grid range
    lo = max(grid_a.min(), grid_b.min())
    hi = min(grid_a.max(), grid_b.max())
    grid = np.unique(np.concatenate([grid_a, grid_b]))
    grid = grid[(grid >= lo) & (grid <= hi)]
    ya = np.interp(grid, grid_a, y_a)
    yb = np.interp(grid, grid_b, y_b)
    return grid, ya, yb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdirs", nargs="+", required=True,
                        help="List of run log directories (each contains events.out.tfevents.* somewhere inside).")
    parser.add_argument("--outdir", default="report_figures", help="Output directory for figures.")
    parser.add_argument("--num_tasks", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    num_tasks = args.num_tasks

    # Tags we expect
    TAG_SUCCESS_MEAN = "task/ep_success_rate_mean"
    TAG_EVAL_REWARD = "eval/mean_reward"
    TAG_CRITIC_LOSS = "train/critic_loss"
    TAG_UNIFORM_DEV = "task/sample_frac_mean_abs_dev"

    task_success_tags = [f"task/ep_success_rate_task_{k}" for k in range(num_tasks)]
    alpha_tags = [f"alpha/alpha_task_{k}" for k in range(num_tasks)]
    sample_frac_tags = [f"task/sample_frac_task_{k}" for k in range(num_tasks)]

    # Load all runs
    runs = []
    for ld in args.logdirs:
        tags = [TAG_SUCCESS_MEAN, TAG_EVAL_REWARD, TAG_CRITIC_LOSS, TAG_UNIFORM_DEV] + task_success_tags + alpha_tags + sample_frac_tags
        data = load_scalars(ld, tags)
        runs.append((ld, data))

    # Group by algorithm name from path
    groups = {}
    for ld, data in runs:
        g = group_name_from_path(ld)
        groups.setdefault(g, []).append((ld, data))

    # ---------------- Plot 1: Success mean across tasks (aggregated over seeds) ----------------
    fig, ax = plt.subplots()
    ax.set_title("Mean success across MT10 tasks")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Success rate")

    for gname, g_runs in groups.items():
        # collect curves
        curves = []
        all_steps = []
        for _, data in g_runs:
            if TAG_SUCCESS_MEAN not in data:
                continue
            s, v = data[TAG_SUCCESS_MEAN]
            curves.append((s, v))
            all_steps.append(s)
        if not curves:
            continue

        grid = np.unique(np.concatenate(all_steps))
        mean, ci = aggregate_curves(curves, grid)
        plot_mean_with_ci(ax, grid, mean, ci, label=gname)

    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "success_mean_ci.png"), dpi=200)
    plt.close(fig)

    # ---------------- Plot 1b: Delta success (disent - baseline) ----------------
    def pick_group(name_substr):
        for gname in groups.keys():
            if name_substr in gname:
                return gname
        return None

    g_base = pick_group("baseline") # wichtig, name immer richtig eingeben in train script
    g_dis  = pick_group("disent")

    if g_base is not None and g_dis is not None:
        base_stats = aggregate_group_tag(groups[g_base], TAG_SUCCESS_MEAN)
        dis_stats  = aggregate_group_tag(groups[g_dis],  TAG_SUCCESS_MEAN)

        if base_stats is not None and dis_stats is not None:
            grid_b, mean_b, _ = base_stats
            grid_d, mean_d, _ = dis_stats

            grid, mb, md = align_two_series(grid_b, mean_b, grid_d, mean_d)
            delta = md - mb

            fig, ax = plt.subplots()
            ax.set_title("Δ Mean success (disent - baseline)")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Δ Success rate")
            ax.plot(grid, delta)
            ax.axhline(0.0, linewidth=1)
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, "delta_success_mean.png"), dpi=200)
            plt.close(fig)

    # ---------------- Plot 2: Heatmap per-task success for ONE representative run per group ----------------
    # (For report: one heatmap per method; pick the first run in group.)
    for gname, g_runs in groups.items():
        ld0, data0 = g_runs[0]
        # Build common grid from tasks that exist
        steps_union = None
        task_curves = []
        for k in range(num_tasks):
            tag = f"task/ep_success_rate_task_{k}"
            if tag not in data0:
                task_curves.append(None)
                continue
            s, v = data0[tag]
            task_curves.append((s, v))
            steps_union = s if steps_union is None else np.unique(np.concatenate([steps_union, s]))

        if steps_union is None:
            continue

        grid = steps_union
        M = np.full((num_tasks, grid.shape[0]), np.nan, dtype=np.float64)
        for k in range(num_tasks):
            if task_curves[k] is None:
                continue
            s, v = task_curves[k]
            M[k, :] = interp_to_grid(s, v, grid)

        fig, ax = plt.subplots()
        im = ax.imshow(
                        M,
                        aspect="auto",
                        origin="lower",
                        extent=[grid[0], grid[-1], -0.5, num_tasks - 0.5]
                    )
        ax.set_xlabel("Timesteps")

        ax.set_ylabel("Task id")
        ax.set_yticks(np.arange(num_tasks))
        ax.set_yticklabels([f"task_{i}" for i in range(num_tasks)])
        ax.set_ylim(-0.5, num_tasks - 0.5)

        ax.set_title(f"Per-task success heatmap ({gname})\n(run: {os.path.basename(ld0)})")
        fig.colorbar(im, ax=ax, label="Success rate")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"heatmap_success_{gname}.png"), dpi=200)
        plt.close(fig)

    # ---------------- Plot 3: Stability (critic loss) + Uniform sampling deviation ----------------
    fig, ax = plt.subplots()
    ax.set_title("Critic loss (stability indicator)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Critic loss")

    for gname, g_runs in groups.items():
        curves = []
        all_steps = []
        for _, data in g_runs:
            if TAG_CRITIC_LOSS not in data:
                continue
            s, v = data[TAG_CRITIC_LOSS]
            curves.append((s, v))
            all_steps.append(s)
        if not curves:
            continue
        grid = np.unique(np.concatenate(all_steps))
        mean, ci = aggregate_curves(curves, grid)
        plot_mean_with_ci(ax, grid, mean, ci, label=gname)

    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "critic_loss_ci.png"), dpi=200)
    plt.close(fig)

    # ---------------- Plot 3b: Delta critic loss (disent - baseline) ----------------
    if g_base is not None and g_dis is not None:
        base_stats = aggregate_group_tag(groups[g_base], TAG_CRITIC_LOSS)
        dis_stats  = aggregate_group_tag(groups[g_dis],  TAG_CRITIC_LOSS)

        if base_stats is not None and dis_stats is not None:
            grid_b, mean_b, _ = base_stats
            grid_d, mean_d, _ = dis_stats

            grid, mb, md = align_two_series(grid_b, mean_b, grid_d, mean_d)
            delta = md - mb

            fig, ax = plt.subplots()
            ax.set_title("Δ Critic loss (disent - baseline)")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Δ Critic loss")
            ax.plot(grid, delta)
            ax.axhline(0.0, linewidth=1)
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, "delta_critic_loss.png"), dpi=200)
            plt.close(fig)

    fig, ax = plt.subplots()
    ax.set_title("Uniform sampling deviation (mean abs dev from 1/10)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean abs deviation")

    for gname, g_runs in groups.items():
        curves = []
        all_steps = []
        for _, data in g_runs:
            if TAG_UNIFORM_DEV not in data:
                continue
            s, v = data[TAG_UNIFORM_DEV]
            curves.append((s, v))
            all_steps.append(s)
        if not curves:
            continue
        grid = np.unique(np.concatenate(all_steps))
        mean, ci = aggregate_curves(curves, grid)
        plot_mean_with_ci(ax, grid, mean, ci, label=gname)

    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "uniform_sampling_dev_ci.png"), dpi=200)
    plt.close(fig)

    # ---------------- Plot 4: Disentangled alpha per task (one representative run per group) ----------------
    for gname, g_runs in groups.items():
        ld0, data0 = g_runs[0]
        # only if alpha tags exist
        present = [t for t in alpha_tags if t in data0]
        if not present:
            continue

        fig, ax = plt.subplots()
        ax.set_title(f"Disentangled alpha per task ({gname})\n(run: {os.path.basename(ld0)})")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("alpha")

        for k in range(num_tasks):
            tag = f"alpha/alpha_task_{k}"
            if tag not in data0:
                continue
            s, v = data0[tag]
            ax.plot(s, v, label=f"task_{k}")

        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"alpha_per_task_{gname}.png"), dpi=200)
        plt.close(fig)

    print(f"Saved figures to: {args.outdir}")


if __name__ == "__main__":
    main()
