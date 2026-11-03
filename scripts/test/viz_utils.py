#!/usr/bin/env python3
"""
Shared visualization for the evaluation scripts in this directory.

Everything here was extracted from get_reconstruction.py, where it had been
copied verbatim into two forks (get_reconstruction_multiview.py and
get_reconstruction_point_error.py) and partially into several others. There are
two families:

  * the plotly viewer - `write_recon_html`, plus the colour/symbol scheme and the
    title formatting it shares with the other HTML writers
  * the matplotlib coverage plots - `write_coverage_outputs` and the binning /
    rank-correlation primitives underneath it

Import it by bare name, the way these scripts already import `classes` and each
other; they all run with scripts/test on sys.path.

The figures follow one system: colour carries identity (which keypoint slot,
which input source, which multiview variant) and never carries it alone - every
series is also named in a legend or direct-labeled at its anchor. The matplotlib
ink tokens are deliberately separate from the plotly slot hues, because those
hues mean "keypoint slot" wherever they appear.
"""
import collections
import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


# ----------------------------
# Palette
# ----------------------------

# Keypoint identity is carried by colour (which slot) and marker symbol (which
# partial view), so the two can be read independently. Slots past the eighth reuse
# a hue, and this palette only clears the colour-blind separation floors for about
# three simultaneous slots, so every slot is also direct-labeled at its anchor -
# identity never rests on colour alone.
SLOT_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
# Scatter3d supports only this symbol set. 'x' is reserved for the GT keypoints so
# they never collide with a predicted view.
VIEW_SYMBOLS = ("circle", "square", "diamond", "circle-open", "square-open", "diamond-open", "cross")
GT_KP_SYMBOL = "x"
# Colours for the two sources making up a combined-input view, so the first and
# second partial samples read apart even after they've been merged into one cloud.
INPUT_SOURCE_COLORS = ("#1baf7a", "#9a6324")  # first sample (aqua-green), second sample (brown)
# Default reconstruction colour; a view's `recon_color` overrides it.
RECON_COLOR = "#e34948"

# matplotlib ink, kept off the plotly slot palette above: those hues carry keypoint
# identity, these carry chart roles
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
POINT_COLOR = "#2a78d6"     # per-sample scatter
SUMMARY_COLOR = "#eb6834"   # binned median
BAND_COLOR = "#f5b79c"      # interquartile band, a pale step of the summary hue

# Sequential blue ramp for density (a single hue, light to dark - never a rainbow)
DENSITY_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
DENSITY_CMAP = LinearSegmentedColormap.from_list("recon_density", DENSITY_RAMP)

# The multiview variants. Blue / orange / aqua are the first three categorical
# slots, the only three that clear the colour-blind separation floors as an
# all-pairs set, and every series is also named in the legend.
VARIANT_COLORS = {"first": POINT_COLOR, "second": SUMMARY_COLOR, "combined": "#1baf7a"}


def slot_color(k: int) -> str:
    """Colour for keypoint slot k."""
    return SLOT_COLORS[k % len(SLOT_COLORS)]


def view_symbol(v: int) -> str:
    """Marker symbol for the v-th partial view of an object."""
    return VIEW_SYMBOLS[v % len(VIEW_SYMBOLS)]


# ----------------------------
# Small helpers
# ----------------------------


def subsample(arr, max_points: int) -> np.ndarray:
    """Stride-subsample a cloud for display. Deterministic, and keeps the spread."""
    a = np.asarray(arr, dtype=np.float32)
    if max_points <= 0 or a.shape[0] <= max_points:
        return a
    step = int(np.ceil(a.shape[0] / max_points))
    return a[::step]


def as_xyz(arr) -> np.ndarray:
    """Coerce to (N,3); missing data becomes an empty (0,3) array."""
    if arr is None:
        return np.zeros((0, 3), dtype=np.float32)
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return a.reshape(-1, 3)


def keypoint_distance_stats(pred, gt) -> dict:
    """
    Per-sample keypoint agreement. DAS and mIoU are pairwise / dataset-level and
    have no single-sample value, so what is reported per sample is the symmetric
    nearest-neighbour distance between predicted and GT keypoints.
    """
    if pred is None or gt is None:
        return {}
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 3)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 3)
    if pred.size == 0 or gt.size == 0:
        return {}
    d = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=-1)
    return {
        "kp_pred_to_gt_mean": float(d.min(axis=1).mean()),
        "kp_gt_to_pred_mean": float(d.min(axis=0).mean()),
        "kp_pred_to_gt_max": float(d.min(axis=1).max()),
    }


# ----------------------------
# Title formatting
# ----------------------------

# (key, format) tables. A key the metrics dict does not carry is skipped, so one
# table serves every reconstruction script - the multiview-only first_*/second_*
# rows simply do not appear elsewhere. Scripts measuring something else entirely
# (get_mmd) pass their own table instead.
RUN_METRIC_FIELDS = (
    ("model", "{}"),
    ("category", "{}"),
    ("split", "split={}"),
    ("input_type", "input={}"),
    ("n_samples", "N={}"),
    ("coverage", "coverage={:.4f}"),
    ("cd", "CD={:.4e}"),
    ("emd", "EMD={:.4e}"),
    ("first_cd", "first CD={:.4e}"),
    ("first_emd", "first EMD={:.4e}"),
    ("second_cd", "second CD={:.4e}"),
    ("second_emd", "second EMD={:.4e}"),
    ("kvc", "KVC={:.4f}"),
    ("ukvc", "uKVC={:.4f}"),
    ("swap_gap", "swap gap={:.4f}"),
    ("fwd", "fwd={:.4f}"),
    ("bwd", "bwd={:.4f}"),
    ("das", "DAS={:.4f}"),
    ("miou_at_0_1", "mIoU@0.1={:.4f}"),
)
SAMPLE_METRIC_FIELDS = (
    ("n_views", "{:.0f} views"),
    ("coverage", "coverage={:.4f}"),
    ("cd", "CD={:.4e}"),
    ("emd", "EMD={:.4e}"),
    ("kvc", "KVC={:.4f}"),
    ("ukvc", "uKVC={:.4f}"),
    ("kp_pred_to_gt_mean", "kp p->g={:.4f}"),
    ("kp_gt_to_pred_mean", "kp g->p={:.4f}"),
)
# non-numeric parts placed before the metrics
SAMPLE_PREFIX_FIELDS = (("partial_sample_name", "{}"),)


def format_run_metrics(run_metrics: dict, fields=RUN_METRIC_FIELDS) -> str:
    """Header line with the dataset-level metrics, shared by every sample."""
    parts = []
    for key, fmt in fields:
        val = (run_metrics or {}).get(key)
        # NaN means the metric did not apply to this run, so leave it out entirely
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            continue
        try:
            parts.append(fmt.format(val))
        except (TypeError, ValueError):
            parts.append(f"{key}={val}")
    return "  |  ".join(parts)


def format_sample_title(name: str, meta: dict, fields=SAMPLE_METRIC_FIELDS,
                        prefix_fields=SAMPLE_PREFIX_FIELDS) -> str:
    """Per-sample line: object id plus whatever metrics that sample carries."""
    meta = meta or {}
    parts = [name]
    for key, fmt in prefix_fields:
        if meta.get(key):
            parts.append(fmt.format(meta[key]))
    for key, fmt in fields:
        val = meta.get(key)
        if val is None:
            continue
        with contextlib.suppress(TypeError, ValueError):
            fval = float(val)
            if np.isfinite(fval):
                parts.append(fmt.format(fval))
    return "  |  ".join(parts)


# ----------------------------
# Interactive 3D viewer
# ----------------------------


def kp_anchors(gt_kp, pred_stack, K: int):
    """
    Where the spokes for each slot start.

    The GT keypoints when they exist and share the slot ordering; otherwise the
    across-view centroid of the predictions, which is what get_kp_consistency.py
    uses when there is no ground truth to anchor to.
    """
    if gt_kp is not None:
        g = np.asarray(gt_kp, dtype=np.float32).reshape(-1, 3)
        if g.shape[0] == K:
            return g, "gt"
    if pred_stack.size:
        return pred_stack.mean(axis=0), "centroid"
    return np.zeros((K, 3), dtype=np.float32), "centroid"


def write_recon_html(objects, out_path: Path, run_metrics: dict = None, key_points: int = 10):
    """
    Interactive viewer: one dropdown entry per object.

    Each keypoint slot gets its own colour and each partial view its own marker
    symbol, so slot identity and view identity read independently. Every view's
    position for a slot is joined by a spoke to that slot's anchor - the GT
    keypoint where one exists, otherwise the across-view centroid - so a tight
    star means the slot is stable and a long one means it moves with the view.
    GT keypoints are drawn as 'x', a symbol no view uses.

    `objects` is a list of dicts:
      name     : str, shown in the dropdown
      gt_cloud : (N,3) full ground-truth cloud, or None
      gt_kp    : (G,3) ground-truth keypoints, or None
      views    : list of {label, recon, input | input_parts, pred_kp, meta}
                 `input` is a single (N,3) cloud; `input_parts` (used for a combined
                 view) is a list of up to 2 {label, points, color} dicts instead,
                 so the sources making up the merged input keep separate colours.
                 An optional `recon_color` overrides the reconstruction colour, for
                 scripts that overlay several reconstructions of one object.
      meta     : object-level metrics for the title
    """
    if go is None:
        print("[!] plotly not installed - skipping HTML viewer.")
        return None
    if not objects:
        print("[!] No samples to visualize.")
        return None

    K = int(key_points)
    # every object contributes the same number of traces so the dropdown's
    # visibility offsets stay valid; objects with fewer views pad with empties.
    # the input block reserves 2 traces per view (one per source) so a combined
    # view's input_parts and a plain view's single input share the same layout
    v_max = max(len(o["views"]) for o in objects)
    traces_per_object = 2 + 4 * v_max + K

    fig = go.Figure()
    run_line = format_run_metrics(run_metrics or {})
    buttons = []

    for oi, obj in enumerate(objects):
        views = obj["views"]
        first = oi == 0
        pred_stack = np.stack(
            [as_xyz(v.get("pred_kp")) for v in views], axis=0
        ) if views else np.zeros((0, K, 3), dtype=np.float32)
        anchors, anchor_kind = kp_anchors(obj.get("gt_kp"), pred_stack, K)

        # --- ground-truth cloud -------------------------------------------------
        gt_cloud = as_xyz(obj.get("gt_cloud"))
        fig.add_trace(
            go.Scatter3d(
                x=gt_cloud[:, 0], y=gt_cloud[:, 1], z=gt_cloud[:, 2],
                mode="markers",
                marker={"size": 1.6, "color": "#c8c7c0"},
                name="full cloud",
                visible=first,
                hoverinfo="skip",
            )
        )

        # --- reconstructions, one per view -------------------------------------
        for vi in range(v_max):
            # objects with fewer than v_max views pad with an empty, legend-hidden
            # trace so the stride stays constant across the dropdown
            pts = np.zeros((0, 3), np.float32)
            name = ""
            padding = vi >= len(views)
            if not padding:
                v = views[vi]
                pts = as_xyz(v.get("recon"))
                meta = v.get("meta") or {}
                cd, emd, coverage = meta.get("cd"), meta.get("emd"), meta.get("coverage")
                # coverage is NaN (not None) for a live mesh sample - it isn't tracked there
                has_coverage = coverage is not None and np.isfinite(coverage)
                name = (
                    f"recon {v['label']}"
                    + (f"  CD={cd:.2e}" if cd is not None else "")
                    + (f"  EMD={emd:.2e}" if emd is not None else "")
                    + (f"  coverage={coverage:.4f}" if has_coverage else "")
                )
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode="markers",
                    marker={
                        "size": 2.0,
                        "color": RECON_COLOR if padding else views[vi].get("recon_color", RECON_COLOR),
                        "opacity": 0.85,
                    },
                    name=name,
                    showlegend=not padding,
                    visible=first,
                    hoverinfo="skip",
                )
            )

        # --- model inputs (the partial views), off by default --------------------
        # 2 traces per view (one per source), so a combined view's "input_parts"
        # (first sample / second sample) and a plain view's single "input" both fit
        # the same fixed layout - the second trace is just empty for a plain view
        for vi in range(v_max):
            padding = vi >= len(views)
            parts = None if padding else views[vi].get("input_parts")
            for part_idx in range(2):
                pts = np.zeros((0, 3), np.float32)
                name = ""
                color = INPUT_SOURCE_COLORS[part_idx % len(INPUT_SOURCE_COLORS)]
                if not padding:
                    if parts is not None:
                        if part_idx < len(parts):
                            part = parts[part_idx]
                            pts = as_xyz(part.get("points"))
                            color = part.get("color", color)
                            name = f"input {views[vi]['label']} ({part.get('label', part_idx)})"
                    elif part_idx == 0:
                        pts = as_xyz(views[vi].get("input"))
                        name = f"input {views[vi]['label']}"
                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode="markers",
                        marker={"size": 1.8, "color": color, "opacity": 0.9},
                        name=name,
                        showlegend=(not padding) and pts.shape[0] > 0,
                        visible="legendonly" if first else False,
                        hoverinfo="skip",
                    )
                )

        # --- GT keypoints: slot-coloured, 'x', direct-labeled -------------------
        gt_kp = as_xyz(obj.get("gt_kp"))
        fig.add_trace(
            go.Scatter3d(
                x=gt_kp[:, 0], y=gt_kp[:, 1], z=gt_kp[:, 2],
                mode="markers+text",
                marker={
                    "size": 8,
                    "color": [slot_color(k) for k in range(gt_kp.shape[0])],
                    "symbol": GT_KP_SYMBOL,
                    "line": {"width": 2, "color": INK_PRIMARY},
                },
                text=[str(k) for k in range(gt_kp.shape[0])],
                textposition="top center",
                textfont={"size": 11, "color": INK_PRIMARY},
                hovertext=[f"GT kp {k}" for k in range(gt_kp.shape[0])],
                hoverinfo="text",
                name="GT keypoints",
                visible=first,
            )
        )

        # --- predicted keypoints, one trace per view ----------------------------
        for vi in range(v_max):
            pts = np.zeros((0, 3), np.float32)
            name = ""
            hover = []
            padding = vi >= len(views)
            if not padding:
                pts = as_xyz(views[vi].get("pred_kp"))
                name = f"kp {views[vi]['label']}"
                hover = [
                    f"kp {k} - {views[vi]['label']}<br>"
                    f"{float(np.linalg.norm(pts[k] - anchors[k])):.4f} from {anchor_kind}"
                    for k in range(pts.shape[0])
                ]
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode="markers",
                    marker={
                        "size": 6,
                        "color": [slot_color(k) for k in range(pts.shape[0])],
                        "symbol": view_symbol(vi),
                        "line": {"width": 1, "color": INK_PRIMARY},
                    },
                    hovertext=hover,
                    hoverinfo="text",
                    name=name,
                    showlegend=not padding,
                    visible=first,
                )
            )

        # --- spokes: anchor -> each view's position, one trace per slot ---------
        for k in range(K):
            xs, ys, zs = [], [], []
            if pred_stack.size and k < anchors.shape[0]:
                ax, ay, az = anchors[k]
                for vi in range(pred_stack.shape[0]):
                    if k >= pred_stack.shape[1]:
                        continue
                    px, py, pz = pred_stack[vi, k]
                    # a None break keeps the spokes from joining end to end
                    xs += [ax, px, None]
                    ys += [ay, py, None]
                    zs += [az, pz, None]
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines",
                    line={"width": 2, "color": slot_color(k)},
                    name=f"slot {k}",
                    showlegend=False,
                    visible=first,
                    hoverinfo="skip",
                )
            )

        # --- dropdown entry -----------------------------------------------------
        visible = [False] * (traces_per_object * len(objects))
        for t in range(traces_per_object):
            # trace order is: full cloud | recon x v_max | input(2 parts) x v_max |
            # GT keypoints | pred keypoints x v_max | spokes x K.
            # the input clouds sit behind the keypoints, so leave them collapsed
            is_input = 1 + v_max <= t < 1 + v_max + 2 * v_max
            visible[oi * traces_per_object + t] = "legendonly" if is_input else True

        object_line = format_sample_title(obj.get("name", f"object_{oi}"), obj.get("meta"))
        buttons.append(
            {
                "label": object_line,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"{run_line}<br>{object_line}" if run_line else object_line},
                ],
            }
        )

    first_line = format_sample_title(objects[0].get("name", "object_0"), objects[0].get("meta"))
    fig.update_layout(
        title=f"{run_line}<br>{first_line}" if run_line else first_line,
        template="plotly_white",
        paper_bgcolor=SURFACE,
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
            "bgcolor": SURFACE,
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
            }
        ],
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 90, "b": 0},
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    print(f"[✓] Saved interactive HTML to {out_path} ({len(objects)} objects)")
    return out_path


# ----------------------------
# Chart primitives
# ----------------------------


def binned_stats(x, y, n_bins: int):
    """
    Median and interquartile range of y in equal-count bins of x.

    Equal-count rather than equal-width: neither coverage nor keypoint distance is
    uniformly distributed, and equal-width bins out in the tail hold a handful of
    samples and swing wildly.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return {k: np.array([]) for k in ("center", "count", "mean", "median", "p25", "p75")}

    n_bins = max(1, min(n_bins, x.size))
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([x.min(), x.max() + 1e-12])
    idx = np.clip(np.digitize(x, edges[1:-1], right=False), 0, edges.size - 2)

    out = collections.defaultdict(list)
    for b in range(edges.size - 1):
        sel = idx == b
        if not sel.any():
            continue
        ys = y[sel]
        out["center"].append(float(np.median(x[sel])))
        out["count"].append(int(sel.sum()))
        out["mean"].append(float(ys.mean()))
        out["median"].append(float(np.median(ys)))
        out["p25"].append(float(np.quantile(ys, 0.25)))
        out["p75"].append(float(np.quantile(ys, 0.75)))
    return {k: np.asarray(v) for k, v in out.items()}


def spearman(x, y) -> float:
    """Rank correlation, so a monotone but non-linear trend still registers."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[m])).astype(np.float64)
    ry = np.argsort(np.argsort(y[m])).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def style_axis(ax, xlabel=None, ylabel=None, labelsize: int = 10):
    """
    Recessive grid and axes, ink-coloured text - applied to every panel.

    Axis labels are optional: some callers set them alongside other per-panel
    text and only want the chrome.
    """
    ax.set_facecolor(SURFACE)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=labelsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=labelsize)
    ax.grid(True, alpha=0.18, linewidth=0.7, color=INK_SECONDARY)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_SECONDARY)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def legend_on_surface(ax, **kwargs):
    """
    A legend that sits over data without a border: a surface-coloured plate keeps
    it readable where a hairline box would just add another line to the chart.
    """
    opts = {"fontsize": 8, "labelcolor": INK_SECONDARY, "loc": "best", "framealpha": 0.92}
    opts.update(kwargs)
    leg = ax.legend(**opts)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor("none")
    return leg


# ----------------------------
# Metrics vs input coverage
# ----------------------------


def coverage_array(covs) -> np.ndarray:
    """Coverage list (with Nones for samples that carry none) to a float array."""
    return np.array([np.nan if c is None else float(c) for c in covs], dtype=np.float64)


def coverage_has_signal(cov: np.ndarray, what: str = "coverage plot") -> bool:
    """
    Whether there is anything to regress the metrics against. Says why when not,
    so a skipped figure never looks like a silent failure.
    """
    finite = np.isfinite(cov)
    if finite.sum() < 3:
        print(f"[!] fewer than 3 samples carry a coverage value; skipping the {what}")
        return False
    # full clouds all report coverage 1.0
    if np.ptp(cov[finite]) < 1e-6:
        print(f"[!] coverage is constant at {cov[finite][0]:.4f}; skipping the {what}")
        return False
    return True


def _coverage_panel(ax, series, ylabel: str, n_bins: int,
                     xlabel: str = "input coverage (fraction of the full cloud visible)",
                     zero_line: bool = False):
    """
    One metric against an x-axis quantity (coverage by default, but any per-sample
    scalar works - see plot_metrics_vs_scalar), the per-sample cloud, plus the binned
    median.

    `series` is a list of (label, x_values, y_values, point_colour, line_colour).
    Each entry carries its own x so the multiview script can put the first view,
    the second view and their combination on one panel, each against its own
    coverage. One entry draws a median line with an interquartile band, two or
    more drop the bands (overlapping translucent bands read as mud) and rely on
    the legend. `zero_line` draws a dashed reference line at y=0 - for an improvement
    metric (positive = better), that line is the "no better/worse than baseline" mark.
    Returns the Spearman rho of each series, in order.
    """
    rhos = []
    if zero_line:
        ax.axhline(0.0, color=INK_SECONDARY, linewidth=1, linestyle="--", alpha=0.6, zorder=1)
    single = len(series) == 1
    for label, xs, vals, pt_color, line_color in series:
        xs = np.asarray(xs, dtype=np.float64)
        vals = np.asarray(vals, dtype=np.float64)
        m = np.isfinite(xs) & np.isfinite(vals)
        x, y = xs[m], vals[m]
        rho = spearman(x, y)
        rhos.append(rho)
        if x.size == 0:
            continue

        # alpha falls as the sample count grows, so the cloud stays readable
        # whether this is 50 views or 5000, and stays translucent enough for two
        # overlapping series (KVC / uKVC) to both show through
        alpha = float(np.clip(12.0 / max(x.size, 1) ** 0.5, 0.05, 0.35))
        ax.scatter(
            x, y, s=12, color=pt_color, alpha=alpha, linewidths=0,
            label=f"{label} (per view)" if not single else None, zorder=2,
        )

        stats = binned_stats(x, y, n_bins)
        if not stats["center"].size:
            continue
        if single:
            ax.fill_between(
                stats["center"], stats["p25"], stats["p75"],
                color=BAND_COLOR, alpha=0.55, linewidth=0,
                label="interquartile range", zorder=3,
            )
        ax.plot(
            stats["center"], stats["median"],
            color=line_color, linewidth=2, marker="o", markersize=4,
            markeredgecolor=SURFACE, markeredgewidth=0.8,
            label=f"{label} median" if not single else "median", zorder=4,
        )

    # a few bad views can stretch the axis and flatten everything else; clip the
    # view, not the data - every sample is still in the csv
    all_y = np.concatenate([np.asarray(v, dtype=np.float64) for _, _, v, _, _ in series])
    all_y = all_y[np.isfinite(all_y)]
    if all_y.size:
        hi = float(np.quantile(all_y, 0.995))
        ax.set_ylim(min(0.0, float(all_y.min())), hi if hi > 0 else None)

    rho_txt = "   |   ".join(
        f"rho{'' if single else f'({lbl})'} = {r:+.3f}"
        for (lbl, *_), r in zip(series, rhos)
    )
    # EMD values are ~1e-6, so matplotlib hangs a shared exponent just above the
    # axes - exactly where a pad=6 title would land. Clear it above both.
    ax.set_title(rho_txt, color=INK_SECONDARY, fontsize=9, loc="left", pad=16)
    ax.yaxis.get_offset_text().set(color=INK_SECONDARY, fontsize=8)
    style_axis(ax, xlabel=xlabel, ylabel=ylabel, labelsize=9)
    legend_on_surface(ax)
    return rhos


def render_coverage_figure(panels, out_path: Path, run_label: str, n_bins: int,
                           summary: str = "",
                           xlabel: str = "input coverage (fraction of the full cloud visible)",
                           title: str = "Reconstruction and keypoint quality vs input coverage",
                           zero_line: bool = False):
    """
    Draw one panel per metric and write the png. `panels` is a list of
    (key, ylabel, series), with series in the form _coverage_panel expects.
    `xlabel`/`title`/`zero_line` let callers reuse this against something other than
    coverage (see plot_metrics_vs_scalar) - the panel layout, binning and rho
    annotation are otherwise identical.

    One panel per metric rather than one axis with several scales: CD is a mean of
    squared distances, EMD carries emd_approx's 1/2N normalization and KVC is a
    raw distance, so they live orders of magnitude apart and sharing a y-axis
    would hide every trend but the largest.

    Returns {series label (lowercased): Spearman rho}.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)

    rhos = {}
    for ax, (key, ylabel, series) in zip(axes, panels):
        got = _coverage_panel(ax, ylabel=ylabel, n_bins=n_bins, series=series, xlabel=xlabel, zero_line=zero_line)
        for (label, *_), r in zip(series, got):
            # qualify with the panel only when the label does not already say it
            # ("cd" stays "cd"; the multiview "first"/"second" become "cd:first")
            lbl = label.lower()
            rhos[lbl if lbl == key else f"{key}:{lbl}"] = r

    lines = [title]
    if run_label:
        lines.append(run_label)
    if summary:
        lines.append(summary)
    fig.suptitle("\n".join(lines), color=INK_PRIMARY, fontsize=11,
                 x=0.01, y=0.995, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    rho_txt = "  ".join(f"{k}={v:+.3f}" for k, v in rhos.items())
    print(f"[✓] Wrote {out_path}  ({rho_txt})")
    return rhos


def plot_metrics_vs_coverage(covs, cds, emds, kvc, ukvc, out_path: Path,
                             run_label: str = "", n_bins: int = 12, mv=None):
    """
    Reconstruction and keypoint quality against how much of the object the input
    actually showed the model.

    A rising CD/EMD median towards low coverage is the expected shape - less input,
    worse reconstruction. KVC rising faster than uKVC means the keypoints still land
    in roughly the right places as the view degrades but get assigned to the wrong
    slots; the two rising together means they genuinely drift.

    `mv` carries the multiview per-variant arrays when a second input view was
    reconstructed too: {"first": {...}, "second": {...}}, each with `cds`, `emds`,
    `covs`, `psns`. With it the CD and EMD panels put all three variants side by
    side, each against ITS OWN coverage - first view alone, second view alone, and
    the combination. KVC / uKVC stay combined-only either way: the per-variant
    reconstructions carry no keypoint-consistency numbers of their own.

    Returns a dict of Spearman rho per series, or None if coverage carries no
    signal to plot against.
    """
    cov = coverage_array(covs)
    # the primary (combined) coverage is what the headline metrics are reported
    # against, so it decides whether the figure is worth drawing at all
    if not coverage_has_signal(cov):
        return None

    if mv:
        cov_first = coverage_array(mv["first"]["covs"])
        cov_second = coverage_array(mv["second"]["covs"])

        def metric_series(metric, primary):
            return [
                ("first", cov_first, mv["first"][metric],
                 VARIANT_COLORS["first"], VARIANT_COLORS["first"]),
                ("second", cov_second, mv["second"][metric],
                 VARIANT_COLORS["second"], VARIANT_COLORS["second"]),
                ("combined", cov, primary,
                 VARIANT_COLORS["combined"], VARIANT_COLORS["combined"]),
            ]
        cd_series = metric_series("cds", cds)
        emd_series = metric_series("emds", emds)
        kvc_ylabel = "keypoint distance to GT\n(combined input only)"
        all_cov = np.concatenate([cov, cov_first, cov_second])
        all_cov = all_cov[np.isfinite(all_cov)]
        summary = (f"{int(np.isfinite(cov).sum()):,} objects x 3 variants   |   coverage "
                   f"{all_cov.min():.3f}-{all_cov.max():.3f}   |   "
                   f"{n_bins} equal-count bins")
    else:
        cd_series = [("CD", cov, cds, POINT_COLOR, SUMMARY_COLOR)]
        emd_series = [("EMD", cov, emds, POINT_COLOR, SUMMARY_COLOR)]
        kvc_ylabel = "keypoint distance to GT"
        finite = np.isfinite(cov)
        summary = (f"{int(finite.sum()):,} views   |   coverage "
                   f"{cov[finite].min():.3f}-{cov[finite].max():.3f}   |   "
                   f"{n_bins} equal-count bins")

    panels = [
        ("cd", "CD (mean squared NN distance)", cd_series),
        ("emd", "EMD (emd_approx, normalized)", emd_series),
    ]
    # kvc/ukvc are NaN for every sample when there are no gt keypoints to compare
    # against, so the panel is only worth drawing when something is finite
    if np.any(np.isfinite(np.asarray(kvc, dtype=np.float64))):
        panels.append(
            ("kvc", kvc_ylabel,
             [("KVC", cov, kvc, POINT_COLOR, POINT_COLOR),
              ("uKVC", cov, ukvc, SUMMARY_COLOR, SUMMARY_COLOR)])
        )
    else:
        print("[!] no finite KVC values; the coverage figure will show CD and EMD only")

    return render_coverage_figure(panels, out_path, run_label, n_bins, summary)


def write_coverage_outputs(names, psns, covs, cds, emds, kvc, ukvc,
                           out_dir: Path, run_label: str, n_bins: int, mv=None,
                           category: str | None = None):
    """
    The figure plus the per-sample table behind it. See plot_metrics_vs_coverage
    for `mv`.

    The csv is raw rather than binned: there is one row per (view, variant), not
    per point, so it stays small enough to re-bin or re-plot however you like. The
    long format - a `variant` column rather than a wide header - is what lets the
    multiview rows sit in the same table as the single-view ones.

    `category`, when given, prefixes both filenames (<category>_metrics_vs_coverage.*)
    and fills the csv's `category` column. The sweep drivers point every category
    at one --output-dir, so without the prefix each category overwrote the last;
    plot_coverage_across_classes.py reads the column to label the classes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{category}_metrics_vs_coverage" if category else "metrics_vs_coverage"
    cat = category or ""
    primary = "combined" if mv else "single"
    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w") as f:
        f.write("category,name,variant,partial_sample,coverage,cd,emd,kvc,ukvc\n")
        for i in range(len(covs)):
            cov = "" if covs[i] is None else f"{float(covs[i]):.6f}"
            f.write(
                f"{cat},{names[i]},{primary},{psns[i]},{cov},"
                f"{float(cds[i]):.8e},{float(emds[i]):.8e},"
                f"{float(kvc[i]):.8e},{float(ukvc[i]):.8e}\n"
            )
        for variant in ("first", "second"):
            d = (mv or {}).get(variant)
            if not d:
                continue
            for i in range(len(d["covs"])):
                # keypoint columns stay blank for a variant that carries no
                # "kvc"/"ukvc" of its own
                kp = (f"{float(d['kvc'][i]):.8e},{float(d['ukvc'][i]):.8e}"
                      if "kvc" in d else ",")
                f.write(
                    f"{cat},{names[i]},{variant},{d['psns'][i]},{float(d['covs'][i]):.6f},"
                    f"{float(d['cds'][i]):.8e},{float(d['emds'][i]):.8e},{kp}\n"
                )
    print(f"[✓] Wrote {csv_path}")

    return plot_metrics_vs_coverage(
        covs, cds, emds, kvc, ukvc,
        out_path=out_dir / f"{stem}.png",
        run_label=run_label, n_bins=n_bins, mv=mv,
    )


# ----------------------------
# Metrics vs an arbitrary second-sample quality scalar
# ----------------------------


def plot_metrics_vs_scalar(cds, emds, kvc, ukvc, x, xlabel: str, out_path: Path,
                           run_label: str = "", n_bins: int = 12,
                           improvement: bool = False) -> dict | None:
    """
    Single-series version of plot_metrics_vs_coverage: CD/EMD/KVC/uKVC against one
    arbitrary per-object scalar `x` (e.g. how much coverage the second sample added,
    or how far apart the first and second samples were) instead of input coverage.
    Same panel layout and binning underneath (render_coverage_figure), just unsplit -
    there is only one series to plot here, not first/second/combined.

    `improvement=True` means cds/emds/kvc/ukvc are (first-alone − combined) deltas
    rather than raw combined-view values: positive means combining helped (lower
    error is better for all four), so the panels get an "improvement" y-label and a
    y=0 reference line marking "no better than the first sample alone".

    Returns Spearman rho per panel, or None if `x` carries no signal (constant, or
    fewer than 3 finite samples) - coverage_has_signal already prints why.
    """
    xv = np.asarray(x, dtype=np.float64)
    if not coverage_has_signal(xv, what=f"'{xlabel}' figure"):
        return None

    cd_ylabel = "CD improvement (first alone − combined)" if improvement else "combined CD (mean squared NN distance)"
    emd_ylabel = "EMD improvement (first alone − combined)" if improvement else "combined EMD (emd_approx, normalized)"
    kvc_ylabel = "keypoint distance improvement (first alone − combined)" if improvement else "keypoint distance to GT (combined input)"

    panels = [
        ("cd", cd_ylabel, [("CD", xv, cds, POINT_COLOR, SUMMARY_COLOR)]),
        ("emd", emd_ylabel, [("EMD", xv, emds, POINT_COLOR, SUMMARY_COLOR)]),
    ]
    if np.any(np.isfinite(np.asarray(kvc, dtype=np.float64))):
        panels.append(
            ("kvc", kvc_ylabel,
             [("KVC", xv, kvc, POINT_COLOR, POINT_COLOR),
              ("uKVC", xv, ukvc, SUMMARY_COLOR, SUMMARY_COLOR)])
        )

    finite = np.isfinite(xv)
    summary = (f"{int(finite.sum()):,} objects   |   {xlabel} "
               f"{xv[finite].min():.3f} to {xv[finite].max():.3f}   |   {n_bins} equal-count bins")
    title_metric = "reconstruction improvement over the first sample alone" if improvement else "combined reconstruction quality"

    return render_coverage_figure(
        panels, out_path, run_label, n_bins, summary,
        xlabel=xlabel, title=f"{title_metric[:1].upper()}{title_metric[1:]} vs {xlabel}",
        zero_line=improvement,
    )


def write_second_sample_outputs(names, psns, cd_improvement, emd_improvement, kvc_improvement, ukvc_improvement,
                                coverage_added, first_second_dist,
                                out_dir: Path, run_label: str, n_bins: int,
                                category: str | None = None) -> dict:
    """
    Two more figures alongside metrics_vs_coverage: how much the COMBINED
    reconstruction IMPROVES on the first-sample-alone baseline (first − combined;
    positive = combining helped) as a function of (a) how much new surface the
    second sample added to the merged input, net of what it re-saw of the first, and
    (b) how far apart the first and second model inputs are (symmetric mean
    nearest-neighbour distance). Only meaningful for a multiview run - the caller
    skips this entirely otherwise.

    Each gets its own csv (one row per object, same long/tidy shape as
    metrics_vs_coverage.csv) plus a png with the same CD/EMD/KVC panel layout.
    `category` prefixes the filenames, as in write_coverage_outputs.
    Returns {"coverage_added": rho_dict, "first_second_dist": rho_dict} (a key is
    omitted if its figure had no signal to plot).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{category}_" if category else ""
    rhos = {}

    for key, x, xlabel, csv_col in (
        ("coverage_added", coverage_added, "coverage added by the second sample", "coverage_added"),
        ("first_second_dist", first_second_dist,
         "distance between first and second samples", "first_second_dist"),
    ):
        csv_path = out_dir / f"{prefix}metrics_vs_{key}.csv"
        with open(csv_path, "w") as f:
            f.write(f"name,partial_sample,{csv_col},cd_improvement,emd_improvement,kvc_improvement,ukvc_improvement\n")
            for i in range(len(x)):
                f.write(
                    f"{names[i]},{psns[i]},{float(x[i]):.6f},"
                    f"{float(cd_improvement[i]):.8e},{float(emd_improvement[i]):.8e},"
                    f"{float(kvc_improvement[i]):.8e},{float(ukvc_improvement[i]):.8e}\n"
                )
        print(f"[✓] Wrote {csv_path}")

        got = plot_metrics_vs_scalar(
            cd_improvement, emd_improvement, kvc_improvement, ukvc_improvement, x, xlabel,
            improvement=True,
            out_path=out_dir / f"{prefix}metrics_vs_{key}.png",
            run_label=run_label, n_bins=n_bins,
        )
        if got is not None:
            rhos[key] = got

    return rhos
