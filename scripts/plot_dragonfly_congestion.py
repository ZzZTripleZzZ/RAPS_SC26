#!/usr/bin/env python3
"""
plot_dragonfly_congestion.py  —  Dragonfly network congestion chord diagram.

Produces a radial chord diagram showing:
  - Outer ring arcs : one sector per group, colored by peak intra-group link utilization
  - Inner chords    : inter-group links colored and weighted by utilization
  - Group labels    : numbered around the ring
  - Colorbar        : maps arc/chord color → link utilization

Style mirrors DragonView: dark background, plasma colormap, hot chords for congestion.

────────────────────────────────────────────────────────────────────────────────
INPUT CSV  (src, dst, bytes)
  src  = node name:  r_{group}_{router_idx}  or  h_{group}_{router_idx}_{port}
  dst  = same convention
  bytes = raw bytes transferred on this link during the simulation interval

  Example rows:
    r_0_0,r_1_0,8589934592
    h_0_0_2,r_0_0,12345678

────────────────────────────────────────────────────────────────────────────────
USAGE

  # Visualise a link-loads dump from a raps simulation:
  python scripts/plot_dragonfly_congestion.py link_loads.csv \\
      --system config/frontier.yaml --title "Frontier t=3 h" -o congestion.png

  # Quick demo with synthetic Frontier-sized data (no CSV needed):
  python scripts/plot_dragonfly_congestion.py --demo --system config/frontier.yaml

  # Override bandwidth / timestep manually:
  python scripts/plot_dragonfly_congestion.py link_loads.csv --bw 25e9 --dt 15

────────────────────────────────────────────────────────────────────────────────
GENERATING THE CSV FROM A RAPS SIMULATION

  After running raps with --net enabled, call:

      network_model.dump_link_loads("link_loads.csv", dt=time_delta)

  where `dt` is the simulation timestep (seconds).  The NetworkModel.dump_link_loads()
  method is defined in raps/network/__init__.py.

  Alternatively, patch the engine tick to dump at every timestep:

      # in engine.py, after self.network_model.reset_link_loads():
      self.network_model.dump_link_loads(f"links_{self.current_time:.0f}.csv",
                                         dt=self.time_delta)
"""

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# ── node name helpers ────────────────────────────────────────────────────────

def _parse_node(name: str):
    """Return (kind, group, ...) or None for unknown format."""
    p = name.split("_")
    try:
        if p[0] == "r" and len(p) == 3:
            return ("router", int(p[1]), int(p[2]))
        if p[0] == "h" and len(p) == 4:
            return ("host", int(p[1]), int(p[2]), int(p[3]))
    except ValueError:
        pass
    return None


def _group_of(name: str):
    """Fast group extraction — returns group int or None."""
    info = _parse_node(name)
    return info[1] if info else None


# ── CSV loading & aggregation ────────────────────────────────────────────────

def load_and_aggregate(path: str, bw_bytes_per_s: float, dt_s: float):
    """
    Parse a link-loads CSV and return per-group utilisation dicts.

    Returns
    -------
    intra_util : dict  {group_id: float}   peak intra-group link utilisation
    inter_util : dict  {(g1,g2): float}    summed inter-group utilisation (g1 < g2)
    num_groups : int   inferred from router names
    """
    bw_per_tick = bw_bytes_per_s * dt_s
    intra_lists: dict = defaultdict(list)
    inter_sums:  dict = defaultdict(float)
    all_groups:  set  = set()
    skipped = 0

    with open(path, newline="") as f:
        # Skip comment lines (e.g. "# dt=15" written by dump_link_loads)
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            src_info = _parse_node(row["src"])
            dst_info = _parse_node(row["dst"])
            if src_info is None or dst_info is None:
                skipped += 1
                continue
            raw  = float(row["bytes"])
            util = raw / bw_per_tick if bw_per_tick > 0 else 0.0
            sg   = src_info[1]
            dg   = dst_info[1]
            all_groups.add(sg)
            all_groups.add(dg)

            if sg == dg:
                intra_lists[sg].append(util)
            else:
                key = (min(sg, dg), max(sg, dg))
                inter_sums[key] += util

    if skipped:
        print(f"  Warning: {skipped} rows had unrecognised node names, skipped.")

    num_groups = (max(all_groups) + 1) if all_groups else 0
    intra_util = {g: max(v) for g, v in intra_lists.items() if v}
    return intra_util, dict(inter_sums), num_groups


# ── geometry helpers ─────────────────────────────────────────────────────────

def _sector_angles(num_groups: int, gap_frac: float = 0.12):
    """Return list of (theta_start, theta_end, theta_center) per group."""
    sector = 2 * math.pi / num_groups
    gap    = sector * gap_frac
    angles = []
    for g in range(num_groups):
        theta_c = 2 * math.pi * g / num_groups
        theta_s = theta_c - sector / 2 + gap / 2
        theta_e = theta_c + sector / 2 - gap / 2
        angles.append((theta_s, theta_e, theta_c))
    return angles


def _draw_arc_sector(ax, theta_s, theta_e, r_in, r_out, color, n=128):
    """Filled arc sector (the group 'tile' on the outer ring)."""
    t  = np.linspace(theta_s, theta_e, n)
    xo = r_out * np.cos(t)
    yo = r_out * np.sin(t)
    xi = r_in  * np.cos(t[::-1])
    yi = r_in  * np.sin(t[::-1])
    ax.fill(
        np.concatenate([xo, xi, [xo[0]]]),
        np.concatenate([yo, yi, [yo[0]]]),
        color=color, zorder=3, linewidth=0,
    )


def _draw_chord(ax, p1, p2, color, alpha: float, lw: float, pull: float = 0.28):
    """Cubic Bézier chord from p1 → p2, bowing toward the origin."""
    c1 = (p1[0] * pull, p1[1] * pull)
    c2 = (p2[0] * pull, p2[1] * pull)
    patch = mpatches.PathPatch(
        mpath.Path(
            [p1, c1, c2, p2],
            [mpath.Path.MOVETO, mpath.Path.CURVE4,
             mpath.Path.CURVE4, mpath.Path.CURVE4],
        ),
        facecolor="none", edgecolor=color,
        linewidth=lw, alpha=alpha, capstyle="round", zorder=1,
    )
    ax.add_patch(patch)


# ── main plot ────────────────────────────────────────────────────────────────

def plot_dragonfly_congestion(
    intra_util: dict,
    inter_util: dict,
    num_groups:  int,
    *,
    title:      str = "",
    output:     str = "dragonfly_congestion.png",
    top_n:      int | None = None,
    label_size: int | None = None,
):
    """
    Render and save the chord-diagram congestion plot.

    Parameters
    ----------
    intra_util  peak intra-group link utilisation per group  {g: float}
    inter_util  summed inter-group utilisation               {(g1,g2): float}
    num_groups  total number of groups
    title       optional figure title
    output      output image path
    top_n       if set, draw only the top_n highest-utilisation inter-group chords
    """
    if num_groups == 0:
        print("No groups found — nothing to plot.", file=sys.stderr)
        return

    R_OUT  = 0.90   # outer edge of group arc
    R_IN   = 0.82   # inner edge of group arc
    R_CH   = 0.81   # chord attachment point (just inside arc)
    R_LBL  = 0.97   # group-label radius

    BG      = "white"
    FG_TXT  = "#1e293b"
    DIM_TXT = "#64748b"

    cmap = plt.cm.plasma

    # Chord normalisation against the *actual* maximum, so the busiest link
    # always maps to the hot end of the colormap.
    max_inter  = max(inter_util.values(), default=1.0)
    norm_arc   = Normalize(vmin=0.0, vmax=1.0)
    norm_chord = Normalize(vmin=0.0, vmax=max_inter)

    angles = _sector_angles(num_groups)

    fig, ax = plt.subplots(figsize=(14, 13), facecolor=BG)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.20, 1.15)
    ax.axis("off")
    ax.set_facecolor(BG)

    # ── draw chords first (behind arcs) ─────────────────────────────────────
    sorted_chords = sorted(inter_util.items(), key=lambda kv: kv[1])
    if top_n is not None:
        sorted_chords = sorted_chords[-top_n:]

    for (g1, g2), util in sorted_chords:
        if util <= 0:
            continue
        _, _, c1 = angles[g1]
        _, _, c2 = angles[g2]
        p1 = (R_CH * math.cos(c1), R_CH * math.sin(c1))
        p2 = (R_CH * math.cos(c2), R_CH * math.sin(c2))
        n_util = norm_chord(util)
        color  = cmap(n_util)
        lw     = 0.3 + 4.5 * n_util
        alpha  = 0.25 + 0.70 * n_util
        _draw_chord(ax, p1, p2, color, alpha, lw)

    # ── draw group arcs (on top of chords) ───────────────────────────────────
    for g in range(num_groups):
        theta_s, theta_e, theta_c = angles[g]
        util  = min(intra_util.get(g, 0.0), 1.0)
        color = cmap(norm_arc(util))
        _draw_arc_sector(ax, theta_s, theta_e, R_IN, R_OUT, color)

        # label
        lx  = R_LBL * math.cos(theta_c)
        ly  = R_LBL * math.sin(theta_c)
        rot = math.degrees(theta_c)
        if 90 < rot % 360 <= 270:
            rot += 180
        fontsize = label_size if label_size is not None else 12
        ax.text(lx, ly, str(g),
                ha="center", va="center",
                fontsize=fontsize, color=FG_TXT,
                rotation=rot, rotation_mode="anchor", zorder=5)

    # ── centre annotation ────────────────────────────────────────────────────
    ax.text(0, 0, f"Dragonfly\n{num_groups} groups",
            ha="center", va="center", fontsize=10,
            color=DIM_TXT, style="italic")

    # ── legend patches ───────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=cmap(0.05), label="≈0 % utilisation"),
        mpatches.Patch(color=cmap(0.50), label="50 % utilisation"),
        mpatches.Patch(color=cmap(1.00), label="≥100 % (congested)"),
    ]
    legend = ax.legend(handles=handles, loc="lower left",
                       fontsize=9, framealpha=0.8,
                       facecolor="white", edgecolor="#cbd5e1",
                       labelcolor=FG_TXT)

    # ── colourbar ────────────────────────────────────────────────────────────
    sm = ScalarMappable(cmap=cmap, norm=norm_arc)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        pad=0.02, fraction=0.03, shrink=0.50, aspect=28)
    cbar.set_label(
        "Link Utilisation  (arc = peak intra-group,  chord = total inter-group)",
        color=FG_TXT, fontsize=10,
    )
    cbar.ax.tick_params(colors=FG_TXT)
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color=FG_TXT)
    cbar.outline.set_edgecolor("#cbd5e1")
    cbar.ax.set_facecolor(BG)

    if title:
        ax.set_title(title, color=FG_TXT, fontsize=13, pad=14)

    plt.tight_layout(pad=0.5)
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {output}")
    plt.close(fig)


# ── demo data generation ─────────────────────────────────────────────────────

def _generate_demo_csv(path: str, D: int, A: int, P: int,
                       bw_bytes: float, dt: float, n_jobs: int = 6):
    """
    Generate a synthetic link-loads CSV using raps dragonfly code.
    Falls back to pure-random data if raps is not importable.
    """
    try:
        from raps.network.dragonfly import (
            build_dragonfly2, link_loads_for_job_dragonfly_adaptive,
        )
        G = build_dragonfly2(D=D, A=A, P=P)
        all_hosts = [n for n in G if n.startswith("h_")]
        random.seed(42)

        link_loads: dict = {tuple(sorted(e)): 0.0 for e in G.edges()}

        for job_idx in range(n_jobs):
            # Random allocation: each job gets 10–25% of hosts in 1–3 groups
            n_groups = random.randint(1, min(3, D))
            job_groups = random.sample(range(D), n_groups)
            job_hosts  = [h for h in all_hosts
                          if int(h.split("_")[1]) in job_groups]
            job_hosts  = random.sample(job_hosts, min(len(job_hosts), 64))

            # Traffic volume: some jobs are heavy hitters
            volume = bw_bytes * dt * random.uniform(0.1, 1.8) / max(len(job_hosts), 1)

            loads = link_loads_for_job_dragonfly_adaptive(
                G, job_hosts, volume,
                algorithm="ugal", d=A, a=D - 1,
                link_loads=link_loads,
            )
            for edge, b in loads.items():
                if edge in link_loads:
                    link_loads[edge] += b

        print(f"  Generated {n_jobs} synthetic jobs on {D}-group dragonfly (raps).")

    except ImportError:
        # Pure-random fallback — does not require raps to be installed
        print("  raps not importable; using random synthetic data.")
        link_loads = {}
        # Inter-group links — a handful of congested group pairs
        congested_pairs = [(random.randint(0, D-1), random.randint(0, D-1))
                           for _ in range(D // 2)]
        for g1 in range(D):
            for g2 in range(D):
                if g1 == g2:
                    continue
                key = (f"r_{g1}_{random.randint(0,A-1)}",
                       f"r_{g2}_{random.randint(0,A-1)}")
                base = random.uniform(0.0, 0.3)
                if (g1, g2) in congested_pairs or (g2, g1) in congested_pairs:
                    base += random.uniform(0.5, 1.5)
                link_loads[key] = base * bw_bytes * dt

        for g in range(D):
            for r in range(A):
                key = (f"r_{g}_{r}", f"r_{g}_{(r+1) % A}")
                link_loads[key] = random.uniform(0.0, 0.6) * bw_bytes * dt

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "dst", "bytes"])
        for (u, v), b in link_loads.items():
            writer.writerow([u, v, f"{b:.0f}"])

    print(f"  Demo CSV → {path}  ({len(link_loads)} links)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _load_system_yaml(path: str):
    """Return (bw_bytes_per_s, dt_s, D, A, P) from a raps system YAML config."""
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed; ignoring --system.", file=sys.stderr)
        return None
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        net = cfg.get("network", {})
        sch = cfg.get("scheduler", {})
        bw  = float(net.get("network_max_bw", 25e9))
        dt  = float(sch.get("trace_quanta",  15.0))
        D   = int(net.get("dragonfly_a", 48)) + 1  # a global links/router → D=a+1 groups
        A   = int(net.get("dragonfly_d", 48))       # routers per group
        Pp  = int(net.get("dragonfly_p", 4))
        return bw, dt, D, A, Pp
    except Exception as e:
        print(f"Warning: could not parse system YAML ({e}); using defaults.",
              file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot Dragonfly network congestion as a chord diagram.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv", nargs="?",
                        help="Link loads CSV (src,dst,bytes). "
                             "Omit when using --demo.")
    parser.add_argument("--demo", action="store_true",
                        help="Generate synthetic data and plot it "
                             "(no CSV required).")
    parser.add_argument("-o", "--output", default="dragonfly_congestion.png",
                        help="Output image path [%(default)s].")
    parser.add_argument("-s", "--system",
                        help="raps system YAML config (auto-reads bw, dt, topology).")
    parser.add_argument("--bw", type=float, default=25e9,
                        help="Link bandwidth in bytes/s [%(default)s].")
    parser.add_argument("--dt", type=float, default=15.0,
                        help="Simulation timestep in seconds [%(default)s].")
    parser.add_argument("-D", type=int, default=None,
                        help="Number of groups (overrides YAML; used for --demo).")
    parser.add_argument("-A", type=int, default=None,
                        help="Routers per group (overrides YAML; used for --demo).")
    parser.add_argument("-P", type=int, default=None,
                        help="Hosts per router (used for --demo).")
    parser.add_argument("--title", default="",
                        help="Figure title.")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Draw only the N most congested inter-group chords.")
    parser.add_argument("--label-size", type=int, default=None,
                        help="Font size for group number labels (default: auto-scaled).")
    args = parser.parse_args()

    # ── resolve bandwidth / topology params ──────────────────────────────────
    bw, dt = args.bw, args.dt
    D, A, Pp = 49, 48, 4   # Frontier defaults

    if args.system:
        result = _load_system_yaml(args.system)
        if result:
            bw, dt, D, A, Pp = result
            print(f"System config: {D} groups, {A} routers/group, "
                  f"bw={bw/1e9:.1f} GB/s, dt={dt}s")

    # CLI overrides
    if args.D is not None: D  = args.D
    if args.A is not None: A  = args.A
    if args.P is not None: Pp = args.P
    if args.bw != 25e9:    bw = args.bw
    if args.dt != 15.0:    dt = args.dt

    # ── demo or CSV mode ─────────────────────────────────────────────────────
    if args.demo:
        demo_csv = Path(args.output).with_suffix(".demo.csv")
        print(f"Generating demo data for {D}-group dragonfly "
              f"({A} routers/group, {Pp} hosts/router)...")
        _generate_demo_csv(str(demo_csv), D=D, A=A, P=Pp,
                           bw_bytes=bw, dt=dt, n_jobs=8)
        csv_path = str(demo_csv)
        if not args.title:
            args.title = f"Dragonfly Congestion (synthetic demo — {D} groups)"
    else:
        if not args.csv:
            parser.error("Provide a CSV path or use --demo.")
        csv_path = args.csv

    # ── load & plot ───────────────────────────────────────────────────────────
    print(f"Loading {csv_path}  (bw={bw/1e9:.1f} GB/s, dt={dt}s) ...")
    intra_util, inter_util, num_groups = load_and_aggregate(csv_path, bw, dt)

    active_inter = sum(1 for v in inter_util.values() if v > 0)
    print(f"  {num_groups} groups, {active_inter}/{len(inter_util)} "
          f"inter-group pairs have traffic")
    if intra_util:
        peak = max(intra_util.values())
        busiest_g = max(intra_util, key=intra_util.get)
        print(f"  Peak intra-group utilisation: {peak:.2%} in group {busiest_g}")
    if inter_util:
        busiest_pair = max(inter_util, key=inter_util.get)
        peak_inter   = inter_util[busiest_pair]
        print(f"  Peak inter-group traffic: groups {busiest_pair[0]}↔{busiest_pair[1]} "
              f"({peak_inter:.2%} aggregate utilisation)")

    plot_dragonfly_congestion(
        intra_util, inter_util, num_groups,
        title=args.title, output=args.output, top_n=args.top_n,
        label_size=args.label_size,
    )


if __name__ == "__main__":
    main()
