#!/usr/bin/env python3
"""
Airfoil Selector — Interactive Airfoil Comparison & Selection Tool
==================================================================
Compares candidate low-Reynolds-number airfoils for the NOOBLERS UAV.
Generates geometry overlays, polar comparisons, and a ranked selection table.

Usage:
    python scripts/airfoil_selector.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
AIRFOIL_DIR = DATA / "airfoils"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# ── Empirical airfoil performance data (at Re ≈ 200 000) ────────────────────
# Sources: UIUC Airfoil Data Site, Profili, XFLR5 published polars
AIRFOIL_DB = {
    "S1223": {
        "file": "s1223.dat",
        "cl_max": 2.20,
        "cl_0": 0.90,
        "cd_min": 0.0140,
        "alpha_stall_deg": 14.0,
        "alpha_zero_lift_deg": -8.0,
        "cm_quarter": -0.13,
        "ld_max": 55.0,
        "thickness_pct": 12.1,
        "camber_pct": 8.7,
        "re_design": 200000,
        "notes": "Purpose-designed high-lift, low-Re. Gradual stall.",
    },
    "Clark Y": {
        "file": "clark_y.dat",
        "cl_max": 1.40,
        "cl_0": 0.40,
        "cd_min": 0.0095,
        "alpha_stall_deg": 15.0,
        "alpha_zero_lift_deg": -4.0,
        "cm_quarter": -0.07,
        "ld_max": 48.0,
        "thickness_pct": 11.7,
        "camber_pct": 3.4,
        "re_design": 200000,
        "notes": "Classic flat-bottom trainer airfoil. Docile stall.",
    },
    "NACA 2412": {
        "file": "naca2412.dat",
        "cl_max": 1.35,
        "cl_0": 0.25,
        "cd_min": 0.0085,
        "alpha_stall_deg": 16.0,
        "alpha_zero_lift_deg": -2.5,
        "cm_quarter": -0.05,
        "ld_max": 50.0,
        "thickness_pct": 12.0,
        "camber_pct": 2.0,
        "re_design": 200000,
        "notes": "General-purpose NACA 4-digit. Low pitching moment.",
    },
    "E423": {
        "file": "e423.dat",
        "cl_max": 2.05,
        "cl_0": 0.82,
        "cd_min": 0.0130,
        "alpha_stall_deg": 13.0,
        "alpha_zero_lift_deg": -7.5,
        "cm_quarter": -0.15,
        "ld_max": 52.0,
        "thickness_pct": 12.5,
        "camber_pct": 9.3,
        "re_design": 200000,
        "notes": "Eppler high-lift. Slightly abrupt stall.",
    },
    "MH 114": {
        "file": "mh114.dat",
        "cl_max": 1.60,
        "cl_0": 0.55,
        "cd_min": 0.0100,
        "alpha_stall_deg": 13.5,
        "alpha_zero_lift_deg": -4.5,
        "cm_quarter": -0.09,
        "ld_max": 58.0,
        "thickness_pct": 10.2,
        "camber_pct": 5.4,
        "re_design": 200000,
        "notes": "Hepperle low-Re optimised. Good L/D but lower CL_max.",
    },
}

# ── Scoring weights for selection ────────────────────────────────────────────
WEIGHTS = {
    "cl_max": 0.30,        # High lift is critical for slow flight
    "ld_max": 0.20,        # Efficiency matters for endurance
    "cd_min": 0.10,        # Lower drag is always better
    "alpha_stall_deg": 0.10,  # Higher stall angle → more margin
    "cm_quarter": 0.15,    # Lower |Cm| eases trim design
    "stall_gentleness": 0.15,  # Gradual stall preferred
}


def load_airfoil_coords(filepath):
    """Load airfoil coordinate file, skipping header line."""
    x, y = [], []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i == 0:  # skip name header
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(x), np.array(y)


def estimate_cl_vs_alpha(airfoil_data, alpha_range_deg):
    """
    Estimate CL vs alpha using thin airfoil theory slope (2π/rad)
    corrected for viscous effects, with a soft-stall model beyond CL_max.
    """
    cl_alpha_slope = 2 * np.pi * (1 - 0.15)  # ~5.34 /rad, viscous correction
    alpha_rad = np.deg2rad(alpha_range_deg)
    alpha_0 = np.deg2rad(airfoil_data["alpha_zero_lift_deg"])
    alpha_stall = np.deg2rad(airfoil_data["alpha_stall_deg"])

    cl = np.zeros_like(alpha_rad)
    for i, a in enumerate(alpha_rad):
        cl_linear = cl_alpha_slope * (a - alpha_0)
        if cl_linear <= airfoil_data["cl_max"]:
            cl[i] = cl_linear
        else:
            # Soft stall model: gradual decay beyond CL_max
            excess = a - alpha_stall
            cl[i] = airfoil_data["cl_max"] - 2.0 * excess**2
    return cl


def estimate_cd_vs_cl(airfoil_data, cl_array, AR=4.3, e=0.75):
    """
    Estimate CD = CD_min + k*(CL - CL_minD)^2 + CL^2/(π·AR·e)
    Profile drag + induced drag.
    """
    cl_min_drag = airfoil_data["cl_0"] * 0.5
    k_profile = (airfoil_data["cd_min"] * 0.8) / max((airfoil_data["cl_max"] - cl_min_drag)**2, 0.01)
    cd_profile = airfoil_data["cd_min"] + k_profile * (cl_array - cl_min_drag)**2
    cd_induced = cl_array**2 / (np.pi * AR * e)
    return cd_profile + cd_induced


def plot_airfoil_geometries():
    """Plot all airfoil shapes overlaid for visual comparison."""
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(AIRFOIL_DB)))

    for (name, data), color in zip(AIRFOIL_DB.items(), colors):
        filepath = AIRFOIL_DIR / data["file"]
        if filepath.exists():
            x, y = load_airfoil_coords(filepath)
            ax.plot(x, y, linewidth=2, label=name, color=color)
            ax.fill(x, y, alpha=0.08, color=color)

    ax.set_xlabel("x/c", fontsize=12)
    ax.set_ylabel("y/c", fontsize=12)
    ax.set_title("Airfoil Geometry Comparison", fontsize=16, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.05)
    fig.tight_layout()
    fig.savefig(OUTPUT / "airfoil_geometry_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {OUTPUT / 'airfoil_geometry_comparison.png'}")
    plt.close(fig)


def plot_cl_vs_alpha():
    """Plot CL vs angle of attack for all airfoils."""
    fig, ax = plt.subplots(figsize=(10, 7))
    alpha = np.linspace(-5, 20, 200)
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(AIRFOIL_DB)))

    for (name, data), color in zip(AIRFOIL_DB.items(), colors):
        cl = estimate_cl_vs_alpha(data, alpha)
        ax.plot(alpha, cl, linewidth=2.2, label=name, color=color)
        # Mark CL_max
        idx_max = np.argmax(cl)
        ax.plot(alpha[idx_max], cl[idx_max], 'o', color=color, markersize=8)

    ax.set_xlabel("Angle of Attack α (°)", fontsize=12)
    ax.set_ylabel("Lift Coefficient CL", fontsize=12)
    ax.set_title("CL vs α — Airfoil Comparison (Re ≈ 200,000)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT / "cl_vs_alpha_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {OUTPUT / 'cl_vs_alpha_comparison.png'}")
    plt.close(fig)


def plot_drag_polar():
    """Plot CL vs CD drag polars for all airfoils."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(AIRFOIL_DB)))

    for (name, data), color in zip(AIRFOIL_DB.items(), colors):
        cl = np.linspace(0, data["cl_max"], 100)
        cd = estimate_cd_vs_cl(data, cl)
        ax.plot(cd, cl, linewidth=2.2, label=name, color=color)

    ax.set_xlabel("Drag Coefficient CD", fontsize=12)
    ax.set_ylabel("Lift Coefficient CL", fontsize=12)
    ax.set_title("Drag Polar — Airfoil Comparison", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "drag_polar_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {OUTPUT / 'drag_polar_comparison.png'}")
    plt.close(fig)


def plot_ld_ratio():
    """Plot L/D ratio vs CL for all airfoils."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(AIRFOIL_DB)))

    for (name, data), color in zip(AIRFOIL_DB.items(), colors):
        cl = np.linspace(0.1, data["cl_max"], 100)
        cd = estimate_cd_vs_cl(data, cl)
        ld = cl / cd
        ax.plot(cl, ld, linewidth=2.2, label=name, color=color)
        idx_max = np.argmax(ld)
        ax.plot(cl[idx_max], ld[idx_max], 'o', color=color, markersize=8)

    ax.set_xlabel("Lift Coefficient CL", fontsize=12)
    ax.set_ylabel("Lift-to-Drag Ratio (L/D)", fontsize=12)
    ax.set_title("L/D vs CL — Airfoil Comparison", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "ld_ratio_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {OUTPUT / 'ld_ratio_comparison.png'}")
    plt.close(fig)


def compute_scores():
    """Score and rank each airfoil based on weighted criteria."""
    stall_gentleness = {
        "S1223": 0.95,      # "gradual stall onset" — well-documented
        "Clark Y": 0.85,    # docile stall, but not as gentle at low Re
        "NACA 2412": 0.80,  # moderate
        "E423": 0.60,       # "slightly abrupt stall"
        "MH 114": 0.75,     # moderate
    }

    # Normalise each metric to [0, 1] range
    names = list(AIRFOIL_DB.keys())
    metrics = {}
    for key in ["cl_max", "ld_max", "alpha_stall_deg"]:
        vals = [AIRFOIL_DB[n][key] for n in names]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax != vmin else 1
        metrics[key] = {n: (AIRFOIL_DB[n][key] - vmin) / rng for n in names}

    # Lower CD_min is better → invert
    vals = [AIRFOIL_DB[n]["cd_min"] for n in names]
    vmin, vmax = min(vals), max(vals)
    rng = vmax - vmin if vmax != vmin else 1
    metrics["cd_min"] = {n: 1 - (AIRFOIL_DB[n]["cd_min"] - vmin) / rng for n in names}

    # Lower |Cm| is better → invert
    vals = [abs(AIRFOIL_DB[n]["cm_quarter"]) for n in names]
    vmin, vmax = min(vals), max(vals)
    rng = vmax - vmin if vmax != vmin else 1
    metrics["cm_quarter"] = {n: 1 - (abs(AIRFOIL_DB[n]["cm_quarter"]) - vmin) / rng for n in names}

    metrics["stall_gentleness"] = stall_gentleness

    # Compute weighted score
    scores = {}
    for n in names:
        score = sum(WEIGHTS[k] * metrics[k][n] for k in WEIGHTS)
        scores[n] = round(score, 3)

    return scores, metrics


def print_comparison_table():
    """Print a formatted comparison table and winner announcement."""
    scores, _ = compute_scores()

    table = []
    for name, data in AIRFOIL_DB.items():
        table.append([
            name,
            f"{data['cl_max']:.2f}",
            f"{data['cd_min']:.4f}",
            f"{data['ld_max']:.0f}",
            f"{data['alpha_stall_deg']:.1f}°",
            f"{data['cm_quarter']:.3f}",
            f"{data['thickness_pct']:.1f}%",
            f"{scores[name]:.3f}",
        ])

    headers = ["Airfoil", "CL_max", "CD_min", "L/D_max", "α_stall", "Cm_c/4", "t/c", "Score"]
    print("\n" + "=" * 90)
    print("  AIRFOIL COMPARISON TABLE  (Re ≈ 200,000)")
    print("=" * 90)
    print(tabulate(table, headers=headers, tablefmt="fancy_grid", stralign="center"))

    # Winner
    winner = max(scores, key=scores.get)
    print(f"\n  🏆  RECOMMENDED AIRFOIL: {winner}  (score: {scores[winner]:.3f})")
    print(f"  └── {AIRFOIL_DB[winner]['notes']}")
    print()


def main():
    print("\n" + "━" * 60)
    print("  ✈  AIRFOIL SELECTOR — Team NOOBLERS UAV")
    print("━" * 60)

    print("\n📊 Generating comparison plots...")
    plot_airfoil_geometries()
    plot_cl_vs_alpha()
    plot_drag_polar()
    plot_ld_ratio()

    print_comparison_table()

    print("All outputs saved to:", OUTPUT)
    print()


if __name__ == "__main__":
    main()
