#!/usr/bin/env python3
"""
Wing Loading Calculator — Wing Design Analysis
=================================================
Analyses wing loading, lift distribution, stall speed sensitivity,
and aspect ratio trade-offs for the NOOBLERS UAV.

Usage:
    python scripts/wing_loading_calculator.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

with open(DATA / "specs.json") as f:
    SPECS = json.load(f)

WING = SPECS["wing"]
PERF = SPECS["performance"]

RHO = 1.225   # kg/m³
G = 9.81      # m/s²

W = PERF["auw_g"] / 1000 * G       # Weight (N)
S = WING["area_m2"]                  # Wing area (m²)
B = WING["span_mm"] / 1000          # Span (m)
C = WING["chord_mm"] / 1000         # Chord (m)
AR = WING["aspect_ratio"]
CL_MAX = PERF["cl_max"]

# ── Reference aircraft for comparison ────────────────────────────────────────
REFERENCE_AIRCRAFT = [
    {"name": "NOOBLERS UAV", "wl_npm2": 24.8, "auw_g": 790, "highlight": True},
    {"name": "DLG Glider (typical)", "wl_npm2": 10, "auw_g": 300, "highlight": False},
    {"name": "Park Flyer (typical)", "wl_npm2": 20, "auw_g": 500, "highlight": False},
    {"name": "Trainer RC (typical)", "wl_npm2": 35, "auw_g": 1200, "highlight": False},
    {"name": "FPV Racing Quad", "wl_npm2": 100, "auw_g": 600, "highlight": False},
    {"name": "Bixler 2 (HobbyKing)", "wl_npm2": 28, "auw_g": 870, "highlight": False},
    {"name": "Mini Talon", "wl_npm2": 42, "auw_g": 1500, "highlight": False},
]


def lift_distribution(y_span, planform="rectangular"):
    """
    Compute spanwise lift distribution.
    For rectangular wing, use a corrected elliptic + linear approximation.
    y_span: array of spanwise stations from 0 (root) to b/2 (tip)
    """
    b_half = B / 2
    y_norm = y_span / b_half  # normalised 0..1

    if planform == "rectangular":
        # Rectangular wing produces roughly trapezoidal distribution
        # with slight rounding at tips
        gamma_elliptic = np.sqrt(1 - y_norm**2)
        gamma_rect = 1 - 0.1 * y_norm**2  # slight taper
        gamma = 0.6 * gamma_elliptic + 0.4 * gamma_rect
    else:
        gamma = np.sqrt(1 - y_norm**2)

    # Normalise to total lift
    gamma = gamma / np.trapz(gamma, y_span) * (W / 2)
    return gamma


def plot_wing_loading_comparison():
    """Compare wing loading against reference aircraft."""
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [a["name"] for a in REFERENCE_AIRCRAFT]
    wl = [a["wl_npm2"] for a in REFERENCE_AIRCRAFT]
    colors = ['#e74c3c' if a["highlight"] else '#3498db' for a in REFERENCE_AIRCRAFT]

    bars = ax.barh(names, wl, color=colors, edgecolor='white', linewidth=1.5, height=0.6)

    for bar, val in zip(bars, wl):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} N/m²", va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel("Wing Loading (N/m²)", fontsize=13)
    ax.set_title("Wing Loading Comparison", fontsize=16, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(OUTPUT / "wing_loading_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: wing_loading_comparison.png")
    plt.close(fig)


def plot_lift_distribution():
    """Plot spanwise lift distribution."""
    y = np.linspace(0, B / 2, 200)
    gamma_rect = lift_distribution(y, "rectangular")
    gamma_elliptic = lift_distribution(y, "elliptic")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(y * 1000, gamma_rect, 'b-', linewidth=2.5, label="Rectangular wing (actual)")
    ax.plot(y * 1000, gamma_elliptic, 'g--', linewidth=2, label="Elliptic (ideal)")

    ax.fill_between(y * 1000, 0, gamma_rect, alpha=0.1, color='blue')
    ax.set_xlabel("Spanwise Station (mm from root)", fontsize=13)
    ax.set_ylabel("Lift per Unit Span (N/m)", fontsize=13)
    ax.set_title("Spanwise Lift Distribution", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "lift_distribution.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: lift_distribution.png")
    plt.close(fig)


def plot_stall_sensitivity():
    """Plot stall speed sensitivity to weight and wing area."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Stall vs weight
    weights_g = np.linspace(500, 1200, 200)
    weights_N = weights_g / 1000 * G
    v_stall_w = np.sqrt(2 * weights_N / (RHO * S * CL_MAX))

    ax1.plot(weights_g, v_stall_w, 'b-', linewidth=2.5)
    ax1.plot(PERF["auw_g"], np.sqrt(2 * W / (RHO * S * CL_MAX)), 'ro', markersize=10,
             label=f"Current: {PERF['auw_g']}g → {np.sqrt(2*W/(RHO*S*CL_MAX)):.1f} m/s")
    ax1.set_xlabel("All-Up Weight (g)", fontsize=12)
    ax1.set_ylabel("Stall Speed (m/s)", fontsize=12)
    ax1.set_title("Stall Speed vs Weight", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Stall vs wing area
    areas = np.linspace(0.15, 0.50, 200)
    v_stall_s = np.sqrt(2 * W / (RHO * areas * CL_MAX))

    ax2.plot(areas * 10000, v_stall_s, 'r-', linewidth=2.5)
    ax2.plot(S * 10000, np.sqrt(2 * W / (RHO * S * CL_MAX)), 'bo', markersize=10,
             label=f"Current: {S*10000:.0f} cm² → {np.sqrt(2*W/(RHO*S*CL_MAX)):.1f} m/s")
    ax2.set_xlabel("Wing Area (cm²)", fontsize=12)
    ax2.set_ylabel("Stall Speed (m/s)", fontsize=12)
    ax2.set_title("Stall Speed vs Wing Area", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Stall Speed Sensitivity Analysis", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT / "stall_sensitivity.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: stall_sensitivity.png")
    plt.close(fig)


def plot_ar_trade():
    """Aspect ratio trade study: induced drag vs structural weight."""
    ar_range = np.linspace(2, 10, 200)
    e = 0.75
    CL_cruise = 1.0  # typical cruise CL

    # Induced drag coefficient
    cdi = CL_cruise**2 / (np.pi * ar_range * e)

    # Structural weight penalty (relative) — higher AR = heavier wing
    w_struct_rel = 1 + 0.05 * (ar_range - 4)**2  # normalised

    fig, ax1 = plt.subplots(figsize=(10, 7))
    color1 = '#2980b9'
    color2 = '#e74c3c'

    ax1.plot(ar_range, cdi, color=color1, linewidth=2.5, label="Induced drag coefficient")
    ax1.set_xlabel("Aspect Ratio", fontsize=13)
    ax1.set_ylabel("CDi (at CL = 1.0)", fontsize=13, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(ar_range, w_struct_rel, color=color2, linewidth=2.5, linestyle='--',
             label="Relative structural weight")
    ax2.set_ylabel("Relative Wing Weight", fontsize=13, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Mark current AR
    ax1.axvline(AR, color='green', linestyle=':', linewidth=2,
                label=f"Current AR = {AR}")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper right")

    ax1.set_title("Aspect Ratio Trade Study", fontsize=15, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "ar_trade_study.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: ar_trade_study.png")
    plt.close(fig)


def print_summary():
    """Print wing loading summary."""
    v_stall = np.sqrt(2 * W / (RHO * S * CL_MAX))

    rows = [
        ["Wing span", f"{B*1000:.0f} mm"],
        ["Chord", f"{C*1000:.0f} mm"],
        ["Wing area", f"{S:.4f} m² ({S*10000:.0f} cm²)"],
        ["Aspect ratio", f"{AR:.1f}"],
        ["Planform", WING["planform"].title()],
        ["Wing loading", f"{PERF['wing_loading_Npm2']} N/m²"],
        ["CL_max", f"{CL_MAX:.2f}"],
        ["Stall speed (1g)", f"{v_stall:.1f} m/s"],
        ["AUW", f"{PERF['auw_g']} g"],
    ]

    print("\n" + "=" * 50)
    print("  WING LOADING SUMMARY")
    print("=" * 50)
    print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
    print()


def main():
    print("\n" + "━" * 60)
    print("  🛩  WING LOADING CALCULATOR — NOOBLERS UAV")
    print("━" * 60)

    print("\n📊 Generating wing analysis plots...")
    plot_wing_loading_comparison()
    plot_lift_distribution()
    plot_stall_sensitivity()
    plot_ar_trade()
    print_summary()

    print("All outputs saved to:", OUTPUT)
    print()


if __name__ == "__main__":
    main()
