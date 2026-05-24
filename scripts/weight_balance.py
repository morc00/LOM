#!/usr/bin/env python3
"""
Weight & Balance Calculator
============================
Computes weight breakdown, CG position, and static margin
for the NOOBLERS UAV. Includes "what-if" battery position analysis.

Usage:
    python scripts/weight_balance.py
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

WEIGHT = SPECS["weight_breakdown_g"]
WING = SPECS["wing"]
PERF = SPECS["performance"]
TAIL = SPECS["tail"]

CHORD_MM = WING["chord_mm"]   # 270 mm
CG_TARGET_MM = PERF["cg_from_le_mm"]  # 90 mm from LE = 33% chord

# ── Component positions (mm from nose) — estimated from report ───────────────
# Fuselage is ~850 mm total. Wing at ~200 mm from nose (leading edge).
WING_LE_FROM_NOSE = 200  # mm

COMPONENT_POSITIONS_MM = {
    "wing":                     WING_LE_FROM_NOSE + 0.40 * CHORD_MM,    # 40% chord
    "fuselage":                 350,                                      # ~mid fuselage
    "tail_assembly":            750,                                      # near tail
    "motor":                    50,                                       # nose
    "esc":                      180,                                      # behind motor
    "battery":                  WING_LE_FROM_NOSE + CG_TARGET_MM - 20,   # adjustable on rail
    "receiver":                 300,                                      # mid fuselage
    "servos":                   WING_LE_FROM_NOSE + 0.60 * CHORD_MM,    # near trailing edge
    "propeller_and_mount":      25,                                       # nose
    "wiring_connectors_hardware": 300,                                    # distributed
    "payload":                  400,                                      # payload bay
}


def compute_cg(positions=None):
    """Compute CG position from component masses and positions."""
    if positions is None:
        positions = COMPONENT_POSITIONS_MM

    total_mass = sum(WEIGHT.values())
    moment = sum(WEIGHT[comp] * positions[comp] for comp in WEIGHT)
    cg_from_nose = moment / total_mass
    cg_from_le = cg_from_nose - WING_LE_FROM_NOSE
    cg_pct_chord = cg_from_le / CHORD_MM * 100

    return {
        "cg_from_nose_mm": cg_from_nose,
        "cg_from_le_mm": cg_from_le,
        "cg_pct_chord": cg_pct_chord,
        "total_mass_g": total_mass,
    }


def plot_weight_breakdown():
    """Generate pie chart and bar chart of weight breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    labels = [k.replace("_", " ").title() for k in WEIGHT.keys()]
    masses = list(WEIGHT.values())
    total = sum(masses)
    percentages = [m / total * 100 for m in masses]

    # Colour palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(masses)))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        masses, labels=None, autopct='%1.1f%%',
        startangle=140, colors=colors, pctdistance=0.8,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax1.legend(labels, loc="center left", bbox_to_anchor=(-0.3, 0.5), fontsize=9)
    ax1.set_title(f"Weight Breakdown (AUW = {total} g)", fontsize=14, fontweight="bold")

    # Bar chart
    y_pos = np.arange(len(labels))
    bars = ax2.barh(y_pos, masses, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel("Mass (g)", fontsize=12)
    ax2.set_title("Component Masses", fontsize=14, fontweight="bold")
    ax2.invert_yaxis()

    # Add value labels on bars
    for bar, mass, pct in zip(bars, masses, percentages):
        ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                 f"{mass}g ({pct:.1f}%)", va='center', fontsize=9)

    fig.suptitle("NOOBLERS UAV — Weight Statement", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT / "weight_breakdown.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: weight_breakdown.png")
    plt.close(fig)


def plot_cg_diagram():
    """Visualise CG position on the aircraft planform."""
    cg = compute_cg()

    fig, ax = plt.subplots(figsize=(14, 5))

    # Draw simplified side view
    fuselage_length = 850
    ax.plot([0, fuselage_length], [0, 0], 'k-', linewidth=2)

    # Wing box
    wing_le = WING_LE_FROM_NOSE
    wing_te = wing_le + CHORD_MM
    ax.fill([wing_le, wing_te, wing_te, wing_le],
            [-8, -8, 8, 8], alpha=0.3, color='blue', label="Wing")
    ax.plot([wing_le, wing_te, wing_te, wing_le, wing_le],
            [-8, -8, 8, 8, -8], 'b-', linewidth=1.5)

    # Tail
    tail_start = 700
    ax.fill([tail_start, 850, 850, tail_start],
            [-5, -5, 5, 5], alpha=0.2, color='gray', label="Tail")

    # Motor/Prop
    ax.plot(25, 0, 'rs', markersize=12, label="Motor/Prop")

    # CG marker
    cg_pos = cg["cg_from_nose_mm"]
    ax.plot(cg_pos, 0, 'r^', markersize=18, zorder=10, label=f"CG = {cg_pos:.0f} mm from nose")
    ax.annotate(f"CG\n{cg['cg_pct_chord']:.1f}% chord",
                xy=(cg_pos, 0), xytext=(cg_pos, 18),
                ha='center', fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Neutral point (estimated: CG + static margin)
    np_pos = wing_le + CHORD_MM * 0.40  # ~40% chord assumed neutral point
    ax.plot(np_pos, 0, 'gv', markersize=14, zorder=10, label=f"NP ≈ {np_pos:.0f} mm from nose")

    # Component positions
    for comp, pos in COMPONENT_POSITIONS_MM.items():
        if comp not in ["wing", "fuselage", "tail_assembly"]:
            mass = WEIGHT[comp]
            ax.plot(pos, -15, 'ko', markersize=max(4, mass / 20), alpha=0.6)

    ax.set_xlabel("Distance from Nose (mm)", fontsize=12)
    ax.set_title("Side View — CG & Component Layout", fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-30, 900)
    ax.set_ylim(-25, 30)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(OUTPUT / "cg_diagram.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: cg_diagram.png")
    plt.close(fig)


def plot_battery_cg_sensitivity():
    """What-if: how CG shifts as battery position changes."""
    battery_offsets = np.linspace(-80, 80, 200)  # mm from current position
    cg_pcts = []

    for offset in battery_offsets:
        pos = COMPONENT_POSITIONS_MM.copy()
        pos["battery"] = COMPONENT_POSITIONS_MM["battery"] + offset
        cg = compute_cg(pos)
        cg_pcts.append(cg["cg_pct_chord"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(battery_offsets, cg_pcts, 'b-', linewidth=2.5)

    # Safe CG range (25% — 35% chord)
    ax.axhspan(25, 35, alpha=0.15, color='green', label="Safe CG range (25–35% chord)")
    ax.axhline(33, color='green', linestyle='--', linewidth=1, label="Target CG (33%)")

    # Current position
    ax.plot(0, compute_cg()["cg_pct_chord"], 'ro', markersize=10, zorder=5,
            label=f"Current: {compute_cg()['cg_pct_chord']:.1f}% chord")

    ax.set_xlabel("Battery Position Offset (mm, + = aft)", fontsize=13)
    ax.set_ylabel("CG Position (% chord)", fontsize=13)
    ax.set_title("CG Sensitivity to Battery Position", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "battery_cg_sensitivity.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: battery_cg_sensitivity.png")
    plt.close(fig)


def print_summary():
    """Print CG and weight summary."""
    cg = compute_cg()

    static_margin_fwd = 40 - cg["cg_pct_chord"]  # NP at ~40% chord

    rows = [
        ["Total AUW", f"{cg['total_mass_g']} g"],
        ["CG from nose", f"{cg['cg_from_nose_mm']:.1f} mm"],
        ["CG from wing LE", f"{cg['cg_from_le_mm']:.1f} mm"],
        ["CG as % chord", f"{cg['cg_pct_chord']:.1f}%"],
        ["Target CG", f"{CG_TARGET_MM} mm ({PERF['cg_pct_chord']}% chord)"],
        ["Estimated static margin", f"{static_margin_fwd:.1f}% chord"],
        ["Heaviest component", f"{max(WEIGHT, key=WEIGHT.get).replace('_', ' ').title()} ({max(WEIGHT.values())} g)"],
    ]

    print("\n" + "=" * 55)
    print("  WEIGHT & BALANCE SUMMARY")
    print("=" * 55)
    print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
    print()


def main():
    print("\n" + "━" * 60)
    print("  ⚖  WEIGHT & BALANCE — NOOBLERS UAV")
    print("━" * 60)

    print("\n📊 Generating weight & balance plots...")
    plot_weight_breakdown()
    plot_cg_diagram()
    plot_battery_cg_sensitivity()
    print_summary()

    print("All outputs saved to:", OUTPUT)
    print()


if __name__ == "__main__":
    main()
