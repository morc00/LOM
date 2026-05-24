#!/usr/bin/env python3
"""
Stability Analysis — Longitudinal Static Stability
=====================================================
Analyses tail volume, neutral point, static margin, elevator authority,
and CG travel limits for the NOOBLERS UAV.

Usage:
    python scripts/stability_analysis.py
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
TAIL = SPECS["tail"]
PERF = SPECS["performance"]

# ── Wing parameters ─────────────────────────────────────────────────────────
C_W = WING["chord_mm"] / 1000       # Wing MAC (m)
S_W = WING["area_m2"]                # Wing area (m²)
AR_W = WING["aspect_ratio"]
B_W = WING["span_mm"] / 1000         # Wing span (m)

# ── Tail parameters (estimated from report) ─────────────────────────────────
# Horizontal tail sizing: ~20% of wing area typical for trainer-class
S_H = 0.065                          # Horizontal tail area (m²) — estimated
C_H = 0.12                           # Horizontal tail chord (m) — estimated
L_T = TAIL["moment_arm_mm"] / 1000   # Tail moment arm (m)
AR_H = 3.5                           # Horizontal tail AR

# Vertical tail
S_V = 0.035                          # Vertical tail area (m²) — estimated
L_V = L_T                            # Vertical tail moment arm ≈ horizontal

# ── Lift curve slopes ────────────────────────────────────────────────────────
CL_ALPHA_2D = 2 * np.pi              # 2D lift curve slope (1/rad)

def cl_alpha_3d(AR, eta=0.95):
    """3D lift curve slope correction using Helmbold equation."""
    return CL_ALPHA_2D * AR / (2 + np.sqrt(4 + AR**2 * (1 + (np.tan(0))**2))) * eta


CL_ALPHA_W = cl_alpha_3d(AR_W)      # Wing 3D lift slope
CL_ALPHA_H = cl_alpha_3d(AR_H)      # Tail 3D lift slope


def tail_volume_coefficient():
    """Horizontal tail volume coefficient: V_H = (S_H * L_T) / (S_W * C_W)"""
    return (S_H * L_T) / (S_W * C_W)


def vertical_tail_volume():
    """Vertical tail volume coefficient: V_V = (S_V * L_V) / (S_W * B_W)"""
    return (S_V * L_V) / (S_W * B_W)


def neutral_point():
    """
    Estimate the neutral point (stick-fixed).
    x_np/c = x_ac_w/c + V_H * (CL_alpha_h / CL_alpha_w) * eta_h
    where x_ac_w ≈ 0.25 (quarter chord)
    """
    x_ac_w = 0.25  # wing aerodynamic centre at quarter chord
    eta_h = 0.90   # tail efficiency (downwash and dynamic pressure reduction)
    V_H = tail_volume_coefficient()
    x_np = x_ac_w + V_H * (CL_ALPHA_H / CL_ALPHA_W) * eta_h
    return x_np


def static_margin(cg_pct_chord):
    """Static margin as percentage of MAC."""
    np_pct = neutral_point() * 100
    return np_pct - cg_pct_chord


def plot_static_margin_vs_cg():
    """Plot static margin as CG moves."""
    cg_range = np.linspace(15, 55, 200)
    sm = np.array([static_margin(cg) for cg in cg_range])
    np_pct = neutral_point() * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(cg_range, sm, 'b-', linewidth=2.5, label="Static margin")

    # Safe range
    ax.axhspan(5, 15, alpha=0.12, color='green', label="Ideal range (5–15%)")
    ax.axhline(0, color='red', linewidth=2, linestyle='-', label="Neutral stability")

    # Current CG
    cg_current = PERF["cg_pct_chord"]
    sm_current = static_margin(cg_current)
    ax.plot(cg_current, sm_current, 'ro', markersize=12, zorder=5,
            label=f"Current CG: {cg_current}% → SM = {sm_current:.1f}%")

    # Neutral point
    ax.axvline(np_pct, color='purple', linestyle='--', linewidth=1.5,
               label=f"Neutral point: {np_pct:.1f}% chord")

    ax.set_xlabel("CG Position (% chord from LE)", fontsize=13)
    ax.set_ylabel("Static Margin (% chord)", fontsize=13)
    ax.set_title("Static Margin vs CG Position", fontsize=16, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "static_margin_vs_cg.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: static_margin_vs_cg.png")
    plt.close(fig)


def plot_trim_diagram():
    """Plot elevator deflection required for trim at various speeds."""
    V_stall = np.sqrt(2 * PERF["auw_g"] / 1000 * 9.81 / (1.225 * S_W * PERF["cl_max"]))
    V = np.linspace(V_stall, PERF["max_speed_ms"], 200)

    W = PERF["auw_g"] / 1000 * 9.81
    CL_required = 2 * W / (1.225 * V**2 * S_W)

    # CL at zero elevator
    CL_0_wing = 0.90  # S1223 at 0° AoA already produces lift
    CL_trim_deficiency = CL_required - CL_0_wing

    # Simplified: elevator deflection ∝ trim deficiency
    # δe ≈ CL_deficit / (V_H * CL_alpha_h * tau_e)
    V_H = tail_volume_coefficient()
    tau_e = 0.5   # elevator effectiveness factor
    delta_e_rad = CL_trim_deficiency / (V_H * CL_ALPHA_H * tau_e)
    delta_e_deg = np.rad2deg(delta_e_rad)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # CL required
    ax1.plot(V, CL_required, 'b-', linewidth=2.5, label="CL required")
    ax1.axhline(PERF["cl_max"], color='red', linestyle='--', label=f"CL_max = {PERF['cl_max']}")
    ax1.axhline(CL_0_wing, color='gray', linestyle=':', label=f"CL at α=0° ≈ {CL_0_wing}")
    ax1.set_xlabel("Airspeed (m/s)", fontsize=12)
    ax1.set_ylabel("CL required", fontsize=12)
    ax1.set_title("CL Required for Level Flight", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Elevator deflection
    ax2.plot(V, delta_e_deg, 'r-', linewidth=2.5)
    ax2.axhline(15, color='orange', linestyle='--', label="Max elevator throw (+15°)")
    ax2.axhline(-15, color='orange', linestyle='--', label="Max elevator throw (−15°)")
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_xlabel("Airspeed (m/s)", fontsize=12)
    ax2.set_ylabel("Elevator Deflection δe (°)", fontsize=12)
    ax2.set_title("Trim Elevator Deflection", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Trim Analysis — NOOBLERS UAV", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT / "trim_diagram.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: trim_diagram.png")
    plt.close(fig)


def plot_tail_sizing():
    """Show where the current tail sits on typical tail volume charts."""
    V_H = tail_volume_coefficient()
    V_V = vertical_tail_volume()

    # Reference values for different aircraft classes
    ref = {
        "Sailplane":     {"vh": 0.50, "vv": 0.020},
        "Homebuilt":     {"vh": 0.50, "vv": 0.040},
        "GA Single":     {"vh": 0.70, "vv": 0.040},
        "GA Twin":       {"vh": 0.80, "vv": 0.070},
        "Jet Trainer":   {"vh": 0.70, "vv": 0.060},
        "Model Trainer": {"vh": 0.45, "vv": 0.030},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Horizontal tail
    names = list(ref.keys())
    vh_vals = [ref[n]["vh"] for n in names]
    colors = ['#3498db'] * len(names)
    names.append("NOOBLERS UAV")
    vh_vals.append(V_H)
    colors.append('#e74c3c')

    ax1.barh(names, vh_vals, color=colors, edgecolor='white', height=0.6)
    for i, v in enumerate(vh_vals):
        ax1.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=10)
    ax1.set_xlabel("Horizontal Tail Volume (V_H)", fontsize=12)
    ax1.set_title("Horizontal Tail Volume Comparison", fontsize=14, fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # Vertical tail
    vv_vals = [ref[n]["vv"] for n in list(ref.keys())]
    names_v = list(ref.keys())
    colors_v = ['#3498db'] * len(names_v)
    names_v.append("NOOBLERS UAV")
    vv_vals.append(V_V)
    colors_v.append('#e74c3c')

    ax2.barh(names_v, vv_vals, color=colors_v, edgecolor='white', height=0.6)
    for i, v in enumerate(vv_vals):
        ax2.text(v + 0.001, i, f"{v:.3f}", va='center', fontsize=10)
    ax2.set_xlabel("Vertical Tail Volume (V_V)", fontsize=12)
    ax2.set_title("Vertical Tail Volume Comparison", fontsize=14, fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')

    fig.suptitle("Tail Sizing Check", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT / "tail_sizing.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: tail_sizing.png")
    plt.close(fig)


def print_summary():
    """Print stability summary."""
    V_H = tail_volume_coefficient()
    V_V = vertical_tail_volume()
    np_pct = neutral_point() * 100
    sm = static_margin(PERF["cg_pct_chord"])

    rows = [
        ["Wing area (S_W)", f"{S_W:.4f} m²"],
        ["Wing MAC", f"{C_W*1000:.0f} mm"],
        ["Wing AR", f"{AR_W:.1f}"],
        ["H-tail area (S_H)", f"{S_H*10000:.0f} cm²"],
        ["V-tail area (S_V)", f"{S_V*10000:.0f} cm²"],
        ["Tail moment arm", f"{L_T*1000:.0f} mm"],
        ["H-tail volume coeff (V_H)", f"{V_H:.3f}"],
        ["V-tail volume coeff (V_V)", f"{V_V:.3f}"],
        ["CL_α wing (3D)", f"{CL_ALPHA_W:.3f} /rad"],
        ["CL_α tail (3D)", f"{CL_ALPHA_H:.3f} /rad"],
        ["Neutral point", f"{np_pct:.1f}% chord"],
        ["Current CG", f"{PERF['cg_pct_chord']}% chord"],
        ["Static margin", f"{sm:.1f}% chord"],
        ["Stability assessment", "STABLE ✓" if sm > 0 else "UNSTABLE ✗"],
    ]

    print("\n" + "=" * 55)
    print("  STABILITY ANALYSIS SUMMARY")
    print("=" * 55)
    print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="fancy_grid"))

    if sm < 5:
        print("\n  ⚠ WARNING: Static margin is below 5%. Aircraft may be difficult to control.")
    elif sm > 15:
        print("\n  ⚠ NOTE: Static margin above 15%. Aircraft may feel sluggish in pitch.")
    else:
        print(f"\n  ✅ Static margin of {sm:.1f}% is within the ideal range (5–15%).")
    print()


def main():
    print("\n" + "━" * 60)
    print("  📐 STABILITY ANALYSIS — NOOBLERS UAV")
    print("━" * 60)

    print("\n📊 Generating stability plots...")
    plot_static_margin_vs_cg()
    plot_trim_diagram()
    plot_tail_sizing()
    print_summary()

    print("All outputs saved to:", OUTPUT)
    print()


if __name__ == "__main__":
    main()
