#!/usr/bin/env python3
"""
Lift & Drag Calculator — Aerodynamic Coefficient Analysis
==========================================================
Analyses the Selig S1223 airfoil performance across the full angle-of-attack
range and Reynolds number conditions relevant to the NOOBLERS UAV.

Usage:
    python scripts/lift_drag_calculator.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# ── Load aircraft specs ─────────────────────────────────────────────────────
with open(DATA / "specs.json") as f:
    SPECS = json.load(f)

WING = SPECS["wing"]
PERF = SPECS["performance"]

# ── Airfoil parameters (S1223 at Re ≈ 200,000) ─────────────────────────────
CL_MAX = 2.20
CL_0 = 0.90                    # CL at zero alpha (cambered airfoil)
ALPHA_0_DEG = -8.0              # Zero-lift angle of attack
ALPHA_STALL_DEG = 14.0
CD_MIN = 0.014
CM_C4 = -0.13
CL_ALPHA = 2 * np.pi * 0.85    # ~5.34 /rad with viscous correction
AR = WING["aspect_ratio"]
E_OSWALD = 0.75                 # Oswald efficiency factor
S = WING["area_m2"]
CHORD = WING["chord_mm"] / 1000  # m


def cl_vs_alpha(alpha_deg):
    """Compute CL vs angle of attack with soft-stall model."""
    alpha_rad = np.deg2rad(alpha_deg)
    alpha_0_rad = np.deg2rad(ALPHA_0_DEG)
    alpha_stall_rad = np.deg2rad(ALPHA_STALL_DEG)

    cl = CL_ALPHA * (alpha_rad - alpha_0_rad)
    # Soft stall beyond CL_max
    mask = cl > CL_MAX
    excess = alpha_rad[mask] - alpha_stall_rad
    cl[mask] = CL_MAX - 2.0 * excess**2
    return cl


def cd_total(cl_array):
    """Total drag coefficient: profile + induced."""
    cl_min_drag = 0.5 * CL_0
    k = 0.0035  # profile drag form factor
    cd_profile = CD_MIN + k * (cl_array - cl_min_drag)**2
    cd_induced = cl_array**2 / (np.pi * AR * E_OSWALD)
    return cd_profile + cd_induced


def reynolds_number(velocity, chord, nu=1.516e-5):
    """Compute Reynolds number. nu = kinematic viscosity of air at 20°C."""
    return velocity * chord / nu


def plot_cl_alpha():
    """Plot CL vs angle of attack."""
    alpha = np.linspace(-5, 22, 300)
    cl = cl_vs_alpha(alpha)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(alpha, cl, 'b-', linewidth=2.5, label="S1223 — CL vs α")

    # Mark key points
    idx_max = np.argmax(cl)
    ax.plot(alpha[idx_max], cl[idx_max], 'ro', markersize=10, zorder=5,
            label=f"CL_max = {cl[idx_max]:.2f} at α = {alpha[idx_max]:.1f}°")

    # Mark zero-lift
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(ALPHA_0_DEG, color='gray', linewidth=0.5, linestyle='--',
               label=f"Zero-lift α = {ALPHA_0_DEG}°")

    # Mark cruise region
    ax.axhspan(0.8, 1.2, alpha=0.1, color='green', label="Cruise CL range")

    ax.set_xlabel("Angle of Attack α (°)", fontsize=13)
    ax.set_ylabel("Lift Coefficient CL", fontsize=13)
    ax.set_title("Selig S1223 — Lift Curve", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "s1223_cl_vs_alpha.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: s1223_cl_vs_alpha.png")
    plt.close(fig)


def plot_drag_polar():
    """Plot CL vs CD drag polar."""
    cl = np.linspace(0, CL_MAX, 200)
    cd = cd_total(cl)
    ld = cl / cd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Drag polar
    ax1.plot(cd, cl, 'b-', linewidth=2.5)
    idx_best_ld = np.argmax(ld)
    ax1.plot(cd[idx_best_ld], cl[idx_best_ld], 'go', markersize=10, zorder=5,
             label=f"Best L/D point (CL={cl[idx_best_ld]:.2f})")
    # Tangent line from origin
    ax1.plot([0, cd[idx_best_ld]*1.5], [0, cl[idx_best_ld]*1.5], 'g--', alpha=0.5)
    ax1.set_xlabel("Drag Coefficient CD", fontsize=13)
    ax1.set_ylabel("Lift Coefficient CL", fontsize=13)
    ax1.set_title("Drag Polar", fontsize=15, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # L/D vs CL
    ax2.plot(cl, ld, 'r-', linewidth=2.5)
    ax2.plot(cl[idx_best_ld], ld[idx_best_ld], 'go', markersize=10, zorder=5,
             label=f"Max L/D = {ld[idx_best_ld]:.1f} at CL = {cl[idx_best_ld]:.2f}")
    ax2.set_xlabel("Lift Coefficient CL", fontsize=13)
    ax2.set_ylabel("Lift-to-Drag Ratio (L/D)", fontsize=13)
    ax2.set_title("Aerodynamic Efficiency", fontsize=15, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Selig S1223 — Drag Analysis (AR = 4.3, e = 0.75)", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT / "s1223_drag_polar.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: s1223_drag_polar.png")
    plt.close(fig)


def plot_reynolds_sensitivity():
    """Show how CL_max varies with Reynolds number."""
    re_range = np.array([50000, 100000, 150000, 200000, 250000, 300000])
    # Empirical CL_max correction for Re effects
    cl_max_ref = 2.20
    re_ref = 200000
    cl_max_vals = cl_max_ref * (1 - 0.15 * np.log10(re_ref / re_range))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(re_range / 1000, cl_max_vals, 'bo-', linewidth=2, markersize=8)
    ax.axvline(PERF["cruise_reynolds"] / 1000, color='green', linestyle='--',
               linewidth=1.5, label=f"Cruise Re = {PERF['cruise_reynolds']:,}")
    ax.fill_between([50, 300], cl_max_vals.min() * 0.95, cl_max_vals.max() * 1.02,
                    alpha=0.05, color='blue')
    ax.set_xlabel("Reynolds Number (×10³)", fontsize=13)
    ax.set_ylabel("CL_max", fontsize=13)
    ax.set_title("S1223 — CL_max vs Reynolds Number", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "s1223_reynolds_sensitivity.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: s1223_reynolds_sensitivity.png")
    plt.close(fig)


def print_summary():
    """Print a summary table of aerodynamic characteristics."""
    cl = np.linspace(0.1, CL_MAX, 200)
    cd = cd_total(cl)
    ld = cl / cd
    idx = np.argmax(ld)

    print("\n" + "=" * 60)
    print("  S1223 AERODYNAMIC SUMMARY")
    print("=" * 60)
    rows = [
        ["CL_max", f"{CL_MAX:.2f}", "—"],
        ["CL at zero α", f"{CL_0:.2f}", "—"],
        ["Zero-lift α", f"{ALPHA_0_DEG:.1f}°", "—"],
        ["Stall α", f"{ALPHA_STALL_DEG:.1f}°", "—"],
        ["CD_min (profile)", f"{CD_MIN:.4f}", "—"],
        ["Best L/D", f"{ld[idx]:.1f}", f"at CL = {cl[idx]:.2f}"],
        ["CD at best L/D", f"{cd[idx]:.4f}", "—"],
        ["Cm_c/4", f"{CM_C4:.3f}", "—"],
        ["Oswald efficiency", f"{E_OSWALD:.2f}", "—"],
        ["Aspect ratio", f"{AR:.1f}", "—"],
        ["Cruise Re", f"{PERF['cruise_reynolds']:,}", f"at chord = {CHORD*1000:.0f} mm"],
    ]
    from tabulate import tabulate
    print(tabulate(rows, headers=["Parameter", "Value", "Note"], tablefmt="fancy_grid"))
    print()


def main():
    print("\n" + "━" * 60)
    print("  ✈  LIFT & DRAG CALCULATOR — S1223 Analysis")
    print("━" * 60)

    print("\n📊 Generating aerodynamic plots...")
    plot_cl_alpha()
    plot_drag_polar()
    plot_reynolds_sensitivity()
    print_summary()

    print("All outputs saved to:", OUTPUT)
    print()


if __name__ == "__main__":
    main()
