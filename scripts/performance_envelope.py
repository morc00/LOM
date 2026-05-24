#!/usr/bin/env python3
"""
Performance Envelope — Flight Performance Analysis
====================================================
Computes V-n diagram, power curves, rate of climb, and stall speeds
for the NOOBLERS UAV across its flight envelope.

Usage:
    python scripts/performance_envelope.py
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

# ── Constants ────────────────────────────────────────────────────────────────
RHO = 1.225         # kg/m³ (sea level ISA)
G = 9.81             # m/s²
WING = SPECS["wing"]
PERF = SPECS["performance"]
PROP = SPECS["propulsion"]

W = PERF["auw_g"] / 1000 * G     # Weight in N
S = WING["area_m2"]                # Wing area
AR = WING["aspect_ratio"]
CHORD = WING["chord_mm"] / 1000
CL_MAX = PERF["cl_max"]
CD_MIN = 0.014
E_OSWALD = 0.75
CL_0 = 0.90
THRUST_MAX_N = PROP["propeller"]["static_thrust_g"] / 1000 * G  # static thrust in N

# Load factor limits (typical for small UAV)
N_POS_MAX = 3.0
N_NEG_MAX = -1.5


def stall_speed(n=1.0, cl_max=CL_MAX):
    """Compute stall speed at load factor n."""
    return np.sqrt(2 * n * W / (RHO * S * cl_max))


def drag(V, n=1.0):
    """Total drag at speed V and load factor n."""
    cl = 2 * n * W / (RHO * V**2 * S)
    cl_min_d = 0.5 * CL_0
    k = 0.0035
    cd_profile = CD_MIN + k * (cl - cl_min_d)**2
    cd_induced = cl**2 / (np.pi * AR * E_OSWALD)
    cd = cd_profile + cd_induced
    D = 0.5 * RHO * V**2 * S * cd
    return D


def power_required(V, n=1.0):
    """Power required for level flight at speed V."""
    D = drag(V, n)
    return D * V


def thrust_available(V):
    """
    Simplified thrust model: static thrust decreasing linearly with speed.
    T(V) = T_static * (1 - V/V_max_prop)
    """
    V_max_prop = 25.0  # m/s approx prop washout speed
    T = THRUST_MAX_N * np.maximum(1 - V / V_max_prop, 0)
    return T


def power_available(V):
    """Power available from propulsion system."""
    return thrust_available(V) * V


def rate_of_climb(V):
    """Rate of climb: ROC = (P_avail - P_req) / W."""
    P_excess = power_available(V) - power_required(V)
    return P_excess / W


def plot_vn_diagram():
    """Generate V-n (velocity vs load factor) diagram."""
    fig, ax = plt.subplots(figsize=(12, 7))

    V_s1 = stall_speed(1.0)
    V_max = PERF["max_speed_ms"]

    # Positive stall boundary
    V_pos = np.linspace(V_s1, V_max * 1.1, 200)
    n_pos_stall = 0.5 * RHO * V_pos**2 * S * CL_MAX / W
    n_pos_stall = np.minimum(n_pos_stall, N_POS_MAX)

    # Negative stall boundary (CL_max_neg ≈ -1.0 for cambered airfoil)
    CL_MAX_NEG = 1.0
    V_neg = np.linspace(V_s1 * 0.7, V_max * 1.1, 200)
    n_neg_stall = -0.5 * RHO * V_neg**2 * S * CL_MAX_NEG / W
    n_neg_stall = np.maximum(n_neg_stall, N_NEG_MAX)

    # Plot positive envelope
    ax.plot(V_pos, n_pos_stall, 'b-', linewidth=2.5, label="Positive stall limit")
    ax.plot([V_max, V_max], [N_NEG_MAX, N_POS_MAX], 'r-', linewidth=2, label=f"V_max = {V_max} m/s")
    ax.plot([V_pos[np.argmin(np.abs(n_pos_stall - N_POS_MAX))], V_max],
            [N_POS_MAX, N_POS_MAX], 'b-', linewidth=2.5)

    # Plot negative envelope
    ax.plot(V_neg, n_neg_stall, 'b--', linewidth=2, label="Negative stall limit")
    ax.plot([V_neg[np.argmin(np.abs(n_neg_stall - N_NEG_MAX))], V_max],
            [N_NEG_MAX, N_NEG_MAX], 'b--', linewidth=2)

    # Mark key speeds
    ax.axvline(V_s1, color='orange', linestyle=':', linewidth=1.5,
               label=f"V_stall = {V_s1:.1f} m/s")
    V_cruise = np.mean(PERF["cruise_speed_ms"])
    ax.axvline(V_cruise, color='green', linestyle=':', linewidth=1.5,
               label=f"V_cruise = {V_cruise:.0f} m/s")

    # n=1 line
    ax.axhline(1, color='gray', linewidth=0.8, linestyle='-')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    ax.set_xlabel("Airspeed V (m/s)", fontsize=13)
    ax.set_ylabel("Load Factor n", fontsize=13)
    ax.set_title("V-n Diagram — NOOBLERS UAV", fontsize=16, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, V_max * 1.2)
    ax.set_ylim(N_NEG_MAX - 0.5, N_POS_MAX + 0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT / "vn_diagram.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: vn_diagram.png")
    plt.close(fig)


def plot_power_curves():
    """Plot power required vs power available."""
    V = np.linspace(stall_speed(1.0), PERF["max_speed_ms"], 200)
    P_req = np.array([power_required(v) for v in V])
    P_avail = np.array([power_available(v) for v in V])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Power curves
    ax1.plot(V, P_req, 'b-', linewidth=2.5, label="Power Required")
    ax1.plot(V, P_avail, 'r-', linewidth=2.5, label="Power Available")
    ax1.fill_between(V, P_req, P_avail, where=P_avail > P_req,
                     alpha=0.15, color='green', label="Excess power")

    # Mark minimum power required
    idx_min = np.argmin(P_req)
    ax1.plot(V[idx_min], P_req[idx_min], 'go', markersize=10, zorder=5,
             label=f"Min power: {P_req[idx_min]:.1f} W at {V[idx_min]:.1f} m/s")

    ax1.set_xlabel("Airspeed V (m/s)", fontsize=13)
    ax1.set_ylabel("Power (W)", fontsize=13)
    ax1.set_title("Power Required vs Available", fontsize=15, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Rate of climb
    roc = np.array([rate_of_climb(v) for v in V])
    ax2.plot(V, roc, 'g-', linewidth=2.5)
    ax2.fill_between(V, 0, roc, where=roc > 0, alpha=0.15, color='green')
    idx_max_roc = np.argmax(roc)
    ax2.plot(V[idx_max_roc], roc[idx_max_roc], 'ro', markersize=10, zorder=5,
             label=f"Max ROC: {roc[idx_max_roc]:.2f} m/s at V = {V[idx_max_roc]:.1f} m/s")
    ax2.axhline(0, color='gray', linewidth=0.8)

    ax2.set_xlabel("Airspeed V (m/s)", fontsize=13)
    ax2.set_ylabel("Rate of Climb (m/s)", fontsize=13)
    ax2.set_title("Rate of Climb vs Airspeed", fontsize=15, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("NOOBLERS UAV — Performance Analysis", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT / "power_and_roc.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: power_and_roc.png")
    plt.close(fig)


def plot_stall_speeds():
    """Plot stall speed vs bank angle and load factor."""
    bank_angles = np.linspace(0, 60, 100)
    load_factors = 1 / np.cos(np.deg2rad(bank_angles))
    v_stall = np.array([stall_speed(n) for n in load_factors])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bank_angles, v_stall, 'b-', linewidth=2.5)
    ax.axhline(stall_speed(1.0), color='gray', linestyle='--', linewidth=1,
               label=f"Level stall = {stall_speed(1.0):.1f} m/s")

    # Mark common bank angles
    for angle in [15, 30, 45, 60]:
        n = 1 / np.cos(np.deg2rad(angle))
        vs = stall_speed(n)
        ax.plot(angle, vs, 'ro', markersize=8)
        ax.annotate(f"{vs:.1f} m/s\n(n={n:.2f})",
                    xy=(angle, vs), xytext=(angle + 3, vs + 0.3),
                    fontsize=9)

    ax.set_xlabel("Bank Angle (°)", fontsize=13)
    ax.set_ylabel("Stall Speed (m/s)", fontsize=13)
    ax.set_title("Stall Speed vs Bank Angle — NOOBLERS UAV", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "stall_speed_vs_bank.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: stall_speed_vs_bank.png")
    plt.close(fig)


def print_summary():
    """Print performance summary table."""
    V = np.linspace(stall_speed(1.0), PERF["max_speed_ms"], 200)
    P_req = np.array([power_required(v) for v in V])
    roc = np.array([rate_of_climb(v) for v in V])

    V_s = stall_speed(1.0)
    V_best_glide = V[np.argmax(V / np.array([drag(v) for v in V]) * W)]
    idx_max_roc = np.argmax(roc)

    rows = [
        ["Stall speed (1g)", f"{V_s:.1f} m/s"],
        ["Stall speed (30° bank)", f"{stall_speed(1/np.cos(np.deg2rad(30))):.1f} m/s"],
        ["Stall speed (45° bank)", f"{stall_speed(1/np.cos(np.deg2rad(45))):.1f} m/s"],
        ["Min power speed", f"{V[np.argmin(P_req)]:.1f} m/s"],
        ["Min power required", f"{min(P_req):.1f} W"],
        ["Best climb speed (Vy)", f"{V[idx_max_roc]:.1f} m/s"],
        ["Max rate of climb", f"{roc[idx_max_roc]:.2f} m/s"],
        ["Max level speed", f"{PERF['max_speed_ms']} m/s"],
        ["Positive load limit", f"+{N_POS_MAX:.1f}g"],
        ["Negative load limit", f"{N_NEG_MAX:.1f}g"],
    ]

    print("\n" + "=" * 50)
    print("  PERFORMANCE ENVELOPE SUMMARY")
    print("=" * 50)
    print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
    print()


def main():
    print("\n" + "━" * 60)
    print("  ✈  PERFORMANCE ENVELOPE — NOOBLERS UAV")
    print("━" * 60)

    print("\n📊 Generating performance plots...")
    plot_vn_diagram()
    plot_power_curves()
    plot_stall_speeds()
    print_summary()

    print("All outputs saved to:", OUTPUT)
    print()


if __name__ == "__main__":
    main()
