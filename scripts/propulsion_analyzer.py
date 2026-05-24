#!/usr/bin/env python3
"""
Propulsion Analyzer — Motor-Propeller-Battery System Analysis
==============================================================
Analyses the 2212/1400KV + 10×4.5 + 3S 2200mAh propulsion system
of the NOOBLERS UAV: thrust curves, efficiency, and power budget.

Usage:
    python scripts/propulsion_analyzer.py
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

PROP = SPECS["propulsion"]
MOTOR = PROP["motor"]
BATTERY = PROP["battery"]
ESC = PROP["esc"]
PROPELLER = PROP["propeller"]
PERF = SPECS["performance"]

# ── Derived constants ────────────────────────────────────────────────────────
KV = MOTOR["kv"]                           # RPM/V
V_BAT = BATTERY["voltage_nominal_V"]       # 11.1 V
V_BAT_FULL = BATTERY["voltage_full_V"]     # 12.6 V
CAP_AH = BATTERY["capacity_mAh"] / 1000   # 2.2 Ah
D_PROP_M = PROPELLER["diameter_in"] * 0.0254   # m
P_PROP_M = PROPELLER["pitch_in"] * 0.0254      # m
THRUST_STATIC_N = PROPELLER["static_thrust_g"] / 1000 * 9.81
RPM_NO_LOAD = PROP["no_load_rpm"]
RHO = 1.225  # kg/m³
G = 9.81


def thrust_vs_airspeed(V):
    """
    Simplified thrust model using momentum theory.
    T(V) = T_static * (1 - V / V_pitch)^n
    where V_pitch = pitch * RPM (prop advance speed)
    """
    V_pitch = P_PROP_M * RPM_NO_LOAD / 60  # theoretical pitch speed
    T = THRUST_STATIC_N * np.maximum(1 - (V / V_pitch)**0.85, 0)
    return T


def current_draw(throttle_pct):
    """
    Estimate motor current draw at a given throttle percentage.
    Uses quadratic model: I ≈ I_noload + k * throttle^2
    """
    I_noload = 0.8   # A
    I_max = 22.0      # A at full throttle (estimated)
    k = (I_max - I_noload)
    return I_noload + k * (throttle_pct / 100)**2


def motor_efficiency(throttle_pct):
    """
    Motor + ESC combined efficiency curve.
    Peak efficiency typically 70-80% at moderate throttle for budget motors.
    """
    # Bell curve centered around 65% throttle
    eta_peak = 0.78
    eta = eta_peak * np.exp(-0.5 * ((throttle_pct - 65) / 40)**2)
    eta = np.maximum(eta, 0.20)  # floor at 20%
    return eta


def propulsive_efficiency(V, throttle_pct=60):
    """
    Overall propulsive efficiency = thrust_power / electrical_power
    """
    T = thrust_vs_airspeed(V)
    P_thrust = T * V  # useful thrust power
    I = current_draw(throttle_pct)
    P_elec = V_BAT * I
    eta = np.where(P_elec > 0, P_thrust / P_elec, 0)
    return np.clip(eta, 0, 1)


def plot_thrust_curve():
    """Plot thrust vs airspeed."""
    V = np.linspace(0, 22, 200)
    T = thrust_vs_airspeed(V)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(V, T, 'r-', linewidth=2.5, label="Thrust available")

    # Weight line
    W = PERF["auw_g"] / 1000 * G
    ax.axhline(W, color='blue', linestyle='--', linewidth=1.5,
               label=f"Weight = {W:.2f} N ({PERF['auw_g']} g)")

    # Mark static thrust
    ax.plot(0, THRUST_STATIC_N, 'ro', markersize=10, zorder=5,
            label=f"Static thrust = {THRUST_STATIC_N:.2f} N ({PROPELLER['static_thrust_g']} g)")

    # Mark T = W intersection (max speed)
    V_max_idx = np.argmin(np.abs(T - W))
    if T[V_max_idx] >= W * 0.95:
        ax.plot(V[V_max_idx], T[V_max_idx], 'g^', markersize=12, zorder=5,
                label=f"V_max ≈ {V[V_max_idx]:.1f} m/s")

    ax.set_xlabel("Airspeed V (m/s)", fontsize=13)
    ax.set_ylabel("Thrust (N)", fontsize=13)
    ax.set_title("Thrust vs Airspeed — 2212/1400KV + 10×4.5", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, THRUST_STATIC_N * 1.1)
    fig.tight_layout()
    fig.savefig(OUTPUT / "thrust_curve.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: thrust_curve.png")
    plt.close(fig)


def plot_current_and_efficiency():
    """Plot current draw and efficiency vs throttle."""
    throttle = np.linspace(0, 100, 200)
    current = np.array([current_draw(t) for t in throttle])
    efficiency = np.array([motor_efficiency(t) for t in throttle])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Current draw
    ax1.plot(throttle, current, 'r-', linewidth=2.5)
    ax1.axhline(ESC["rating_A"], color='orange', linestyle='--', linewidth=1.5,
                label=f"ESC rating: {ESC['rating_A']} A")
    cruise_throttle = 55
    cruise_I = current_draw(cruise_throttle)
    ax1.plot(cruise_throttle, cruise_I, 'go', markersize=10, zorder=5,
             label=f"Cruise: {cruise_I:.1f} A at {cruise_throttle}% throttle")
    ax1.set_xlabel("Throttle (%)", fontsize=13)
    ax1.set_ylabel("Current Draw (A)", fontsize=13)
    ax1.set_title("Motor Current vs Throttle", fontsize=15, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)

    # Efficiency
    ax2.plot(throttle, efficiency * 100, 'g-', linewidth=2.5)
    idx_peak = np.argmax(efficiency)
    ax2.plot(throttle[idx_peak], efficiency[idx_peak] * 100, 'ro', markersize=10, zorder=5,
             label=f"Peak η = {efficiency[idx_peak]*100:.1f}% at {throttle[idx_peak]:.0f}% throttle")
    ax2.set_xlabel("Throttle (%)", fontsize=13)
    ax2.set_ylabel("Combined Efficiency (%)", fontsize=13)
    ax2.set_title("Motor + ESC Efficiency", fontsize=15, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)

    fig.suptitle("Propulsion System Analysis — NOOBLERS UAV", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT / "propulsion_efficiency.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: propulsion_efficiency.png")
    plt.close(fig)


def plot_prop_tip_speed():
    """Check propeller tip speed vs compressibility threshold."""
    throttle = np.linspace(10, 100, 200)
    rpm = RPM_NO_LOAD * throttle / 100
    # Tip speed = π * D * RPM / 60
    tip_speed = np.pi * D_PROP_M * rpm / 60
    mach = tip_speed / 343  # speed of sound ≈ 343 m/s

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(throttle, mach, 'b-', linewidth=2.5, label="Tip Mach number")
    ax.axhline(0.7, color='red', linestyle='--', linewidth=1.5,
               label="Compressibility threshold (M = 0.7)")
    ax.axhline(0.85, color='darkred', linestyle='--', linewidth=1.5,
               label="Critical Mach (M = 0.85)")

    ax.set_xlabel("Throttle (%)", fontsize=13)
    ax.set_ylabel("Tip Mach Number", fontsize=13)
    ax.set_title("Propeller Tip Speed Check", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(OUTPUT / "prop_tip_speed.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: prop_tip_speed.png")
    plt.close(fig)


def print_summary():
    """Print propulsion system summary."""
    cruise_I = current_draw(55)
    max_I = current_draw(100)

    rows = [
        ["Motor", f"2212 / {KV} KV"],
        ["Battery", f"3S {BATTERY['capacity_mAh']} mAh LiPo"],
        ["Voltage (nominal / full)", f"{V_BAT} V / {V_BAT_FULL} V"],
        ["Propeller", f"{PROPELLER['diameter_in']}×{PROPELLER['pitch_in']}"],
        ["Static thrust", f"{THRUST_STATIC_N:.2f} N ({PROPELLER['static_thrust_g']} g)"],
        ["Thrust-to-weight", f"{PROP['thrust_to_weight']:.2f} : 1"],
        ["No-load RPM", f"{RPM_NO_LOAD:,}"],
        ["Cruise current (~55% throttle)", f"{cruise_I:.1f} A"],
        ["Max current (100% throttle)", f"{max_I:.1f} A"],
        ["ESC headroom", f"{ESC['rating_A'] - max_I:.1f} A"],
        ["Prop tip speed (100%)", f"{np.pi * D_PROP_M * RPM_NO_LOAD / 60:.1f} m/s"],
        ["Prop tip Mach (100%)", f"{np.pi * D_PROP_M * RPM_NO_LOAD / 60 / 343:.3f}"],
    ]

    print("\n" + "=" * 55)
    print("  PROPULSION SYSTEM SUMMARY")
    print("=" * 55)
    print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
    print()


def main():
    print("\n" + "━" * 60)
    print("  ⚡ PROPULSION ANALYZER — NOOBLERS UAV")
    print("━" * 60)

    print("\n📊 Generating propulsion plots...")
    plot_thrust_curve()
    plot_current_and_efficiency()
    plot_prop_tip_speed()
    print_summary()

    print("All outputs saved to:", OUTPUT)
    print()


if __name__ == "__main__":
    main()
