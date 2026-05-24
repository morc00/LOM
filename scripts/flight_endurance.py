#!/usr/bin/env python3
"""
Flight Endurance Estimator — Battery & Mission Analysis
=========================================================
Estimates flight endurance, range, and simulates a complete
mission profile for the NOOBLERS UAV.

Usage:
    python scripts/flight_endurance.py
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
PERF = SPECS["performance"]
WING = SPECS["wing"]
BATTERY = PROP["battery"]
MOTOR = PROP["motor"]

# ── Constants ────────────────────────────────────────────────────────────────
RHO = 1.225        # kg/m³
G = 9.81            # m/s²
W = PERF["auw_g"] / 1000 * G
S = WING["area_m2"]
AR = WING["aspect_ratio"]
E_OSWALD = 0.75
CL_MAX = PERF["cl_max"]
CD_MIN = 0.014
CL_0 = 0.90

V_BAT = BATTERY["voltage_nominal_V"]
CAP_AH = BATTERY["capacity_mAh"] / 1000
USABLE_FRACTION = 0.80    # Don't discharge below 20% (LiPo safety)
USABLE_AH = CAP_AH * USABLE_FRACTION


def drag_at_speed(V):
    """Total drag at speed V in level flight."""
    cl = 2 * W / (RHO * V**2 * S)
    cl_mid = 0.5 * CL_0
    k = 0.0035
    cd_profile = CD_MIN + k * (cl - cl_mid)**2
    cd_induced = cl**2 / (np.pi * AR * E_OSWALD)
    cd = cd_profile + cd_induced
    return 0.5 * RHO * V**2 * S * cd


def power_at_speed(V):
    """Power required for level flight at speed V."""
    return drag_at_speed(V) * V


def current_at_speed(V, eta_total=0.55):
    """
    Estimated motor current at speed V.
    eta_total = overall propulsive efficiency (motor × prop × ESC)
    """
    P_mech = power_at_speed(V)
    P_elec = P_mech / eta_total
    return P_elec / V_BAT


def endurance_at_speed(V, eta=0.55):
    """Flight endurance at constant speed V (hours)."""
    I = current_at_speed(V, eta)
    return USABLE_AH / I  # hours


def range_at_speed(V, eta=0.55):
    """Range at constant speed V (km)."""
    t = endurance_at_speed(V, eta)  # hours
    return V * t * 3.6  # km


def plot_endurance_vs_speed():
    """Plot endurance and range vs airspeed."""
    V_stall = np.sqrt(2 * W / (RHO * S * CL_MAX))
    V = np.linspace(V_stall + 0.5, PERF["max_speed_ms"], 200)

    endurance = np.array([endurance_at_speed(v) * 60 for v in V])  # minutes
    range_km = np.array([range_at_speed(v) for v in V])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Endurance
    ax1.plot(V, endurance, 'b-', linewidth=2.5)
    idx_max_e = np.argmax(endurance)
    ax1.plot(V[idx_max_e], endurance[idx_max_e], 'ro', markersize=10, zorder=5,
             label=f"Max endurance: {endurance[idx_max_e]:.1f} min at {V[idx_max_e]:.1f} m/s")

    # Cruise band
    ax1.axvspan(PERF["cruise_speed_ms"][0], PERF["cruise_speed_ms"][1],
                alpha=0.1, color='green', label="Cruise speed range")

    ax1.set_xlabel("Airspeed (m/s)", fontsize=13)
    ax1.set_ylabel("Endurance (minutes)", fontsize=13)
    ax1.set_title("Flight Endurance vs Speed", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Range
    ax2.plot(V, range_km, 'r-', linewidth=2.5)
    idx_max_r = np.argmax(range_km)
    ax2.plot(V[idx_max_r], range_km[idx_max_r], 'go', markersize=10, zorder=5,
             label=f"Max range: {range_km[idx_max_r]:.1f} km at {V[idx_max_r]:.1f} m/s")
    ax2.axvspan(PERF["cruise_speed_ms"][0], PERF["cruise_speed_ms"][1],
                alpha=0.1, color='green', label="Cruise speed range")

    ax2.set_xlabel("Airspeed (m/s)", fontsize=13)
    ax2.set_ylabel("Range (km)", fontsize=13)
    ax2.set_title("Flight Range vs Speed", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("NOOBLERS UAV — Endurance & Range Analysis", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT / "endurance_range.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: endurance_range.png")
    plt.close(fig)


def plot_battery_discharge():
    """Plot battery capacity vs discharge rate (C-rate effects)."""
    c_rates = np.linspace(0.5, 8, 200)
    # Peukert's effect: effective capacity decreases at higher C-rates
    # Using simplified model: C_eff = C_nominal * (C_1 / C_actual)^(n-1)
    peukert_n = 1.05  # typical for LiPo
    cap_effective = CAP_AH * (1 / c_rates)**(peukert_n - 1)
    cap_effective = np.minimum(cap_effective, CAP_AH)  # can't exceed nominal

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(c_rates, cap_effective * 1000, 'b-', linewidth=2.5, label="Effective capacity")
    ax.axhline(BATTERY["capacity_mAh"], color='gray', linestyle='--',
               label=f"Nominal: {BATTERY['capacity_mAh']} mAh")
    ax.axhline(BATTERY["capacity_mAh"] * USABLE_FRACTION, color='orange', linestyle='--',
               label=f"Usable (80%): {BATTERY['capacity_mAh'] * USABLE_FRACTION:.0f} mAh")

    # Mark typical cruise C-rate
    cruise_I = np.mean(PROP["cruise_current_A"])
    cruise_c = cruise_I / CAP_AH
    ax.axvline(cruise_c, color='green', linestyle=':', linewidth=1.5,
               label=f"Cruise C-rate: {cruise_c:.1f}C ({cruise_I:.0f} A)")

    ax.set_xlabel("Discharge Rate (C)", fontsize=13)
    ax.set_ylabel("Effective Capacity (mAh)", fontsize=13)
    ax.set_title("Battery Capacity vs Discharge Rate", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "battery_discharge.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: battery_discharge.png")
    plt.close(fig)


def simulate_mission():
    """Simulate a complete mission profile and plot results."""
    # Mission phases: Takeoff → Climb → Cruise → Payload Drop → Cruise → Descend → Land
    mission_phases = [
        {"name": "Takeoff Roll",  "duration_s": 5,   "throttle_pct": 100, "speed_ms": 8,  "altitude_rate": 0},
        {"name": "Climb",         "duration_s": 30,  "throttle_pct": 85,  "speed_ms": 10, "altitude_rate": 2.0},
        {"name": "Cruise Out",    "duration_s": 180, "throttle_pct": 55,  "speed_ms": 11, "altitude_rate": 0},
        {"name": "Payload Drop",  "duration_s": 15,  "throttle_pct": 45,  "speed_ms": 9,  "altitude_rate": -0.5},
        {"name": "Cruise Back",   "duration_s": 180, "throttle_pct": 55,  "speed_ms": 11, "altitude_rate": 0},
        {"name": "Descent",       "duration_s": 40,  "throttle_pct": 25,  "speed_ms": 10, "altitude_rate": -1.5},
        {"name": "Landing",       "duration_s": 10,  "throttle_pct": 15,  "speed_ms": 8,  "altitude_rate": -1.0},
    ]

    dt = 1  # 1 second timestep
    times = []
    altitudes = []
    speeds = []
    currents = []
    battery_pct = []
    phase_labels = []

    t = 0
    alt = 0
    capacity_used_Ah = 0

    for phase in mission_phases:
        for s in range(phase["duration_s"]):
            # Current draw model
            I_noload = 0.8
            I_max = 22.0
            I = I_noload + (I_max - I_noload) * (phase["throttle_pct"] / 100)**2

            capacity_used_Ah += I * dt / 3600
            remaining_pct = max(0, (1 - capacity_used_Ah / CAP_AH) * 100)

            alt += phase["altitude_rate"] * dt
            alt = max(0, alt)

            times.append(t)
            altitudes.append(alt)
            speeds.append(phase["speed_ms"])
            currents.append(I)
            battery_pct.append(remaining_pct)
            phase_labels.append(phase["name"])
            t += dt

    times = np.array(times) / 60  # convert to minutes

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # Altitude
    axes[0].plot(times, altitudes, 'b-', linewidth=2)
    axes[0].fill_between(times, 0, altitudes, alpha=0.1, color='blue')
    axes[0].set_ylabel("Altitude (m)", fontsize=12)
    axes[0].set_title("Mission Profile Simulation", fontsize=16, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Speed
    axes[1].plot(times, speeds, 'g-', linewidth=2)
    axes[1].set_ylabel("Airspeed (m/s)", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Current
    axes[2].plot(times, currents, 'r-', linewidth=2)
    axes[2].axhline(PROP["esc"]["rating_A"], color='orange', linestyle='--',
                    label=f"ESC limit: {PROP['esc']['rating_A']}A")
    axes[2].set_ylabel("Current (A)", fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    # Battery
    axes[3].plot(times, battery_pct, 'purple', linewidth=2)
    axes[3].axhline(20, color='red', linestyle='--', linewidth=1.5, label="Min safe (20%)")
    axes[3].fill_between(times, 0, 20, alpha=0.1, color='red')
    axes[3].set_ylabel("Battery (%)", fontsize=12)
    axes[3].set_xlabel("Time (minutes)", fontsize=12)
    axes[3].set_ylim(0, 105)
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.3)

    # Add phase annotations to top plot
    prev_phase = ""
    for i, label in enumerate(phase_labels):
        if label != prev_phase:
            axes[0].axvline(times[i], color='gray', linestyle=':', alpha=0.5)
            axes[0].text(times[i] + 0.05, max(altitudes) * 0.9, label,
                         fontsize=8, rotation=45, alpha=0.7)
            prev_phase = label

    fig.tight_layout()
    fig.savefig(OUTPUT / "mission_profile.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: mission_profile.png")
    plt.close(fig)

    return {
        "total_time_min": times[-1],
        "battery_remaining_pct": battery_pct[-1],
        "capacity_used_mAh": capacity_used_Ah * 1000,
        "max_altitude_m": max(altitudes),
    }


def plot_throttle_endurance():
    """Plot endurance at different throttle settings."""
    throttle = np.linspace(20, 100, 200)
    I_noload = 0.8
    I_max = 22.0
    current = I_noload + (I_max - I_noload) * (throttle / 100)**2
    endurance = USABLE_AH / current * 60  # minutes

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(throttle, endurance, 'b-', linewidth=2.5)
    ax.plot(55, USABLE_AH / (I_noload + (I_max - I_noload) * 0.55**2) * 60,
            'ro', markersize=10, label="Cruise (~55%)")
    ax.set_xlabel("Throttle (%)", fontsize=13)
    ax.set_ylabel("Endurance (minutes)", fontsize=13)
    ax.set_title("Flight Endurance vs Throttle Setting", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "throttle_endurance.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: throttle_endurance.png")
    plt.close(fig)


def print_summary():
    """Print endurance summary."""
    V_cruise = np.mean(PERF["cruise_speed_ms"])
    e_cruise = endurance_at_speed(V_cruise) * 60
    r_cruise = range_at_speed(V_cruise)

    V_stall = np.sqrt(2 * W / (RHO * S * CL_MAX))
    V = np.linspace(V_stall + 0.5, PERF["max_speed_ms"], 200)
    e_max = max([endurance_at_speed(v) * 60 for v in V])
    r_max = max([range_at_speed(v) for v in V])

    rows = [
        ["Battery capacity", f"{BATTERY['capacity_mAh']} mAh ({BATTERY['cells']}S)"],
        ["Usable capacity (80%)", f"{USABLE_AH*1000:.0f} mAh"],
        ["Cruise speed", f"{V_cruise:.0f} m/s"],
        ["Endurance at cruise", f"{e_cruise:.1f} min"],
        ["Range at cruise", f"{r_cruise:.1f} km"],
        ["Max endurance (best speed)", f"{e_max:.1f} min"],
        ["Max range (best speed)", f"{r_max:.1f} km"],
        ["Reserve target", "20% capacity"],
    ]

    print("\n" + "=" * 55)
    print("  FLIGHT ENDURANCE SUMMARY")
    print("=" * 55)
    print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
    print()


def main():
    print("\n" + "━" * 60)
    print("  🔋 FLIGHT ENDURANCE — NOOBLERS UAV")
    print("━" * 60)

    print("\n📊 Generating endurance analysis...")
    plot_endurance_vs_speed()
    plot_battery_discharge()
    plot_throttle_endurance()

    print("\n🎯 Simulating mission profile...")
    result = simulate_mission()
    print(f"  Mission time: {result['total_time_min']:.1f} min")
    print(f"  Battery remaining: {result['battery_remaining_pct']:.1f}%")
    print(f"  Max altitude: {result['max_altitude_m']:.0f} m")

    print_summary()

    print("All outputs saved to:", OUTPUT)
    print()


if __name__ == "__main__":
    main()
