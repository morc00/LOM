<h1 align="center">✈️ Laws of Motion — Electric Fixed-Wing UAV</h1>

<p align="center">
  <b>Team NOOBLERS</b> · Laws of Motion, Kshitij 2025 · IIT Kharagpur
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-competition--ready-brightgreen?style=flat-square" alt="Status"/>
  <img src="https://img.shields.io/badge/airfoil-Selig%20S1223-blue?style=flat-square" alt="Airfoil"/>
  <img src="https://img.shields.io/badge/AUW-790g-orange?style=flat-square" alt="Weight"/>
  <img src="https://img.shields.io/badge/T%2FW-1.10%3A1-red?style=flat-square" alt="T/W Ratio"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square" alt="License"/>
</p>

---

## 📋 Overview

This repository contains the complete technical design, aerodynamic analysis tools, and engineering documentation for **Team NOOBLERS'** electric fixed-wing UAV entry for the **Laws of Motion** aerial robotics competition at **Kshitij 2025, IIT Kharagpur**.

The aircraft is a lightweight, foam-composite, fixed-wing platform engineered for stable low-speed flight, reliable payload delivery, and ease of on-field maintenance.

### 🎯 Design Philosophy

| Principle | Implementation |
|-----------|---------------|
| **Aerodynamic Stability** | Selig S1223 high-lift airfoil with gradual stall onset |
| **Structural Simplicity** | Foam-composite construction for rapid prototyping & field repair |
| **System Reliability** | Proven COTS propulsion & modern ELRS digital radio link |

---

## 🛩️ Aircraft Specifications

| Parameter | Value | Unit |
|-----------|-------|------|
| All-Up Weight (AUW) | **790** | g |
| Maximum Static Thrust | **870** | g |
| Thrust-to-Weight Ratio | **1.10** | — |
| Wing Area | **0.313** | m² |
| Wing Span | **1160** | mm |
| Chord | **270** | mm |
| Aspect Ratio | **4.3** | — |
| Wing Loading | **24.8** | N/m² |
| Stall Speed | **7–8** | m/s |
| Cruise Speed | **10–12** | m/s |
| Max Speed | **~18** | m/s |
| Endurance | **12–16** | min |
| Lift-to-Drag Ratio | **8–10** | — |
| RC Frequency | **2.4** | GHz |
| RC Protocol | **ELRS** | — |

---

## 🔧 Systems Overview

### Airfoil — Selig S1223
Selected after comparative analysis of 5 low-Reynolds-number profiles. The S1223 delivers:
- **CL_max ≈ 2.2** — substantially higher than symmetric or lightly cambered profiles
- Gradual stall onset for pilot recovery margin
- Well-documented performance at Re = 50,000–300,000

### Propulsion
| Component | Specification |
|-----------|--------------|
| Motor | 2212 / 1400 KV |
| Battery | 3S 2200 mAh Li-Po (11.1V) |
| ESC | 30A (with 5V/3A BEC) |
| Propeller | 10×4.5 twin-blade |

### Avionics
- **Transmitter:** Radiomaster Pocket
- **Receiver:** ExpressLRS (ELRS) PWM — sub-5ms latency, 120dB link budget
- **Servos:** 5× 9g micro servos (ailerons, elevator, rudder, payload door)

### Structure
- **Wing:** EPS styrofoam, hot-wire cut, carbon-fibre spar at 25% chord
- **Fuselage:** Hybrid corosheet + styrofoam (850mm)
- **Tail:** Conventional cruciform, 520mm moment arm

---

## 📊 Analysis Scripts

A suite of Python tools for aerodynamic design, performance prediction, and system optimisation. All scripts read from a central `data/specs.json` configuration and save plots to `output/`.

### Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/LOM.git
cd LOM

# Install dependencies
pip install -r requirements.txt

# Run any analysis script
python scripts/airfoil_selector.py
python scripts/lift_drag_calculator.py
python scripts/performance_envelope.py
python scripts/propulsion_analyzer.py
python scripts/weight_balance.py
python scripts/wing_loading_calculator.py
python scripts/flight_endurance.py
python scripts/stability_analysis.py
```

### Script Descriptions

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `airfoil_selector.py` | Compares 5 candidate airfoils (S1223, Clark Y, NACA 2412, E423, MH114) with weighted scoring | Geometry overlay, CL vs α, drag polars, L/D comparison, ranked table |
| `lift_drag_calculator.py` | Analyses CL, CD, and L/D for the S1223 across angle-of-attack and Reynolds number | Lift curve, drag polar, L/D efficiency, Re sensitivity |
| `performance_envelope.py` | Computes flight performance limits | V-n diagram, power curves, rate of climb, stall vs bank angle |
| `propulsion_analyzer.py` | Analyses motor-prop-battery system | Thrust vs speed, current draw, efficiency map, prop tip Mach check |
| `weight_balance.py` | CG computation and weight management | Weight pie/bar charts, CG diagram, battery position sensitivity |
| `wing_loading_calculator.py` | Wing design trade studies | Loading comparison, lift distribution, stall sensitivity, AR trade |
| `flight_endurance.py` | Battery endurance and mission simulation | Endurance/range vs speed, battery discharge, full mission profile |
| `stability_analysis.py` | Longitudinal stability assessment | Static margin, trim diagram, tail volume sizing comparison |

---

## 📁 Project Structure

```
LOM/
├── README.md                       # This file
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── scripts/
│   ├── airfoil_selector.py         # Airfoil comparison & selection
│   ├── lift_drag_calculator.py     # CL/CD/L/D analysis
│   ├── performance_envelope.py     # Flight performance limits
│   ├── propulsion_analyzer.py      # Propulsion system analysis
│   ├── weight_balance.py           # Weight & CG calculator
│   ├── wing_loading_calculator.py  # Wing loading & design trades
│   ├── flight_endurance.py         # Endurance & mission simulation
│   └── stability_analysis.py       # Longitudinal stability
├── data/
│   ├── specs.json                  # Aircraft specifications (single source of truth)
│   └── airfoils/                   # Airfoil coordinate files
│       ├── s1223.dat
│       ├── clark_y.dat
│       ├── naca2412.dat
│       ├── e423.dat
│       └── mh114.dat
├── output/                         # Generated plots (gitignored)
```

---

## ⚖️ Weight Breakdown

| Component | Mass (g) | % AUW |
|-----------|----------|-------|
| Battery (3S 2200 mAh) | 185 | 23.4% |
| Wing (foam + spar) | 160 | 20.3% |
| Fuselage | 120 | 15.2% |
| Payload | 79 | 10.0% |
| Motor (2212) | 55 | 7.0% |
| Servos (5×9g) | 45 | 5.7% |
| Tail Assembly | 45 | 5.7% |
| Propeller & Mount | 35 | 4.4% |
| Wiring & Hardware | 30 | 3.8% |
| ESC (30A) | 28 | 3.5% |
| Receiver (ELRS) | 8 | 1.0% |
| **Total AUW** | **790** | **100%** |

---

## 🧪 Testing Protocol

1. **Structural load test** — Wing loaded to 3g (2.4 kg) to confirm elastic limits
2. **Control linkage check** — Zero play and correct direction verification
3. **Motor burn-in** — 2-minute full-throttle with temperature monitoring
4. **Range check** — ELRS link verified at 200m+ with motor running
5. **CG verification** — Balanced at 33% chord before every flight
6. **Maiden flight** — Low-altitude straight-and-level pass, then gradual envelope expansion

---

## 👥 Team

| Member | Role |
|--------|------|
| **Soumyadip Das** | Team Leader |
| Ankan Prasad Roy | Team Member |
| Aparna Dutta | Team Member |
| Souptik De | Team Member |

📧 Contact: soumyadipdas1710@gmail.com

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Submitted for Laws of Motion · Kshitij 2025 · IIT Kharagpur</i>
</p>
