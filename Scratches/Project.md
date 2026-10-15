## Project Proposal: Electric Vehicle Eco-Driving with Reinforcement Learning

**Student:** Ronan Kenny (23404724)
**Supervisor:** Dr. Enda Howley
**Domain:** Deep Reinforcement Learning (DRL) & Autonomous Vehicles
**Primary Tool:** MetaDrive Simulator (PyTorch + Gymnasium API)

### Project details 
Python version - 3.10 


### Links and Resources 

https://karpathy.github.io/2016/05/31/rl/

https://www.youtube.com/watch?v=wDVteayWWvU

https://metadrive-simulator.readthedocs.io/en/latest/install.html
Metadrive documentation 



### Project Title Options

- **Option A:** _Teaching Electric Self-Driving Cars to Save Battery: An Eco-Driving Reinforcement Learning Study_

- **Option B:** _Multi-Objective Reinforcement Learning for Energy-Efficient Autonomous Electric Vehicles_
### What Is This Project About? (Executive Summary)

Standard self-driving AI models focus almost entirely on speed and path completion. This leads to **aggressive driving patterns** such as sudden acceleration spikes and hard braking that can quickly drain an Electric Vehicle's (EV) battery.

This project uses the **MetaDrive** simulator to train a smart self-driving car that learns to **balance travel time against energy usage**. By giving the AI direct awareness of its motor power draw and penalizing jerky movements, the AI learns smooth, fuel-efficient driving habits without sacrificing safety or travel speed.

### Core Research Themes

```
                             ┌──────────────────────────────────┐
                             │        MetaDrive Simulator       │
                             └────────────────┬─────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
          [ Theme 1: Eco-Driving ]                        [ Theme 2: Smoothness ]
     Balance battery usage vs. speed                   Eliminate sudden acceleration spikes
     ├─ Motor power calculation                        ├─ Jerk reduction (smoother turns/brakes)
     └─ Time-to-destination trade-off                  └─ Comparison against aggressive baseline
```

#### Theme 1: Battery & Energy Efficiency

- **The Problem:** EVs have a limited driving range. Aggressive driving drains batteries significantly faster than smooth driving.

- **The Solution:** Give the AI car access to its live battery status (motor torque and current speed) so it can see how much power its choices use.

- **The Experiment:** Train two different AI drivers in MetaDrive and compare them:
  
    1. **Aggressive Driver (Baseline):** Earns points purely for driving fast and reaching the destination.

    2. **Eco-Driver (Experimental):** Earns points for reaching the destination, but loses points whenever it wastes energy or slams on the accelerator/brakes.

#### Theme 2: Driving Smoothness & Passenger Comfort

- **The Problem:** High battery consumption is directly caused by sudden changes in speed (called **jerk**).
- **The Solution:** Mathematically penalize harsh movements in the AI's reward system.
- **The Goal:** Prove that the Eco-Driver drives significantly smoother than the standard driver, leading to longer battery life and a more comfortable ride.
#### Semester 1: Foundations & System Design

- **Weeks 1–4 (Setup & Familiarization):**
  
    - Set up the Python environment with PyTorch, Gymnasium, `Stable-Baselines3`, and MetaDrive.

    - Run baseline driving scripts to ensure GPU rendering works properly.

- **Weeks 5–8 (Energy Wrapper & Reward Engineering):**
  
    - Write the custom Python wrapper that calculates motor power draw ($P = \text{Force} \times \text{Speed}$) and driving jerk.

    - Define the **Eco-Reward system** that penalizes energy waste.

- **Weeks 9–12 (Initial Testing & Mid-Term Review):**

-  Run preliminary training experiments using standard algorithms (PPO / SAC).

 -  Submit the Mid-Project Progress Report showing initial driving performance curves.

#### Semester 2: Experiments, Analysis & Final Dissertation

- **Weeks 13–16 (Main Experiments):**
  
    - Train the **Aggressive Driver** and **Eco-Driver** over 500,000 simulation steps across various road maps.
 
- **Weeks 17–20 (Results & Data Visualization):**
  
    - Calculate exact battery savings per kilometer ($\text{kWh}/\text{km}$) and driving smoothness (jerk).

    - Generate comparative line graphs and scatter plots comparing time vs. energy saved.

- **Weeks 21–24 (Final Report & Project Video):**
  
    - Finalize the FYP dissertation text and polish the codebase.

    - Record demonstration videos showing side-by-side driving behaviors.
 
### Key Performance Metrics

| **What We Are Measuring**                | **Plain English Explanation**                                               | **Target Goal**                                                |
| ---------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Energy Used ($\text{kWh}/\text{km}$)** | How much battery power the car consumes per kilometer driven.               | **20%–30% reduction** in the Eco-Driver vs. Aggressive Driver. |
| **Driving Jerk ($\text{m/s}^3$)**        | How often the car accelerates or brakes abruptly (smoothness).              | **$>40\%$ reduction** in jerk for the Eco-Driver.              |
| **Travel Time Variance**                 | How much extra time the eco-friendly route takes compared to fast driving.  | Keep travel time within **10%** of the fast driver.            |
| **Success Rate (%)**                     | Percentage of driving trips completed without crashing or leaving the lane. | **$>95\%$ success rate** for both models.                      |
