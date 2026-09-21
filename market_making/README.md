## Overview & Current Status

> **Note**: This module is currently under active development.

This project benchmarks classical quantitative market-making models against non-Poissonian order flow dynamics and explores optimal control via Deep Reinforcement Learning (DRL).

### 1. Current Implementation: Avellaneda & Stoikov (2008) Benchmark
* **Baseline:** Implementation of the classic framework from *High-Frequency Trading in a Limit Order Book* (Avellaneda & Stoikov, 2008), using their closed-form reservation price derived via Taylor approximation.
* **Stress-testing with Clustered Arrivals:** We substitute the original homogeneous Poisson process for order arrivals with a **univariate Hawkes process**.
* **Findings:** The analytical approximation degrades significantly under self-exciting dynamics, leading to severe inventory drift and negative expected utility.

### 2. Intermediate Milestone: Exact HJB Solution
* Benchmark the performance against the exact, closed-form solution of the Hamilton-Jacobi-Bellman (HJB) system proposed by **Guéant, Tapia, and Lehalle (2012)** (*Dealing with the inventory risk: a solution to the market making problem*).
* **Objective:** Disentangle the performance loss caused by Avellaneda-Stoikov's asymptotic Taylor approximation from the loss caused by the Poisson misspecification itself.

### 3. Target Framework: Deep RL under Bivariate Hawkes Dynamics
* **Environment:** Simulating high-frequency order flows using a **bivariate Hawkes process** with a sum-of-exponentials kernel (capturing cross-excitation between bid and ask arrivals).
* **Optimization:** Training a Deep Reinforcement Learning agent (continuous action space for optimal half-spreads $\delta^a, \delta^b$) to manage inventory risk and adverse selection without analytical tractability assumptions.
* **Evaluation:** Out-of-sample backtesting against the classical PDE/HJB baselines.