#!/usr/bin/env python3
"""
Power analysis script to compute the minimum number of samples required
to detect a 10ms increase in average latency between stable and canary groups.
"""

from statsmodels.stats.power import TTestIndPower

def calculate_sample_size():
    """
    Calculate the minimum sample size needed per group to detect a 10ms increase in latency.
    
    Assumptions:
    - Standard deviation: 15ms
    - Power: 0.8
    - Significance level (alpha): 0.05
    
    Returns:
        int: The minimum sample size required per group
    """
    # Initialize the power analysis
    power_analysis = TTestIndPower()
    
    # Parameters
    effect_size = 10 / 15  # Cohen's d = mean difference / standard deviation
    alpha = 0.05           # Significance level
    power = 0.8            # Power
    
    # Calculate the required sample size
    sample_size = power_analysis.solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        ratio=1.0,         # Equal sample sizes in both groups
        alternative='two-sided'
    )
    
    # Return the sample size as an integer (rounded up)
    return int(sample_size + 0.5)  # Round up to ensure sufficient power

if __name__ == "__main__":
    sample_size = calculate_sample_size()
    print(f"Minimum sample size required per group: {sample_size}")