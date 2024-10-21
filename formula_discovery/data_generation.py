import random
from typing import List, Dict, Any, Optional 
import sympy as sp 

def generate_data(num_problems: int, var_range=(1, 1000)) -> List[tuple]:
    """Generate training data with random variable combinations."""
    data = []
    for _ in range(num_problems):
        # Select formula and known variables randomly
        formula = random.choice(AP_FORMULAS)
        known_vars = random.sample(list(formula.free_symbols), k=random.randint(2, len(formula.free_symbols) - 1))
        known_values = {var: random.randint(*var_range) for var in known_vars}
        
        # Find the target variable (one not in knowns)
        target_vars = list(set(formula.free_symbols) - set(known_vars))
        if not target_vars:
            continue  # Skip if no target variable
        target_var = target_vars[0]
        target_value = solve_for_unknown(known_values, target_var)
        if target_value is None:
            continue  # Skip if unable to solve numerically
        
        # Include midterm and sum in the known values if applicable
        if ak in formula.free_symbols and ak not in known_values:
            ak_value = solve_for_unknown(known_values, ak)
            if ak_value is not None:
                known_values[ak] = ak_value
            else:
                continue  # Skip if unable to solve
        
        if Sn in formula.free_symbols and Sn not in known_values and (a1 in known_values or an in known_values):
            Sn_value = solve_for_unknown(known_values, Sn)
            if Sn_value is not None:
                known_values[Sn] = Sn_value
            else:
                continue  # Skip if unable to solve
        
        data.append((known_values, target_var, target_value))
    return data
