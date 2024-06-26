# Risk-Sensitive Stochastic Optimal Control as Rao-Blackwellized Markovian Score Climbing

Implements a policy optimization technique via Markovian score climbing

## Installation
 
 Create a conda environment
    
    conda create -n NAME python=3.10
    
 Then head to the cloned repository and execute
 
    pip install -e .
    
 ## Examples
 
 A policy learning example on a simple pendulum environment
 
    python examples/feedback/rb_csmc_pendulum.py
    
 Examples for ILEQG and RAT-ILQR on a simple linear dynamics environment
   
   ```
   python examples/rat_ilqr/ileqg_const_linear.py
   python examples/rat_ilqr/rat_ilqr_const_linear.py
   ```