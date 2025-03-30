# QQQ Monte Carlo Investment Simulator

A Monte Carlo simulator for projecting QQQ (Invesco NASDAQ-100 ETF) investment outcomes with customizable parameters and interactive visualizations.

## Features 
 **Real-World Use Case**: Simulate investment outcomes with historical returns (14.37%) and volatility (23.95%).  
 **Risk Visualization**: Plotly charts show 10th/50th/90th percentile projections.  
 **Interactive 3D visualization with Plotly
 ** Contribution baseline comparison

 ## Tech Stack   
- **Core**: Python (NumPy, Plotly)  
- **Math**: Geometric Brownian Motion, Monte Carlo Simulation  
- **Workflow**: CLI arguments, Object-Oriented Design  
 
## Quick Start
```bash
git clone https://github.com/justinchn359/QQQ-Monte-Carlo-Simulator.git
cd QQQ-Monte-Carlo-Simulator
pip install -r requirements.txt
python qqq_simulator.py --investment 10000 --contribution 20000 --days 5040 --sims 1000
