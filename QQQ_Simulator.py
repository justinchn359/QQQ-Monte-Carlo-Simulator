import numpy as np
import plotly.graph_objects as go
import argparse
import os
import webbrowser
from typing import Optional

class QQQSimulator:
    """
    Simulates portfolio growth for QQQ (NASDAQ-100 ETF) using Monte Carlo methods.
    
    Attributes:
        daily_return (float): Daily return rate (annual return / 252 trading days).
        daily_volatility (float): Daily volatility (annual volatility / sqrt(252)).
        initial_investment (float): Starting portfolio value.
        annual_contribution (float): Yearly contribution amount.
        contribution_interval (int): Days between contributions (default: 21 days ≈ monthly).
    """
    
    def __init__(
        self, 
        annual_return: float, 
        annual_volatility: float,
        initial_investment: float = 10000,
        annual_contribution: float = 20000,
        contribution_interval: int = 21
    ):
        assert initial_investment >= 0, "Initial investment cannot be negative"
        assert annual_contribution >= 0, "Annual contribution cannot be negative"
        assert contribution_interval > 0, "Contribution interval must be positive"

        self.daily_return = annual_return / 252
        self.daily_volatility = annual_volatility / np.sqrt(252)
        self.initial_investment = initial_investment
        self.annual_contribution = annual_contribution
        self.monthly_contribution = annual_contribution / 12
        self.contribution_interval = contribution_interval

    def monte_carlo_simulation(self, days: int = 5040, simulations: int = 1000) -> np.ndarray:
        """
        Run Monte Carlo simulation for portfolio growth.
        
        Args:
            days (int): Total simulation days (default: 5040 ≈ 20 years).
            simulations (int): Number of scenarios to generate.
        
        Returns:
            np.ndarray: (days x simulations) array of portfolio values.
        """
        portfolio = np.zeros((days, simulations), dtype=np.float64)
        current = np.full(simulations, self.initial_investment, dtype=np.float64)
        contrib_interval = self.contribution_interval
        
        for day in range(days):
            if day != 0 and day % contrib_interval == 0:
                current += self.monthly_contribution
                
            log_returns = np.random.normal(
                loc=self.daily_return - 0.5 * self.daily_volatility**2,
                scale=self.daily_volatility,
                size=simulations
            )
            current *= np.exp(log_returns)
            portfolio[day] = current

        return portfolio

    def visualize(self, results: np.ndarray) -> go.Figure:
        """Generate interactive Plotly visualization of results."""
        fig = go.Figure()
        percentiles = [10, 50, 90]
        colors = ['#FF4444', '#339999', '#3366CC']
        days = results.shape[0]
        years = days / 252
        x_axis = np.linspace(0, years, days)

        # Contributions baseline
        contrib_interval = self.contribution_interval
        baseline_contrib = np.zeros(days)
        baseline_contrib[0] = self.initial_investment
        baseline_contrib[contrib_interval::contrib_interval] = self.monthly_contribution
        baseline = np.cumsum(baseline_contrib)
        total_contrib = self.initial_investment + self.annual_contribution * (years)

        # Percentile traces
        for p in percentiles:
            final_val = np.percentile(results[-1], p)
            cagr = ((final_val / total_contrib) ** (1/years) - 1) * 100
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=np.percentile(results, p, axis=1),
                mode='lines',
                name=f'{p}th Percentile (End: ${final_val:,.0f}, CAGR: {cagr:.1f}%)',
                line=dict(color=colors[percentiles.index(p)], width=3)
            ))

        # Sample paths
        sample_size = min(50, results.shape[1])
        for idx in np.random.choice(results.shape[1], sample_size, replace=False):
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=results[:, idx],
                mode='lines',
                opacity=0.1,
                showlegend=False,
                line=dict(color='#666666', width=1.5)
            ))

        # Baseline
        final_baseline = baseline[-1]
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=baseline,
            mode='lines',
            name=f'Contributions Only (End: ${final_baseline:,.0f}, CAGR: 0.0%)',
            line=dict(color='black', dash="3 3", width=2)
        ))

        fig.update_layout(
            title='<b>QQQ Monte Carlo Simulation (20-Year Horizon)</b>',
            xaxis_title='Years',
            yaxis_title='Portfolio Value ($)',
            template='plotly_white',
            legend=dict(
                y=1.00,
                x=0.017,
                bgcolor='white',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=10)
            )
        )

        return fig

def main():
    annual_return = 0.1437  # 14.37% historical annual return for QQQ
    annual_vol = 0.2395     # 23.95% historical annual volatility

    parser = argparse.ArgumentParser(description="QQQ Monte Carlo Simulator")
    parser.add_argument('--investment', type=float, default=10000, help="Initial investment amount")
    parser.add_argument('--contribution', type=float, default=20000, help="Annual contribution amount")
    parser.add_argument('--days', type=int, default=5040, help="Simulation duration in days (default: 20 years)")
    parser.add_argument('--sims', type=int, default=1000, help="Number of Monte Carlo simulations")
    parser.add_argument('--interval', type=int, default=21, help="Contribution interval in days (default: 21 ≈ monthly)")
    args = parser.parse_args()

    simulator = QQQSimulator(
        annual_return=annual_return,
        annual_volatility=annual_vol,
        initial_investment=args.investment,
        annual_contribution=args.contribution,
        contribution_interval=args.interval
    )

    results = simulator.monte_carlo_simulation(args.days, args.sims)
    fig = simulator.visualize(results)
    
    filename = "qqq_simulation_20y.html"
    fig.write_html(filename)
    webbrowser.open(f"file://{os.path.abspath(filename)}")

if __name__ == "__main__":
    main()
