import numpy as np

"""
DATA GENERATING PROCESSES
"""

# market variables: us_eq, dk_eq, usddkk_spot, dk_1y, us_1y, dk_9y, us_9y, dk_10y, us_10y
initial_market = np.array([1.0, 1.0, 6.4, 0.015, 0.038, 0.026, 0.043, 0.026, 0.043])
initial_risk_drivers = initial_market.copy()
initial_risk_drivers[0:3] = np.log(initial_risk_drivers[0:3])  # log transform equity and fx
initial_risk_drivers[3:] = np.log(1 + initial_risk_drivers[3:]) # convert rates to log rates

# risk driver DGP
mean = np.array([
    0.070, # us equities ~3.5% ERP
    0.045,  # dk equities ~3% ERP
    initial_risk_drivers[2] - initial_risk_drivers[3], # fx ~UIP by default
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00 # yields (dk_1y, us_1y, dk_9y, us_9y, dk_10y, us_10y) ~zero change
])

vol = np.array([
    0.15,   # us_eq
    0.16,  # dk_eq.
    0.095,   # usddkk_spot
    0.0075,  # dk_1y
    0.0050,  # us_1y
    0.0080,  # dk_9y
    0.0080,  # us_9y
    0.0085,  # dk_10y
    0.0085   # us_10y
])

corr = np.array([
    [ 1. ,  0.6, -0.3,  0.1,  0.2,  0.1,  0.2,  0.1,  0.2],
    [ 0.6,  1. , -0. ,  0.1,  0.2,  0.1,  0.2,  0.1,  0.2],
    [-0.3, -0. ,  1. ,  0. ,  0.1, -0. ,  0.2, -0. ,  0.2],
    [ 0.1,  0.1,  0. ,  1. ,  0.2,  0.5,  0.4,  0.5,  0.4],
    [ 0.2,  0.2,  0.1,  0.2,  1. ,  0.2,  0.4,  0.2,  0.4],
    [ 0.1,  0.1, -0. ,  0.5,  0.2,  1. ,  0.7,  0.99 ,  0.7],
    [ 0.2,  0.2,  0.2,  0.4,  0.4,  0.7,  1. ,  0.7,  0.99],
    [ 0.1,  0.1, -0. ,  0.5,  0.2,  0.99 ,  0.7,  1. ,  0.7],
    [ 0.2,  0.2,  0.2,  0.4,  0.4,  0.7,  0.99 ,  0.7,  1. ]]
)

param_grid = dict()
param_grid['hedged_fx_return'] = [-0.05, 0.00, 0.05]
param_grid['equity_fx_corr'] = [-0.5, 0.00, 0.5]
param_grid['rates_fx_corr'] = [0.20, 0.20, 0.20]
base_case_ix = 1 # index of the base case in the parameter grid

"""
INDICES
"""
us_eq_rd_ix = 0
dk_eq_rd_ix = 1
usddkk_rd_ix = 2
dk_1y_rd_ix = 3
us_1y_rd_ix = 4
dk_9y_rd_ix = 5
us_9y_rd_ix = 6
dk_10y_rd_ix = 7
us_10y_rd_ix = 8

n_assets = 5 # (us eq, dk_eq, USDDKK fx hedge, dk_10y, us 10y)

us_eq_asset_ix = 0
dk_eq_asset_ix = 1
fx_hedge_asset_ix = 2
dk_10y_asset_ix = 3
us_10y_asset_ix = 4

"""
PLOTS
"""
cash_asset_labels = ['Aktier (US)', 'Aktier (DK)', 'Obligationer (DK)', 'Obligationer (US)']
cash_asset_colors = ['#7A1021', '#D94A5A', '#333399', '#000033']
fx_hedge_label = 'Valutaafdækning'
fx_hedge_color = '#F7B402'

methods = ['Global', 'Pre-hedge', 'Post-hedge']
method_colors = {'Global': '#004132', 'Pre-hedge': '#50A1FB', 'Post-hedge': '#E754B6'}
method_linestyles = {'Global': '-', 'Pre-hedge': 'dashed', 'Post-hedge': 'dashdot'}
method_names = {'Global': 'Global', 'Pre-hedge': 'Præ-afdækning', 'Post-hedge': 'Post-afdækning'}
method_zorder = {'Global': 1, 'Pre-hedge': 2, 'Post-hedge': 3}
method_alpha = {'Global': 1.0, 'Pre-hedge': 1.0, 'Post-hedge': 1.0}
method_linewidth = {'Global': 3, 'Pre-hedge': 2, 'Post-hedge': 2}

facecolor = '#E6E6E6'

"""
SIMULATION PARAMETERS
"""
n_scenarios = 10_000  # number of scenarios to generate
n_periods = 1  # number of periods in the simulation (e.g., years)

"""
ESTIMATION UNCERTAINTY
"""
n_obs = 100 # number of historical observations used to estimate the mean vector and covariance matrix

"""
OPTIMIZATION PARAMETERS
"""
budget = 1.0
min_hedge_ratio = 0.0
max_hedge_ratio = 1.0
n_points = 100  # number of points in the efficient frontier
risk_aversion = 5 # risk aversion coefficient for the utility function

if __name__ == '__main__':
    # print the configuration to verify
    print("Configuration module loaded successfully.")
    print(f"Initial market values: {initial_market}")
    print(f"Number of scenarios: {n_scenarios}")
    print(f"Number of periods: {n_periods}")
    print(f"Budget: {budget}")
    print(f"Minimum hedge ratio: {min_hedge_ratio}")
    print(f"Maximum hedge ratio: {max_hedge_ratio}")
    print(f"Number of points in the efficient frontier: {n_points}")
    # check that covariance matrices are positive semi definite
    for equity_fx_corr, rates_fx_corr in zip(param_grid['equity_fx_corr'],
                                             param_grid['rates_fx_corr']):

        corr[us_eq_rd_ix, usddkk_rd_ix] = equity_fx_corr  # update the correlation matrix with the current equity-fx correlation
        corr[usddkk_rd_ix, us_eq_rd_ix] = equity_fx_corr  # ensure symmetry
        corr[us_10y_rd_ix, usddkk_rd_ix] = rates_fx_corr # update the correlation matrix with the current rates-fx correlation
        corr[usddkk_rd_ix, us_10y_rd_ix] = rates_fx_corr # ensure symmetry
        corr[us_9y_rd_ix, usddkk_rd_ix] = rates_fx_corr # update the correlation matrix with the current rates-fx correlation
        corr[usddkk_rd_ix, us_9y_rd_ix] = rates_fx_corr # ensure symmetry
        cov_matrix = np.diag(vol) @ corr @ np.diag(vol)
        if np.all(np.linalg.eigvals(cov_matrix) >= 0):
            print(f"Covariance matrix for scenario equity_fx_corr: {equity_fx_corr} is positive semi-definite.")
        else:
            print(f"Covariance matrix for scenario equity_fx_corr: {equity_fx_corr} is NOT positive semi-definite.")