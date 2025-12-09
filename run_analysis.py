import numpy as np

import config
import plotting
import optimization

from scipy.stats import multivariate_normal, wishart

def main():

    """
    Main function to run the portfolio construction analysis.
    """

    # creating linear transformation matrix to compute initial prices
    n_risk_drivers = config.initial_risk_drivers.shape[0]
    n_assets = config.n_assets
    a = np.zeros((n_assets, n_risk_drivers))

    # us_eq
    a[config.us_eq_asset_ix, config.us_eq_rd_ix] = 1.0  # equity risk driver in local currency (usd)
    a[config.us_eq_asset_ix, config.usddkk_rd_ix] = 1.0  # fx rate for conversion from usd to dkk

    # dk_eq
    a[config.dk_eq_asset_ix, config.dk_eq_rd_ix] = 1.0  # equity risk driver in local currency (dkk)

    # usdkk_forward
    a[config.fx_hedge_asset_ix, config.usddkk_rd_ix] = 1.0  # usddkk spot rate
    a[config.fx_hedge_asset_ix, config.dk_1y_rd_ix] = 1.0  # dkk 1y rat
    a[config.fx_hedge_asset_ix, config.us_1y_rd_ix] = -1.0  # usd 1y rate

    # dk_10y_bond
    a[config.dk_10y_asset_ix, config.dk_10y_rd_ix] = -1.0 * 10  # dk 10y rate

    # us_10y_bond
    a[config.us_10y_asset_ix, config.us_10y_rd_ix] = -1.0 * 10  # us 10y rate
    a[config.us_10y_asset_ix, config.usddkk_rd_ix] = 1.0 # fx rate for conversion from usd to dkk

    # calculating initial prices as exponential transformation of the risk drivers times the linear transformation matrix
    initial_prices = np.exp(a @ config.initial_risk_drivers)

    # defining the linear transformation matrix for the risk driver distribution at the horizon
    b = np.zeros((n_assets, n_risk_drivers))

    # us_eq
    b[config.us_eq_asset_ix, config.us_eq_rd_ix] = 1.0  # equity risk driver in local currency (usd)
    b[config.us_eq_asset_ix, config.usddkk_rd_ix] = 1.0  # fx rate for conversion from usd to dkk

    # dk_eq
    b[config.dk_eq_asset_ix, config.dk_eq_rd_ix] = 1.0  # equity risk driver in local currency (dkk)

    # usdkk_spot
    b[config.fx_hedge_asset_ix, config.usddkk_rd_ix] = 1.0  # usddkk spot rate

    # dk_10y_bond
    b[config.dk_10y_asset_ix, config.dk_9y_rd_ix] = -1.0 * 9  # dk 10y rate

    # us_9y_bond
    b[config.us_10y_asset_ix, config.us_9y_rd_ix] = -1.0 * 9  # us 9y rate
    b[config.us_10y_asset_ix, config.usddkk_rd_ix] = 1.0  # fx rate for conversion from usd to dkk

    # calculating pnl distributions at the horizon using base assumptions and parameter grid
    pnl_dist = dict()  # keys as (hedged_fx_return, equity_fx_corr) tuples
    invariant_dist = dict()  # keys as (hedged_fx_return, equity_fx_corr) tuples
    for hedged_fx_return in config.param_grid['hedged_fx_return']:

        for equity_fx_corr in config.param_grid['equity_fx_corr']:

            # getting base assumptions
            true_mean_invariant = config.mean.copy()
            vol_invariant = config.vol.copy()
            corr_invariant = config.corr.copy()

            # updating the mean of the FX rate invariant distribution
            true_mean_invariant[config.usddkk_rd_ix] = (

                    np.log(np.exp(config.initial_risk_drivers[config.dk_1y_rd_ix]
                                  - config.initial_risk_drivers[config.us_1y_rd_ix]) - hedged_fx_return)
                    - 0.5 * config.vol[config.usddkk_rd_ix] ** 2 # vol adjustment

            )

            # updating the correlation matrix based on the parameter grid
            corr_invariant[config.us_eq_rd_ix, config.usddkk_rd_ix] = equity_fx_corr
            corr_invariant[config.usddkk_rd_ix, config.us_eq_rd_ix] = equity_fx_corr

            # computing the covariance matrix for the updated DGP
            true_cov_invariant = np.diag(vol_invariant) @ corr_invariant @ np.diag(vol_invariant)

            # projecting the invariant distribution to the horizon
            true_mean_invariant = true_mean_invariant * config.n_periods
            true_cov_invariant = true_cov_invariant * config.n_periods

            # storing the invariant distribution
            invariant_dist[(hedged_fx_return, equity_fx_corr)] = (true_mean_invariant, true_cov_invariant)

            # computing the risk driver distribution at the horizon (cov matrix remains the same)
            mean_risk_driver = true_mean_invariant + config.initial_risk_drivers
            cov_risk_driver = true_cov_invariant.copy()

            # computing the price distribution at the horizon
            mean_param_lognormal = b @ mean_risk_driver
            cov_param_lognormal = b @ cov_risk_driver @ b.T

            mean_price = np.exp(mean_param_lognormal + 0.5 * np.diag(cov_param_lognormal))
            cov_price = np.outer(mean_price, mean_price) * (np.exp(cov_param_lognormal) - 1)

            # computing pnl distribution at the horizon
            mean_pnl = mean_price - initial_prices
            cov_pnl = cov_price.copy()

            # check that pnl covariance matrix is positive semi-definite
            assert np.all(np.linalg.eigvals(cov_pnl) >= -1e-10), "Covariance matrix is not positive semi-definite"

            # storing the pnl distribution
            pnl_dist[(hedged_fx_return, equity_fx_corr)] = (mean_pnl, cov_pnl)

            if (equity_fx_corr == config.param_grid['equity_fx_corr'][config.base_case_ix] and
                hedged_fx_return == config.param_grid['hedged_fx_return'][config.base_case_ix]):

                a_output = np.zeros((6, n_risk_drivers))
                a_output[0, config.us_eq_rd_ix] = 1.0  #  us equity risk driver in local currency (usd)
                a_output[1, config.dk_eq_rd_ix] = 1.0  # dk equity risk driver in local currency (dkk)
                a_output[2, config.us_10y_rd_ix] = -1.0 * 10  # us 10y rate
                a_output[3, config.dk_10y_rd_ix] = -1.0 * 10  # dk 10y rate
                a_output[4, config.usddkk_rd_ix] = 1.0  # usddkk spot rate
                a_output[5, config.usddkk_rd_ix] = 1.0  # usddkk spot rate
                a_output[5, config.dk_1y_rd_ix] = 1.0  # dkk 1y rate
                a_output[5, config.us_1y_rd_ix] = -1.0  # usd 1y rate

                initial_prices_output = np.exp(a_output @ config.initial_risk_drivers)

                b_output = np.zeros((6, n_risk_drivers))
                b_output[0, config.us_eq_rd_ix] = 1.0  # equity risk driver in local currency (usd)
                b_output[1, config.dk_eq_rd_ix] = 1.0  # equity risk driver in local currency (dkk)
                b_output[2, config.us_9y_rd_ix] = -1.0 * 9  # us 9y rate
                b_output[3, config.dk_9y_rd_ix] = -1.0 * 9  # dk 9y rate
                b_output[4, config.usddkk_rd_ix] = 1.0  # usddkk spot rate
                b_output[5, config.usddkk_rd_ix] = 1.0  # usddkk spot rate

                mean_param_lognormal_output = b_output @ mean_risk_driver
                cov_param_lognormal_output = b_output @ cov_risk_driver @ b_output.T

                mean_price_output = np.exp(mean_param_lognormal_output + 0.5 * np.diag(cov_param_lognormal_output))
                cov_price_output = np.outer(mean_price_output, mean_price_output) * (np.exp(cov_param_lognormal_output) - 1)

                mean_pnl_output = mean_price_output - initial_prices_output
                mean_pnl_output[5] *= -1 # pnl of selling usd outright
                cov_pnl_output = cov_price_output.copy()
                cov_pnl_output[5, 4] *= -1
                cov_pnl_output[4, 5] *= -1
                cov_pnl_output[5, 0] *= -1
                cov_pnl_output[0, 5] *= -1
                cov_pnl_output[5, 1] *= -1
                cov_pnl_output[1, 5] *= -1
                cov_pnl_output[5, 2] *= -1
                cov_pnl_output[2, 5] *= -1
                cov_pnl_output[5, 3] *= -1
                cov_pnl_output[3, 5] *= -1

                mean_return_output = mean_pnl_output / initial_prices_output
                mean_return_output[5] = mean_pnl_output[5] / initial_prices_output[4]
                vol_return_output = np.sqrt(np.diag(cov_pnl_output)) / initial_prices_output
                vol_return_output[5] = np.sqrt(np.diag(cov_pnl_output))[5] / initial_prices_output[4]

                # calculate correlation matrix for returns
                corr_mat_output = np.zeros((6, 6))
                for i in range(6):
                    for j in range(6):
                        if i == j:
                            corr_mat_output[i, j] = 1.0
                        else:
                            corr_mat_output[i, j] = cov_pnl_output[i, j] / (np.sqrt(cov_pnl_output[i, i]) * np.sqrt(cov_pnl_output[j, j]))

                print(f"Base case mean return: {(mean_return_output*100).round(2)}")
                print(f"Base case vol return: {(vol_return_output*100).round(2)}")
                print(f"Base case correlation matrix:\n{(corr_mat_output*100).round(0)}")

    # running optimizations using different FX hedging methods (global, pre-hedge, post-hedge)
    cash_asset_weights = dict()
    hedge_ratios = dict()
    mean_return = dict()
    vol_return = dict()

    # calculating the initial prices
    initial_prices = np.exp(a @ config.initial_risk_drivers)

    # calculating the fx delta hedge
    fx_delta_hedge = np.exp(-config.initial_risk_drivers[config.us_1y_rd_ix]) * initial_prices[config.fx_hedge_asset_ix]

    # defining the FX sensitive cash assets
    fx_sensitive_cash_assets = [config.us_eq_asset_ix, config.us_10y_asset_ix]

    # defining cash assets
    cash_assets = [config.us_eq_asset_ix, config.dk_eq_asset_ix, config.dk_10y_asset_ix, config.us_10y_asset_ix]

    # calculating the efficient frontiers for each parameter combination
    for (hedged_fx_return, equity_fx_corr), (mean_pnl, cov_pnl) in pnl_dist.items():

        (mean_frontier_global,
         vol_frontier_global,
         holdings_frontier_global) = optimization.calculate_global_efficient_frontier(mean_pnl,
                                                                                      cov_pnl,
                                                                                      initial_prices,
                                                                                      config.n_points,
                                                                                      config.fx_hedge_asset_ix,
                                                                                      fx_delta_hedge,
                                                                                      fx_sensitive_cash_assets,
                                                                                      config.min_hedge_ratio,
                                                                                      config.max_hedge_ratio)

        (mean_frontier_pre_hedge,
         vol_frontier_pre_hedge,
         holdings_frontier_pre_hedge) = optimization.calculate_pre_hedge_efficient_frontier(mean_pnl,
                                                                                            cov_pnl,
                                                                                            initial_prices,
                                                                                            config.n_points,
                                                                                            config.fx_hedge_asset_ix,
                                                                                            fx_delta_hedge,
                                                                                            fx_sensitive_cash_assets,
                                                                                            config.min_hedge_ratio,
                                                                                            config.max_hedge_ratio)

        (mean_frontier_post_hedge,
         vol_frontier_post_hedge,
         holdings_frontier_post_hedge) = optimization.calculate_post_hedge_efficient_frontier(mean_pnl,
                                                                                              cov_pnl,
                                                                                              initial_prices,
                                                                                              config.n_points,
                                                                                              config.fx_hedge_asset_ix,
                                                                                              fx_delta_hedge,
                                                                                              fx_sensitive_cash_assets,
                                                                                              config.min_hedge_ratio,
                                                                                              config.max_hedge_ratio)


        # calculating frontiers using different optimization methods
        frontier_holdings = {'Global': holdings_frontier_global,
                             'Pre-hedge': holdings_frontier_pre_hedge,
                             'Post-hedge': holdings_frontier_post_hedge}

        frontier_mean_pnl = {'Global': mean_frontier_global,
                             'Pre-hedge': mean_frontier_pre_hedge,
                             'Post-hedge': mean_frontier_post_hedge}

        frontier_vol_pnl = {'Global': vol_frontier_global,
                            'Pre-hedge': vol_frontier_pre_hedge,
                            'Post-hedge': vol_frontier_post_hedge}

        # storing the cash asset weights, hedge ratios, mean returns and vols for each method
        cash_asset_weights[(hedged_fx_return, equity_fx_corr)] = {}
        hedge_ratios[(hedged_fx_return, equity_fx_corr)] = {}
        mean_return[(hedged_fx_return, equity_fx_corr)] = {}
        vol_return[(hedged_fx_return, equity_fx_corr)] = {}
        for method in ['Global', 'Pre-hedge', 'Post-hedge']:
            cash_asset_mv = (frontier_holdings[method] * initial_prices)[:, cash_assets]
            fx_sensitive_cash_asset_mv = (frontier_holdings[method] * initial_prices)[:, fx_sensitive_cash_assets]
            fx_asset_mv = frontier_holdings[method][:, config.fx_hedge_asset_ix] * fx_delta_hedge
            total_mv = cash_asset_mv.sum(axis=1)
            total_mv_fx_sensitive = fx_sensitive_cash_asset_mv.sum(axis=1)
            cash_asset_weights[(hedged_fx_return, equity_fx_corr)][method] = cash_asset_mv / total_mv.reshape(-1, 1)
            hedge_ratios[(hedged_fx_return, equity_fx_corr)][method] = -fx_asset_mv / total_mv_fx_sensitive
            mean_return[(hedged_fx_return, equity_fx_corr)][method] = frontier_mean_pnl[method] / total_mv
            vol_return[(hedged_fx_return, equity_fx_corr)][method] = frontier_vol_pnl[method] / total_mv

    scenarios = list(pnl_dist.keys())
    plotting.plot_efficient_frontiers(mean_return, vol_return, scenarios)
    for method in ['Global', 'Pre-hedge', 'Post-hedge']:
        plotting.plot_optimal_allocations(vol_return, cash_asset_weights, hedge_ratios, scenarios, method)

    # conducting simulation analysis for each parameter combination
    sample_global_mean_return = dict()
    sample_global_vol_return = dict()
    sample_pre_hedge_mean_return = dict()
    sample_pre_hedge_vol_return = dict()
    sample_post_hedge_mean_return = dict()
    sample_post_hedge_vol_return = dict()
    optimal_portfolio_mean_return = dict()
    optimal_portfolio_vol_return = dict()
    for (hedged_fx_return, equity_fx_corr), (true_mean_invariant, true_cov_invariant) in invariant_dist.items():

        # generating a sampling distribution for the mean and covariance of the invariants
        sample_cov_invariant = wishart.rvs(df=config.n_obs, scale=true_cov_invariant / config.n_obs,
                                           size=(config.n_scenarios, config.n_periods))
        sample_mean_invariant = multivariate_normal.rvs(mean=true_mean_invariant, cov=true_cov_invariant / config.n_obs,
                                                        size=(config.n_scenarios, config.n_periods))

        # calculating the distribution for the mean and covariance of the risk drivers
        true_cov_risk_driver = true_cov_invariant.copy()
        true_mean_risk_driver = true_mean_invariant.copy() + config.initial_risk_drivers
        sample_cov_risk_driver = sample_cov_invariant.copy()
        sample_mean_risk_driver = sample_mean_invariant.copy() + config.initial_risk_drivers

        # computing the lognormal parameters for the distribution of the prices
        true_mean_param_lognormal = b @ true_mean_risk_driver.T
        true_cov_param_lognormal = b @ true_cov_risk_driver @ b.T
        sample_mean_param_lognormal = b @ sample_mean_risk_driver.T
        sample_cov_param_lognormal = b @ sample_cov_risk_driver @ b.T

        # computing the distribution for the prices
        true_mean_price = np.exp(true_mean_param_lognormal + 0.5 * np.diag(true_cov_param_lognormal))
        true_cov_price = np.outer(true_mean_price, true_mean_price) * (np.exp(true_cov_param_lognormal) - 1)

        cov_term = np.array([np.diag(cov_param) for cov_param in sample_cov_param_lognormal]).T
        sample_mean_price = np.exp(sample_mean_param_lognormal + 0.5 * cov_term)

        outer_prod_term = np.array([np.outer(mean_price, mean_price) for mean_price in sample_mean_price.T])
        exp_term = np.array([np.exp(cov_param) - 1 for cov_param in sample_cov_param_lognormal])
        sample_cov_price = np.array([opt * exp for opt, exp in zip(outer_prod_term, exp_term)])

        # computing the distribution for the pnl
        true_mean_pnl = true_mean_price - initial_prices
        true_cov_pnl = true_cov_price.copy()

        sample_mean_pnl = sample_mean_price - initial_prices.reshape(n_assets, -1)
        sample_cov_pnl = sample_cov_price.copy()

        # computing the sample optimal holdings for different methods
        sample_global_frontier_holdings = []
        sample_pre_hedge_frontier_holdings = []
        sample_post_hedge_frontier_holdings = []

        for i in range(config.n_scenarios):

            holdings_frontier_global = optimization.calculate_max_utility_portfolio_global(sample_mean_pnl[:, i],
                                                                                           sample_cov_pnl[i],
                                                                                           initial_prices,
                                                                                           config.risk_aversion,
                                                                                           None,
                                                                                           config.fx_hedge_asset_ix,
                                                                                           fx_delta_hedge,
                                                                                           fx_sensitive_cash_assets,
                                                                                           config.min_hedge_ratio,
                                                                                           config.max_hedge_ratio)

            holdings_frontier_pre_hedge = optimization.calculate_max_utility_portfolio_pre_hedge(sample_mean_pnl[:, i],
                                                                                           sample_cov_pnl[i],
                                                                                           initial_prices,
                                                                                           config.risk_aversion,
                                                                                           None,
                                                                                           config.fx_hedge_asset_ix,
                                                                                           fx_delta_hedge,
                                                                                           fx_sensitive_cash_assets,
                                                                                           config.min_hedge_ratio,
                                                                                           config.max_hedge_ratio)

            holdings_frontier_post_hedge = optimization.calculate_max_utility_portfolio_post_hedge(sample_mean_pnl[:, i],
                                                                                           sample_cov_pnl[i],
                                                                                           initial_prices,
                                                                                           config.risk_aversion,
                                                                                           None,
                                                                                           config.fx_hedge_asset_ix,
                                                                                           fx_delta_hedge,
                                                                                           fx_sensitive_cash_assets,
                                                                                           config.min_hedge_ratio,
                                                                                           config.max_hedge_ratio)

            sample_global_frontier_holdings.append(holdings_frontier_global)
            sample_pre_hedge_frontier_holdings.append(holdings_frontier_pre_hedge)
            sample_post_hedge_frontier_holdings.append(holdings_frontier_post_hedge)

            print(f"Scenario {i + 1}/{config.n_scenarios} processed.")

        sample_global_frontier_holdings = np.array([h if h is not None else np.full(n_assets, np.nan)
                                                    for h in sample_global_frontier_holdings])

        sample_pre_hedge_frontier_holdings = np.array([h if h is not None else np.full(n_assets, np.nan)
                                                       for h in sample_pre_hedge_frontier_holdings])

        sample_post_hedge_frontier_holdings = np.array([h if h is not None else np.full(n_assets, np.nan)
                                                        for h in sample_post_hedge_frontier_holdings])

        true_global_frontier_holdings = optimization.calculate_max_utility_portfolio_global(true_mean_pnl,
                                                                                            true_cov_pnl,
                                                                                            initial_prices,
                                                                                            config.risk_aversion,
                                                                                            None,
                                                                                            config.fx_hedge_asset_ix,
                                                                                            fx_delta_hedge,
                                                                                            fx_sensitive_cash_assets,
                                                                                            config.min_hedge_ratio,
                                                                                            config.max_hedge_ratio)

        optimal_portfolio_mean_pnl = true_global_frontier_holdings @ true_mean_pnl
        optimal_portfolio_vol_pnl = np.sqrt(true_global_frontier_holdings @ true_cov_pnl @ true_global_frontier_holdings)

        global_mean_pnl = np.array(sample_global_frontier_holdings) @ true_mean_pnl
        pre_hedge_mean_pnl = np.array(sample_pre_hedge_frontier_holdings) @ true_mean_pnl
        post_hedge_mean_pnl = np.array(sample_post_hedge_frontier_holdings) @ true_mean_pnl

        global_vol_pnl = np.sqrt(np.sum(np.array(sample_global_frontier_holdings) @ true_cov_pnl
                                        * np.array(sample_global_frontier_holdings), axis=1))
        pre_hedge_vol_pnl = np.sqrt(np.sum(np.array(sample_pre_hedge_frontier_holdings) @ true_cov_pnl
                                           * np.array(sample_pre_hedge_frontier_holdings), axis=1))
        post_hedge_vol_pnl = np.sqrt(np.sum(np.array(sample_post_hedge_frontier_holdings) @ true_cov_pnl
                                            * np.array(sample_post_hedge_frontier_holdings), axis=1))

        optimal_portfolio_market_value = true_global_frontier_holdings[cash_assets] @ initial_prices[cash_assets]
        global_market_value = np.array(sample_global_frontier_holdings)[:, cash_assets] @ initial_prices[cash_assets]
        pre_hedge_market_value = np.array(sample_pre_hedge_frontier_holdings)[:, cash_assets] @ initial_prices[cash_assets]
        post_hedge_market_value = np.array(sample_post_hedge_frontier_holdings)[:, cash_assets] @ initial_prices[cash_assets]

        optimal_portfolio_mean_return[(hedged_fx_return, equity_fx_corr)] = (optimal_portfolio_mean_pnl
                                                                             / optimal_portfolio_market_value)
        global_mean_return = global_mean_pnl / global_market_value
        pre_hedge_mean_return = pre_hedge_mean_pnl / pre_hedge_market_value
        post_hedge_mean_return = post_hedge_mean_pnl / post_hedge_market_value

        optimal_portfolio_vol_return[(hedged_fx_return, equity_fx_corr)] = (optimal_portfolio_vol_pnl
                                                                            / optimal_portfolio_market_value)
        global_vol_return = global_vol_pnl / global_market_value
        pre_hedge_vol_return = pre_hedge_vol_pnl / pre_hedge_market_value
        post_hedge_vol_return = post_hedge_vol_pnl / post_hedge_market_value

        sample_global_mean_return[(hedged_fx_return, equity_fx_corr)] = global_mean_return
        sample_global_vol_return[(hedged_fx_return, equity_fx_corr)] = global_vol_return
        sample_pre_hedge_mean_return[(hedged_fx_return, equity_fx_corr)] = pre_hedge_mean_return
        sample_pre_hedge_vol_return[(hedged_fx_return, equity_fx_corr)] = pre_hedge_vol_return
        sample_post_hedge_mean_return[(hedged_fx_return, equity_fx_corr)] = post_hedge_mean_return
        sample_post_hedge_vol_return[(hedged_fx_return, equity_fx_corr)] = post_hedge_vol_return

    sample_mean_return = dict()
    sample_mean_return['Global'] = sample_global_mean_return
    sample_mean_return['Pre-hedge'] = sample_pre_hedge_mean_return
    sample_mean_return['Post-hedge'] = sample_post_hedge_mean_return

    sample_vol_return = dict()
    sample_vol_return['Global'] = sample_global_vol_return
    sample_vol_return['Pre-hedge'] = sample_pre_hedge_vol_return
    sample_vol_return['Post-hedge'] = sample_post_hedge_vol_return

    plotting.plot_resampled_portfolios(mean_return, vol_return, optimal_portfolio_mean_return,
                                       optimal_portfolio_vol_return, sample_mean_return, sample_vol_return)

    # compute certainty equivalents for the different methods and scenarios
    sample_certainty_equivalent = dict()
    optimal_portfolio_certainty_equivalent = dict()

    for (hedged_fx_return, equity_fx_corr) in scenarios:

        sample_certainty_equivalent[(hedged_fx_return, equity_fx_corr)] = dict()

        for method in ['Global', 'Pre-hedge', 'Post-hedge']:

            ce = optimization.compute_quadratic_ce(sample_mean_return[method][(hedged_fx_return, equity_fx_corr)],
                                                   config.risk_aversion,
                                                   sample_vol_return[method][(hedged_fx_return, equity_fx_corr)] ** 2)

            sample_certainty_equivalent[(hedged_fx_return, equity_fx_corr)][method] = ce


        optimal_portfolio_certainty_equivalent[(hedged_fx_return, equity_fx_corr)] = optimization.compute_quadratic_ce(
            optimal_portfolio_mean_return[(hedged_fx_return, equity_fx_corr)],
            config.risk_aversion,
            optimal_portfolio_vol_return[(hedged_fx_return, equity_fx_corr)] ** 2
        )

    # compute the opportunity cost of each method for each scenario as CE of the optimal portfolio minus CE of the method
    opportunity_cost = dict()
    for (hedged_fx_return, equity_fx_corr) in scenarios:

        opportunity_cost[(hedged_fx_return, equity_fx_corr)] = dict()

        for method in ['Global', 'Pre-hedge', 'Post-hedge']:
            opportunity_cost[(hedged_fx_return, equity_fx_corr)][method] = (
                optimal_portfolio_certainty_equivalent[(hedged_fx_return, equity_fx_corr)] -
                sample_certainty_equivalent[(hedged_fx_return, equity_fx_corr)][method]
            )

    plotting.plot_opportunity_cost_violins(scenarios, opportunity_cost)


if __name__ == '__main__':
    main()
