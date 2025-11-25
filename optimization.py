import numpy as np
import cvxpy as cp
from typing import List, Union, Tuple

SOLVER = 'SCS'  # Default solver for CVXPY problems
MAXITER = 100_000  # Maximum number of iterations for the solver
EPS = 1e-10  # Tolerance for convergence
VERBOSE = False  # Verbosity flag for solver output

def define_budget_constraint(h, prices=None, indices=None, budget=1.0) -> cp.Constraint:
    """
    Create a budget constraint for portfolio optimization.

    Parameters
    ----------
    h : cp.Variable or np.ndarray
        Portfolio holdings variable.
    prices : np.ndarray or None
        Asset prices.
    indices : list or np.ndarray or None
        Indices of assets to include in the constraint. If None, use all.
    budget : float
        Total budget (default 1.0).

    Returns
    -------
    constraint : cp.Constraint
        CVXPY constraint object.
    """

    if prices is None:
        prices = np.ones_like(h)

    if indices is not None:
        return cp.sum(cp.multiply(h[indices], prices[indices])) == budget
    else:
        return cp.sum(cp.multiply(h, prices)) == budget


def define_hedge_ratio_constraints(h: cp.Variable, prices: np.ndarray, hedge_index: int,
                                   fx_sensitive_cash_assets: Union[np.ndarray, List], fx_delta_hedge: float,
                                   min_hedge_ratio: float = 0.0, max_hedge_ratio: float = 1.0) -> List[cp.Constraint]:

    """
    Create minimum and maximum hedge ratio constraints.

    Parameters
    ----------
    h : cp.Variable
        Portfolio holdings variable.
    prices : np.ndarray
        The prices of the assets in the optimization.
    hedge_index : int
        Index of the hedge asset in the holdings variable (h).
    fx_sensitive_cash_assets : np.ndarray
        Indices of the FX sensitive cash assets (assets to be hedged).
    fx_delta_hedge : float
        The FX delta of the hedge asset (market value of foreign currency leg).
    min_hedge_ratio : float
        Minimum allowed hedge ratio (default is 0)
    max_hedge_ratio : float
        Maximum allowed hedge ratio (default is 1)

    Returns
    -------
    constraints : List[cp.Constraint]
        List of CVXPY constraints.
    """

    constraints = [h[hedge_index] * fx_delta_hedge <= -min_hedge_ratio
                   * h[fx_sensitive_cash_assets] @ prices[fx_sensitive_cash_assets],
                   h[hedge_index] * fx_delta_hedge >= -max_hedge_ratio
                   * h[fx_sensitive_cash_assets] @ prices[fx_sensitive_cash_assets]]

    return constraints


def calculate_min_variance_portfolio_global(cov_pnl: np.ndarray,
                                            prices: np.ndarray,
                                            hedge_index: int,
                                            fx_delta_hedge: float,
                                            fx_sensitive_cash_assets: List[int],
                                            min_hedge_ratio: float = 0.0,
                                            max_hedge_ratio: float = 1.0,
                                            budget: float = 1.0,
                                            solver: str = SOLVER) -> np.ndarray:

    """
    Calculate the minimum variance portfolio given a covariance matrix.

    Parameters
    ----------
    cov_pnl : np.ndarray
        Covariance matrix of the pnl of assets (n_assets, n_assets).
    prices : np.ndarray
        The prices of the assets.
    hedge_index : int
        The index of the hedge asset.
    fx_delta_hedge : float
        The FX delta of the hedge asset (market value of foreign currency leg).
    fx_sensitive_cash_assets : List[int]
        Indices of the FX sensitive cash assets (assets to be hedged).
    min_hedge_ratio : float
        Minimum allowed hedge ratio (default is 0).
    max_hedge_ratio : float
        Maximum allowed hedge ratio (default is 1).
    budget : float
        The budget constraint.
    solver : str, optional
        The CVXPY solver to use (default is SOLVER).

    Returns
    -------
    holdings : np.ndarray
        Optimal portfolio holdings (n_assets,).

    """

    # defining variables
    n = cov_pnl.shape[0]
    h = cp.Variable(n)
    cash_asset_indices = [i for i in range(n) if i != hedge_index]

    # define budget constraint
    budget_constraint = define_budget_constraint(h, prices=prices, indices=cash_asset_indices, budget=budget)

    # define long-only constraint
    long_only_constraint = [h[i] >= 0 for i in cash_asset_indices]

    # define hedge ratio constraint
    hedge_constraint = define_hedge_ratio_constraints(h, prices=prices, hedge_index=hedge_index,
                                                      fx_sensitive_cash_assets=fx_sensitive_cash_assets,
                                                      fx_delta_hedge=fx_delta_hedge,
                                                      min_hedge_ratio=min_hedge_ratio,
                                                      max_hedge_ratio=max_hedge_ratio)

    # combine constraints
    constraints = [budget_constraint] + hedge_constraint + long_only_constraint

    # defining objective function
    objective = cp.Minimize(cp.quad_form(h, cov_pnl))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, max_iters=MAXITER, eps=EPS, verbose=VERBOSE)

    # returning optimal portfolio holdings
    return h.value


def calculate_min_variance_portfolio_pre_hedge(cov_mat: np.ndarray,
                                               prices: np.ndarray,
                                               hedge_index: int,
                                               fx_delta_hedge: float,
                                               fx_sensitive_cash_assets: List[int],
                                               min_hedge_ratio: float = 0.0,
                                               max_hedge_ratio: float = 1.0,
                                               budget: float = 1.0,
                                               solver: str = SOLVER) -> np.ndarray:

    """
    Minimize portfolio variance using the pre-hedging method with asset-specific hedge ratios:

    1. Optimize cash assets (excluding hedge) for minimum variance.
    2. Given cash allocation, set hedge asset according to asset-specific minimum-variance hedge ratios.

    Parameters
    ----------
    cov_mat : np.ndarray
        The covariance matrix for the P&L of the assets.
    prices : np.ndarray
        The prices of the assets.
    hedge_index : int
        The index of the hedge asset.
    fx_delta_hedge : float
        The FX delta of the hedge asset (market value of foreign currency leg).
    fx_sensitive_cash_assets : List[int]
        Indices of the FX sensitive cash assets (assets to be hedged).
    min_hedge_ratio : float
        Minimum allowed hedge ratio (default is 0).
    max_hedge_ratio : float
        Maximum allowed hedge ratio (default is 1).
    budget : float
        The budget constraint.
    solver : str, optional
        The CVXPY solver to use (default is SOLVER).

    Returns
    -------
    holdings : np.ndarray
        Optimal portfolio holdings (cash + hedge).
    """
    # defining variables
    n = cov_mat.shape[0]
    h = cp.Variable(n)
    cash_asset_indices = [i for i in range(n) if i != hedge_index]

    # define budget constraint
    budget_constraint = define_budget_constraint(h, prices=prices, indices=cash_asset_indices, budget=budget)

    # define long-only constraint
    long_only_constraint = [h[i] >= 0 for i in cash_asset_indices]

    # define hedge ratio constraints as the minimum-variance hedge ratios per asset
    asset_min_var_hedges = calculate_min_var_asset_hedges(cov_mat=cov_mat,
                                                          prices=prices,
                                                          hedge_index=hedge_index,
                                                          fx_delta_hedge=fx_delta_hedge,
                                                          fx_sensitive_cash_assets=fx_sensitive_cash_assets,
                                                          min_hedge_ratio=min_hedge_ratio,
                                                          max_hedge_ratio=max_hedge_ratio,
                                                          solver=solver)

    hedge_holdings_per_unit = []
    for i in range(n):
        if i in fx_sensitive_cash_assets:
            hedge_holdings_per_unit.append(-asset_min_var_hedges[i] * prices[i] / fx_delta_hedge)
        else:
            hedge_holdings_per_unit.append(0.0)

    hedge_holdings_per_unit = np.array(hedge_holdings_per_unit)
    hedge_constraints = h[hedge_index] == h[fx_sensitive_cash_assets] @ hedge_holdings_per_unit[fx_sensitive_cash_assets]

    # combine constraints
    constraints = [budget_constraint, hedge_constraints] + long_only_constraint

    # defining objective function
    objective = cp.Minimize(cp.quad_form(h, cov_mat))

    # solving minimum variance problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, max_iters=MAXITER, eps=EPS, verbose=VERBOSE)

    # calculate maximum return portfolio with variance target
    h = h.value

    return h


def calculate_min_variance_portfolio_post_hedge(cov_mat: np.ndarray,
                                                prices: np.ndarray,
                                                hedge_index: int,
                                                fx_delta_hedge: float,
                                                fx_sensitive_cash_assets: List[int],
                                                min_hedge_ratio: float = 0.0,
                                                max_hedge_ratio: float = 1.0,
                                                budget: float = 1.0,
                                                solver: str = SOLVER) -> np.ndarray:
    """
    Minimize portfolio variance using the post-hedging method:
    1. Optimize cash assets (excluding hedge) for minimum variance (hedge = 0).
    2. With cash allocation fixed, optimize hedge asset holding for min variance, subject to hedge ratio constraints.

    Parameters
    ----------
    cov_mat : np.ndarray
        The covariance matrix for the P&L of the assets.
    prices : np.ndarray
        The prices of the assets.
    hedge_index : int
        The index of the hedge asset.
    fx_delta_hedge : float
        The FX delta of the hedge asset (market value of foreign currency leg).
    fx_sensitive_cash_assets : List[int]
        Indices of the FX sensitive cash assets (assets to be hedged).
    min_hedge_ratio : float
        Minimum allowed hedge ratio (default is 0).
    max_hedge_ratio : float
        Maximum allowed hedge ratio (default is 1).
    budget : float
        The budget constraint.
    solver : str, optional
        The CVXPY solver to use (default is SOLVER).

    Returns
    -------
    holdings : np.ndarray
        Optimal portfolio holdings (cash + hedge).
    """
    n = cov_mat.shape[0]
    cash_asset_indices = [i for i in range(n) if i != hedge_index]

    for i in range(2):

        # defining dynamic variable for post-hedge optimization
        h = cp.Variable(n)

        if i == 0:

            # define hedge ratio constraint to be equal to zero
            hedge_constraint = h[hedge_index] == 0

            # define budget constraint
            budget_constraint = define_budget_constraint(h, prices=prices, indices=cash_asset_indices, budget=budget)

            # define long-only constraint
            long_only_constraint = [h[i] >= 0 for i in cash_asset_indices]

            # define constraints for asset only optimization
            constraints = [budget_constraint] + long_only_constraint + [hedge_constraint]

            # defining objective function
            objective = cp.Minimize(cp.quad_form(h, cov_mat))

            # solving minimum variance problem
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=solver, max_iters=MAXITER, eps=EPS, verbose=VERBOSE)

            # calculate maximum return portfolio with variance target
            pre_hedge_h = h.value

        elif pre_hedge_h is not None:

            # define constraints for cash assets as their pre-hedge holdings
            cash_asset_constraints = [h[i] == pre_hedge_h[i] for i in cash_asset_indices]

            # define constraints for the hedge asset
            hedge_constraint = define_hedge_ratio_constraints(h, prices=prices, hedge_index=hedge_index,
                                                              fx_sensitive_cash_assets=fx_sensitive_cash_assets,
                                                              fx_delta_hedge=fx_delta_hedge,
                                                              min_hedge_ratio=min_hedge_ratio,
                                                              max_hedge_ratio=max_hedge_ratio)

            # combine all constraints
            constraints = cash_asset_constraints + hedge_constraint

            # solving minimum variance problem
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=solver, max_iters=MAXITER, eps=EPS, verbose=VERBOSE)

            # calculate maximum return portfolio with variance target
            h = h.value

            return h


def calculate_max_utility_asset_hedges(mean_pnl: np.ndarray,
                                       cov_mat_pnl: np.ndarray,
                                       prices: np.ndarray,
                                       hedge_index: int,
                                       fx_delta_hedge: float,
                                       fx_sensitive_cash_assets: List[int],
                                       risk_aversion: float,
                                       min_hedge_ratio: float = 0.0,
                                       max_hedge_ratio: float = 1.0,
                                       solver: str = SOLVER) -> List:
    """
    Calculate the maximum-utility hedge for each cash asset using a quadratic utility function.

    Parameters
    ----------
    mean_pnl : np.ndarray
        Expected pnl vector (n_assets,).
    cov_mat_pnl : np.ndarray
        Covariance matrix of the pnl of the assets (n_assets, n_assets).
    prices : np.ndarray
        The prices of the asets.
    hedge_index : int
        Index of the hedge asset.
    fx_delta_hedge : float
        The FX delta of the hedge asset (market value of the foreign leg).
    fx_sensitive_cash_assets : List[int]
        Indices of the FX sensitive cash assets (assets to be hedged).
    risk_aversion : float
        Risk aversion coefficient.
    min_hedge_ratio : float
        Minimum allowed hedge ratio.
    max_hedge_ratio : float
        Maximum allowed hedge ratio.
    solver : str
        The CVXPY solver to use.

    Returns
    -------
    optimal_hedge_ratios : List
        Optimal hedge ratio for each cash asset.
    """
    n = cov_mat_pnl.shape[0]
    optimal_hedge_ratios = []
    for i in range(n):
        if i in fx_sensitive_cash_assets:

            h_fwd = cp.Variable()
            # Quadratic utility: maximize expected return minus risk penalty

            utility = mean_pnl[i] + h_fwd * mean_pnl[hedge_index] - 0.5 * risk_aversion * (
                    cov_mat_pnl[i, i] + h_fwd ** 2 * cov_mat_pnl[hedge_index, hedge_index] + 2 * h_fwd * cov_mat_pnl[i, hedge_index]
            )

            constraints = [h_fwd * fx_delta_hedge <= -min_hedge_ratio * prices[i],
                           h_fwd * fx_delta_hedge >= -max_hedge_ratio * prices[i]]

            problem = cp.Problem(cp.Maximize(utility), constraints)
            problem.solve(solver=solver, max_iters=MAXITER, eps=EPS, verbose=VERBOSE)
            optimal_hedge_ratios.append(-h_fwd.value * fx_delta_hedge / prices[i])

        else:
            optimal_hedge_ratios.append(0.0)

    return optimal_hedge_ratios


def calculate_min_var_asset_hedges(cov_mat: np.ndarray,
                                   prices: np.ndarray,
                                   hedge_index: int,
                                   fx_delta_hedge: float,
                                   fx_sensitive_cash_assets: List[int],
                                   min_hedge_ratio: float = 0.0,
                                   max_hedge_ratio: float = 1.0,
                                   solver: str = SOLVER) -> List:

    """
    Calculate the minimum-variance hedge for a given covariance matrix and array of cash assets.

    Parameters
    ----------
    cov_mat : np.ndarray
        Covariance matrix of the pnl of the assets (n_assets, n_assets).
    prices : np.ndarray
        The prices of the assets.
    hedge_index : int
        Index of the hedge asset.
    fx_delta_hedge : float
        The FX delta of the hedge asset (market value of the foreign leg)
    fx_sensitive_cash_assets : List[int]
        Indices of the FX sensitive cash assets (assets to be hedged).
    min_hedge_ratio : float
        Minimum allowed hedge ratio (default 0.0).
    max_hedge_ratio : float
        Maximum allowed hedge ratio (default 1.0).
    solver : str
        The CVXPY solver to use (default SOLVER).

    Returns
    -------
    optimal_hedge_ratios : List
        Optimal hedge ratio for each cash asset.

    """

    # defining variables
    n = cov_mat.shape[0]

    optimal_hedge_ratios = []
    for i in range(n):

        if i in fx_sensitive_cash_assets:

            h_fwd = cp.Variable()

            objective = (cov_mat[i, i] + h_fwd ** 2 * cov_mat[hedge_index, hedge_index]
                         + 2 * h_fwd * cov_mat[i, hedge_index])

            constraints = [h_fwd * fx_delta_hedge <= -min_hedge_ratio * prices[i],
                           h_fwd * fx_delta_hedge >= -max_hedge_ratio * prices[i]]

            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(solver=solver, max_iters=MAXITER, eps=EPS, verbose=VERBOSE)
            optimal_hedge_ratios.append(-h_fwd.value * fx_delta_hedge / prices[i])

        else:
            # If the cash asset is not FX sensitive, the hedge ratio is zero
            optimal_hedge_ratios.append(0.0)

    return optimal_hedge_ratios


def calculate_max_utility_portfolio(h: cp.Variable,
                                    mean_pnl: np.ndarray,
                                    cov_mat: np.ndarray,
                                    risk_aversion: float = None,
                                    variance_target: float = None,
                                    constraints: list = None,
                                    solver: str = SOLVER) -> np.ndarray:

    """
    Calculate the maximum utility portfolio given expected returns, covariance matrix, and risk aversion.

    Parameters
    ----------
    h: cp.Variable
        Portfolio holdings variable (n_assets,).
    mean_pnl: np.ndarray
        Expected pnl vector of the assets (n_assets,).
    cov_mat: np.ndarray
        Covariance matrix of the pnl of the assets (n_assets, n_assets).
    risk_aversion: float
        Risk aversion coefficient (default None, which means no risk aversion).
    variance_target: float
        The target portfolio variance (default None, which means no variance target).
    constraints: list, optional
        List of CVXPY constraints to apply (default is None).
    solver: str
        The CVXPY solver to use (default SOLVER).

    Returns
    -------
    holdings: np.ndarray
        Optimal portfolio holdings (n_assets,).

    """

    # defining variables
    if variance_target is not None:

        variance_constraint = cp.quad_form(h, cov_mat) <= variance_target
        constraints = constraints + [variance_constraint]

    # calculating optimal portfolio holdings
    objective = cp.Maximize(mean_pnl @ h - risk_aversion * 0.5 * cp.quad_form(h, cov_mat))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, max_iters=MAXITER, eps=EPS, verbose=VERBOSE)

    return h.value

def calculate_max_utility_portfolio_global(mean_pnl: np.ndarray,
                                           cov_pnl: np.ndarray,
                                           prices: np.ndarray,
                                           risk_aversion: float,
                                           variance_target: Union[float, None],
                                           hedge_index: int,
                                           fx_delta_hedge: float,
                                           fx_sensitive_cash_assets: List[int],
                                           min_hedge_ratio: float = 0.0,
                                           max_hedge_ratio: float = 1.0,
                                           budget: float = 1.0,
                                           solver: str = SOLVER) -> np.ndarray:

    """
    Calculate the maximum utility portfolio with a given risk aversion coefficient as a global optimization problem.
    I.e. the FX hedge is optimized simultaneously with cash assets holdings.

    Parameters
    ----------
    mean_pnl : np.ndarray
        Expected pnl vector of the assets (n_assets,).
    cov_pnl : np.ndarray
        Covariance matrix of the pnl of the assets (n_assets, n_assets).
    prices : np.ndarray
        The prices of the assets.
    risk_aversion : float
        Risk aversion coefficient.
    variance_target : float
        The target portfolio variance.
    hedge_index : int
        Index of the hedge asset (default -1, which means the last asset).
    fx_delta_hedge : float
        The FX delta of the hedge asset (market value of the foreign leg).
    fx_sensitive_cash_assets : List[int]
        Indices of the FX sensitive cash assets (assets to be hedged).
    min_hedge_ratio : float
        Minimum allowed hedge ratio (default 0.0).
    max_hedge_ratio : float
        Maximum allowed hedge ratio (default 1.0).
    budget : float
        The budget constraint.
    solver : str
        The CVXPY solver to use (default SOLVER).

    Returns
    -------
    holdings : np.ndarray
        Optimal portfolio holdings.

    """

    # defining variables
    n = cov_pnl.shape[0]
    h = cp.Variable(n)
    cash_asset_indices = [i for i in range(n) if i != hedge_index]

    # define budget constraint
    budget_constraint = define_budget_constraint(h, prices=prices, indices=cash_asset_indices, budget=budget)

    # define long-only constraint
    long_only_constraint = [h[i] >= 0 for i in cash_asset_indices]

    # define hedge ratio constraint
    hedge_constraint = define_hedge_ratio_constraints(h, prices=prices, hedge_index=hedge_index,
                                                      fx_sensitive_cash_assets=fx_sensitive_cash_assets,
                                                      fx_delta_hedge=fx_delta_hedge,
                                                      min_hedge_ratio=min_hedge_ratio, max_hedge_ratio=max_hedge_ratio)

    # combine constraints
    constraints = [budget_constraint] + hedge_constraint + long_only_constraint

    # calculate maximum utility portfolio with risk aversion coefficient
    h = calculate_max_utility_portfolio(h, mean_pnl=mean_pnl, cov_mat=cov_pnl, risk_aversion=risk_aversion,
                                        variance_target=variance_target,
                                        constraints=constraints, solver=solver)

    return h


def calculate_max_utility_portfolio_pre_hedge(mean_pnl: np.ndarray,
                                              cov_pnl: np.ndarray,
                                              prices: np.ndarray,
                                              risk_aversion: float,
                                              variance_target: Union[float, None],
                                              hedge_index: int,
                                              fx_delta_hedge: float,
                                              fx_sensitive_cash_assets: List[int],
                                              min_hedge_ratio: float = 0.0,
                                              max_hedge_ratio: float = 1.0,
                                              budget: float = 1.0,
                                              solver: str = SOLVER) -> np.ndarray:
    """
    Calculate the maximum utility portfolio with a given risk aversion coefficient using pre-determined asset-specific
    hedge ratios, i.e. the FX hedge is optimized separately from cash assets holdings (pre-hedge optimization).

    Parameters
    ----------
    mean_pnl : np.ndarray
        Expected pnl vector of the assets (n_assets,).
    cov_pnl : np.ndarray
        Covariance matrix of the pnl of the assets (n_assets, n_assets).
    prices : np.ndarray
        The prices of the assets.
    risk_aversion : float
        Risk aversion coefficient.
    variance_target : float
        The target portfolio variance.
    hedge_index : int
        Index of the hedge asset (default -1, which means the last asset).
    fx_delta_hedge : float,
        The FX delta of the hedge asset (market value of the foreign leg).
    fx_sensitive_cash_assets : List[int]
        Indices of the FX sensitive cash assets (assets to be hedged).
    min_hedge_ratio : float
        Minimum allowed hedge ratio (default 0.0).
    max_hedge_ratio : float
        Maximum allowed hedge ratio (default 1.0).
    budget : float
        The budget constraint.
    solver : str
        The CVXPY solver to use (default SOLVER).

    Returns
    -------
    holdings : np.ndarray
        Optimal portfolio holdings.

    """

    # defining variables
    n = cov_pnl.shape[0]
    h = cp.Variable(n)
    cash_asset_indices = [i for i in range(n) if i != hedge_index]

    # define budget constraint
    budget_constraint = define_budget_constraint(h, prices=prices, indices=cash_asset_indices, budget=budget)

    # define long-only constraint
    long_only_constraint = [h[i] >= 0 for i in cash_asset_indices]

    # define hedge ratio constraints as the maximum-utility hedge ratios per asset
    assets_max_util_hedges = calculate_max_utility_asset_hedges(mean_pnl=mean_pnl,
                                                                cov_mat_pnl=cov_pnl,
                                                                prices=prices,
                                                                hedge_index=hedge_index,
                                                                fx_delta_hedge=fx_delta_hedge,
                                                                fx_sensitive_cash_assets=fx_sensitive_cash_assets,
                                                                risk_aversion=risk_aversion,
                                                                min_hedge_ratio=min_hedge_ratio,
                                                                max_hedge_ratio=max_hedge_ratio,
                                                                solver=solver)

    hedge_holdings_per_unit = []
    for i in range(n):
        if i in fx_sensitive_cash_assets:
            hedge_holdings_per_unit.append(-assets_max_util_hedges[i] * prices[i] / fx_delta_hedge)
        else:
            hedge_holdings_per_unit.append(0.0)

    hedge_holdings_per_unit = np.array(hedge_holdings_per_unit)
    hedge_constraints = h[hedge_index] == h[fx_sensitive_cash_assets] @ hedge_holdings_per_unit[fx_sensitive_cash_assets]

    # combine constraints
    constraints = [budget_constraint, hedge_constraints] + long_only_constraint

    # calculate maximum return portfolio with variance target
    h = calculate_max_utility_portfolio(h, mean_pnl=mean_pnl, cov_mat=cov_pnl,
                                        risk_aversion=risk_aversion,
                                        variance_target=variance_target,
                                        constraints=constraints, solver=solver)

    return h


def calculate_max_utility_portfolio_post_hedge(mean_pnl: np.ndarray,
                                               cov_pnl: np.ndarray,
                                               prices: np.ndarray,
                                               risk_aversion: float,
                                               variance_target: Union[float, None],
                                               hedge_index: int,
                                               fx_delta_hedge: float,
                                               fx_sensitive_cash_assets: List[int],
                                               min_hedge_ratio: float = 0.0,
                                               max_hedge_ratio: float = 1.0,
                                               budget : float = 1.0,
                                               solver: str = SOLVER) -> np.ndarray:
    """
    Calculate the maximum utility portfolio with a given risk aversion coefficient using pre-determined asset weights,
    i.e. the FX hedge is optimized separately from cash assets holdings (post-hedge optimization).

    Parameters
    ----------
    mean_pnl : np.ndarray
        Expected pnl vector of the assets (n_assets,).
    cov_pnl : np.ndarray
        Covariance matrix for the pnl of the assets (n_assets, n_assets).
    prices : np.ndarray
        The prices of the assets.
    risk_aversion : float
        Risk aversion coefficient.
    variance_target : float
        The target portfolio variance.
    hedge_index : int
        Index of the hedge asset (default -1, which means the last asset).
    fx_delta_hedge : float
        The FX delta of the hedge asset (market value of the foreign leg).
    fx_sensitive_cash_assets : List[int]
        Indices of the FX sensitive cash assets (assets to be hedged).
    min_hedge_ratio : float
        Minimum allowed hedge ratio (default 0.0).
    max_hedge_ratio : float
        Maximum allowed hedge ratio (default 1.0).
    budget : float
        The budget constraint.
    solver : str
        The CVXPY solver to use (default SOLVER).

    Returns
    -------
    holdings : np.ndarray
        Optimal portfolio holdings.

    """

    # defining static variables
    n = len(mean_pnl)
    cash_asset_indices = [i for i in range(n) if i != hedge_index]

    for i in range(2):

        # defining dynamic variable for post-hedge optimization
        h = cp.Variable(n)

        if i == 0:

            # define hedge ratio constraint to be equal to zero
            hedge_constraint = h[hedge_index] == 0

            # define budget constraint
            budget_constraint = define_budget_constraint(h, prices=prices, indices=cash_asset_indices, budget=budget)

            # define long-only constraint
            long_only_constraint = [h[i] >= 0 for i in cash_asset_indices]

            # define constraints for asset only optimization
            constraints = [budget_constraint] + long_only_constraint + [hedge_constraint]

            # calculate maximum return portfolio with variance target
            pre_hedge_h = calculate_max_utility_portfolio(h, mean_pnl=mean_pnl, cov_mat=cov_pnl, risk_aversion=risk_aversion,
                                                          variance_target=variance_target,
                                                          constraints=constraints, solver=solver)

        elif pre_hedge_h is not None:

            # define constraints for cash assets as their pre-hedge holdings
            cash_asset_constraints = [h[i] == pre_hedge_h[i] for i in cash_asset_indices]

            # define constraints for the hedge asset
            hedge_constraint = define_hedge_ratio_constraints(h,
                                                              prices=prices, hedge_index=hedge_index,
                                                              fx_sensitive_cash_assets=fx_sensitive_cash_assets,
                                                              fx_delta_hedge=fx_delta_hedge,
                                                              min_hedge_ratio=min_hedge_ratio,
                                                              max_hedge_ratio=max_hedge_ratio)

            # combine all constraints
            constraints = cash_asset_constraints + hedge_constraint

            # calculate maximum return portfolio with variance target
            h = calculate_max_utility_portfolio(h, mean_pnl=mean_pnl, cov_mat=cov_pnl, risk_aversion=risk_aversion,
                                                variance_target=variance_target,
                                                constraints=constraints, solver=solver)

            return h


def calculate_global_efficient_frontier(mean_pnl: np.ndarray, cov_pnl: np.ndarray, prices: np.ndarray,
                                        n_points: int, hedge_index: int, fx_delta_hedge: float,
                                        fx_sensitive_cash_assets: List[int], min_hedge_ratio: float = 0.0,
                                        max_hedge_ratio: float = 1.0, budget: float = 1.0,
                                        solver=SOLVER) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Calculate the efficient frontier for the global optimization method.

    Parameters
    ----------
    mean_pnl:
        Expected pnl vector of the assets (n_assets,).
    cov_pnl:
        Covariance matrix of the pnl of the assets (n_assets, n_assets).
    prices:
        The prices of the assets.
    n_points:
        Number of points on the efficient frontier.
    hedge_index:
        Index of the hedge asset.
    fx_delta_hedge:
        The FX delta of the hedge asset (market value of the foreign leg).
    fx_sensitive_cash_assets:
        Indices of the FX sensitive cash assets (assets to be hedged).
    min_hedge_ratio:
        Minimum allowed hedge ratio.
    max_hedge_ratio:
        Maximum allowed hedge ratio.
    budget:
        The budget constraint.
    solver:
        The CVXPY solver to use.

    Returns
    -------
    mean_frontier : np.ndarray
        Expected returns on the efficient frontier (n_points,).
    vol_frontier : np.ndarray
        Standard deviations on the efficient frontier (n_points,).
    holdings_frontier : np.ndarray
        Optimal portfolio holdings on the efficient frontier (n_points, n_assets).
    """

    n_assets = len(mean_pnl)

    # spread out lambda using a function which has small intervals to begin with and large intervals in the end
    points_per_interval = int(n_points / 2)
    lambda_targets = np.concat([np.linspace(0, 9, points_per_interval),
                                np.linspace(10, 500, points_per_interval)])

    mean_frontier, vol_frontier, holdings_frontier = [], [], []

    for lt in lambda_targets:

        h = calculate_max_utility_portfolio_global(mean_pnl=mean_pnl, cov_pnl=cov_pnl, prices=prices, risk_aversion=lt,
                                                   variance_target=None, hedge_index=hedge_index,
                                                   fx_delta_hedge=fx_delta_hedge,
                                                   fx_sensitive_cash_assets=fx_sensitive_cash_assets,
                                                   min_hedge_ratio=min_hedge_ratio,
                                                   max_hedge_ratio=max_hedge_ratio,
                                                   budget=budget,
                                                   solver=solver)

        if h is None:
            mean_frontier.append(np.nan)
            vol_frontier.append(np.nan)
            holdings_frontier.append(np.full(n_assets, np.nan))
        else:
            mean_frontier.append(h @ mean_pnl)
            vol_frontier.append(np.sqrt(h @ cov_pnl @ h))
            holdings_frontier.append(h)

    return np.array(mean_frontier), np.array(vol_frontier), np.array(holdings_frontier)


def calculate_pre_hedge_efficient_frontier(mean_pnl: np.ndarray, cov_pnl: np.ndarray, prices: np.ndarray,
                                           n_points: int, hedge_index: int, fx_delta_hedge: float,
                                           fx_sensitive_cash_assets: List[int], min_hedge_ratio: float = 0.0,
                                           max_hedge_ratio: float = 1.0, budget: float = 1.0,
                                           solver=SOLVER) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Calculate the efficient frontier for the pre-hedge optimization method.

    Parameters
    ----------
    mean_pnl: np.ndarray
        Expected pnl vector of the assets (n_assets,).
    cov_pnl: np.ndarray
        Covariance matrix of the pnl of the assets (n_assets, n_assets).
    prices: np.ndarray
        The prices of the assets.
    n_points: int
        Number of points on the efficient frontier.
    hedge_index: int
        Index of the hedge asset.
    fx_delta_hedge: float
        The FX delta of the hedge asset (market value of the foreign leg).
    fx_sensitive_cash_assets:
        Indices of the FX sensitive cash assets (assets to be hedged).
    min_hedge_ratio:
        Minimum allowed hedge ratio.
    max_hedge_ratio:
        Maximum allowed hedge ratio.
    budget:
        The budget constraint.
    solver:
        The CVXPY solver to use.

    Returns
    -------
    mean_frontier : np.ndarray
        Expected returns on the efficient frontier (n_points,).
    vol_frontier : np.ndarray
        Standard deviations on the efficient frontier (n_points,).
    holdings_frontier : np.ndarray
        Optimal portfolio holdings on the efficient frontier (n_points, n_assets).
    """

    n_assets = len(mean_pnl)

    # spread out lambda using a function which has small intervals to begin with and large intervals in the end
    points_per_interval = int(n_points / 2)
    lambda_targets = np.concat([np.linspace(0, 9, points_per_interval),
                                np.linspace(10, 500, points_per_interval)])
    mean_frontier, vol_frontier, holdings_frontier = [], [], []

    for lt in lambda_targets:

        h = calculate_max_utility_portfolio_pre_hedge(mean_pnl=mean_pnl, cov_pnl=cov_pnl, prices=prices,
                                                      risk_aversion=lt, variance_target=None, hedge_index=hedge_index,
                                                      fx_delta_hedge=fx_delta_hedge,
                                                      fx_sensitive_cash_assets=fx_sensitive_cash_assets,
                                                      min_hedge_ratio=min_hedge_ratio,
                                                      max_hedge_ratio=max_hedge_ratio,
                                                      budget=budget,
                                                      solver=solver)
        if h is None:
            mean_frontier.append(np.nan)
            vol_frontier.append(np.nan)
            holdings_frontier.append(np.full(n_assets, np.nan))
        else:
            mean_frontier.append(h @ mean_pnl)
            vol_frontier.append(np.sqrt(h @ cov_pnl @ h))
            holdings_frontier.append(h)

    return np.array(mean_frontier), np.array(vol_frontier), np.array(holdings_frontier)


def calculate_post_hedge_efficient_frontier(mean_pnl: np.ndarray, cov_pnl: np.ndarray, prices: np.ndarray,
                                            n_points: int, hedge_index: int, fx_delta_hedge: float,
                                            fx_sensitive_cash_assets: List[int], min_hedge_ratio: float = 0.0,
                                            max_hedge_ratio: float = 1.0, budget: float = 1.0,
                                            solver=SOLVER) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the efficient frontier for the post-hedge optimization method.

    Parameters
    ----------
    mean_pnl: np.ndarray
        Expected pnl vector of the assets (n_assets,).
    cov_pnl: np.ndarray
        Covariance matrix of the pnl of the assets (n_assets, n_assets).
    prices: np.ndarray
        The prices of the assets.
    n_points: int
        Number of points on the efficient frontier.
    hedge_index: int
        Index of the hedge asset.
    fx_delta_hedge: float
        The FX delta of the hedge asset (market value of the foreign leg).
    fx_sensitive_cash_assets:
        Indices of the FX sensitive cash assets (assets to be hedged).
    min_hedge_ratio: float
        Minimum allowed hedge ratio.
    max_hedge_ratio: float
        Maximum allowed hedge ratio.
    budget: float
        The budget constraint.
    solver: str
        The CVXPY solver to use.

    Returns
    -------
    mean_frontier : np.ndarray
        Expected returns on the efficient frontier (n_points,).
    vol_frontier : np.ndarray
        Standard deviations on the efficient frontier (n_points,).
    holdings_frontier : np.ndarray
        Optimal portfolio holdings on the efficient frontier (n_points, n_assets).

    """
    n_assets = len(mean_pnl)

    # spread out lambda using a function which has small intervals to begin with and large intervals in the end
    points_per_interval = int(n_points / 3)
    lambda_targets = np.concat([np.linspace(0, 5, points_per_interval),
                                np.linspace(6, 10, points_per_interval),
                                np.linspace(10, 500, points_per_interval)])
    mean_frontier, vol_frontier, holdings_frontier = [], [], []

    for lt in lambda_targets:
        h = calculate_max_utility_portfolio_post_hedge(mean_pnl=mean_pnl, cov_pnl=cov_pnl, prices=prices,
                                                       risk_aversion=lt, variance_target=None, hedge_index=hedge_index,
                                                       fx_delta_hedge=fx_delta_hedge,
                                                       fx_sensitive_cash_assets=fx_sensitive_cash_assets,
                                                       min_hedge_ratio=min_hedge_ratio,
                                                       max_hedge_ratio=max_hedge_ratio,
                                                       budget=budget,
                                                       solver=solver)
        if h is None:
            mean_frontier.append(np.nan)
            vol_frontier.append(np.nan)
            holdings_frontier.append(np.full(n_assets, np.nan))
        else:
            mean_frontier.append(h @ mean_pnl)
            vol_frontier.append(np.sqrt(h @ cov_pnl @ h))
            holdings_frontier.append(h)

    return np.array(mean_frontier), np.array(vol_frontier), np.array(holdings_frontier)


def compute_quadratic_ce(mean_pnl: Union[float, np.ndarray], risk_aversion: Union[float, np.ndarray],
                         variance_pnl: Union[float, np.ndarray]) -> Union[float, np.ndarray]:

    """
    Compute the certainty equivalent for a quadratic utility function.

    Parameters
    ----------
    mean_pnl: Union[float, np.ndarray]
        Expected pnl.
    risk_aversion: Union[float, np.ndarray]
        Risk aversion coefficient.
    variance_pnl: Union[float, np.ndarray]
        Variance of the pnl.

    Returns
    -------
    Union[float, np.ndarray]
        Certainty equivalent.

    """

    return mean_pnl - 0.5 * risk_aversion * variance_pnl
