import config
import numpy as np
import matplotlib.pyplot as plt
import locale

locale.setlocale(locale.LC_NUMERIC, 'da_DK.UTF-8')

def plot_efficient_frontiers(mean_return: dict, vol_return: dict, scenarios: list):

    """
    Plot efficient frontiers for different hedged FX returns and equity/FX correlations.

    Parameters
    ----------
    mean_return: dict
        A dictionary where keys are tuples of (hedged_fx_return, equity_fx_corr) and values are dictionaries with keys
        'Global', 'Pre-hedge', 'Post-hedge' and values as arrays of mean returns.
    vol_return: dict
        A dictionary where keys are tuples of (hedged_fx_return, equity_fx_corr) and values are dictionaries with keys
        'Global', 'Pre-hedge', 'Post-hedge' and values as arrays of return volatility.
    scenarios: list
        A list of tuples representing different scenarios of (hedged_fx_return, equity_fx_corr).

    """
    filename = 'frontiers.svg'

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Use config variables
    methods = config.methods
    colors = config.method_colors
    linestyles = config.method_linestyles
    names = config.method_names
    zorder = config.method_zorder
    alpha = config.method_alpha
    linewidth = config.method_linewidth
    facecolor = config.facecolor

    # set background color for all axes
    for ax in axes.flatten():
        ax.set_facecolor(facecolor)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Get unique values for row and column titles
    hedged_fx_returns = sorted(set(k[0] for k in scenarios))
    equity_fx_corrs = sorted(set(k[1] for k in scenarios))

    for hedged_fx_return, equity_fx_corr in scenarios:
        # Find row and column indices
        row = hedged_fx_returns.index(hedged_fx_return)
        col = equity_fx_corrs.index(equity_fx_corr)
        ax = axes[row, col]
        for method in methods:
            ax.plot(vol_return[(hedged_fx_return, equity_fx_corr)][method],
                    mean_return[(hedged_fx_return, equity_fx_corr)][method], label=names[method],
                    color=colors[method], linestyle=linestyles[method], zorder=zorder[method],
                    linewidth=linewidth[method],
                    alpha=alpha[method],
                    solid_capstyle='round')
        if row == 0:
            ax.set_title(f'Aktie/valuta korrelation: {equity_fx_corr:.0%}', fontsize=10)
        if col == 0:
            ax.set_ylabel(f'Forventet afkast på afdækning: '
                          f'{hedged_fx_return:.0%}\n\nForventet afkast', fontsize=10)
        else:
            ax.set_ylabel('Forventet afkast')
        ax.set_xlabel('Volatilitet')
        ax.legend(loc='upper left', fontsize=8)

    # Format axes as percentages
    for ax in axes.flatten():
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.1%}'.replace('.', ',')))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.1%}'.replace('.', ',')))

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_optimal_allocations(vol_return: dict, cash_asset_weights: dict, hedge_ratios,
                             scenarios: list, method='Global'):

    """
    Plot optimal allocations and hedge ratios for different hedged FX returns and equity/FX correlations.

    Parameters
    ----------
    vol_return: dict
        A dictionary where keys are tuples of (hedged_fx_return, equity_fx_corr) and values are dictionaries with keys
        'Global', 'Pre-hedge', 'Post-hedge' and values as arrays of return volatility.
    cash_asset_weights: dict
        A dictionary where keys are tuples of (hedged_fx_return, equity_fx_corr) and values are dictionaries with keys
        'Global', 'Pre-hedge', 'Post-hedge' and values as arrays of cash asset weights.
    hedge_ratios
        A dictionary where keys are tuples of (hedged_fx_return, equity_fx_corr) and values are dictionaries with keys
        'Global', 'Pre-hedge', 'Post-hedge' and values as arrays of hedge ratios.
    scenarios: list
        A list of tuples representing different scenarios of (hedged_fx_return, equity_fx_corr).
    method: str
        The method to plot ('Global', 'Pre-hedge', 'Post-hedge').

    """

    filename = f'optimal_allocations_{method}.pdf'

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Get unique values for row and column titles
    hedged_fx_returns = sorted(set(k[0] for k in scenarios))
    equity_fx_corrs = sorted(set(k[1] for k in scenarios))

    for hedged_fx_return, equity_fx_corr in scenarios:
        # Find row and column indices
        row = hedged_fx_returns.index(hedged_fx_return)
        col = equity_fx_corrs.index(equity_fx_corr)
        ax = axes[row, col]

        ax.stackplot(vol_return[(hedged_fx_return, equity_fx_corr)][method],
                     cash_asset_weights[(hedged_fx_return, equity_fx_corr)][method].T,
                     labels=config.cash_asset_labels, alpha=0.8, colors=config.cash_asset_colors)
        ax.plot(vol_return[(hedged_fx_return, equity_fx_corr)][method],
                hedge_ratios[(hedged_fx_return, equity_fx_corr)][method],
                label=config.fx_hedge_label, color=config.fx_hedge_color, linestyle='--', linewidth=1.5)

        if row == 0:
            ax.set_title(f'Aktie/valuta korrelation: {equity_fx_corr:.0%}', fontsize=10)
        if col == 0:
            ax.set_ylabel(f'Forventet afkast på afdækning: '
                          f'{hedged_fx_return:.0%}\n\nOptimale allokeringer/afdækningsgrad', fontsize=10)
        else:
            ax.set_ylabel('Optimale allokeringer')
        ax.set_xlabel('Volatilitet')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_ylim((0, 1))

        # set axis limits for all axes as the max and min observed values for the specific method
        min_vol = np.nanmin(vol_return[(hedged_fx_return, equity_fx_corr)][method])
        max_vol = np.nanmax(vol_return[(hedged_fx_return, equity_fx_corr)][method])
        ax.set_xlim((min_vol, max_vol))

        # Format axes as percentages
        for ax in axes.flatten():
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.1%}'.replace('.', ',')))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.1%}'.replace('.', ',')))

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_resampled_portfolios(mean_return: dict, vol_return: dict, optimal_portfolio_mean_return: dict,
                              optimal_portfolio_vol_return: dict, sample_mean_return: np.ndarray,
                              sample_vol_return: dict):

    """
    Plot resampled portfolios against the true efficient frontier and optimal portfolio.

    Parameters
    ----------
    mean_return: dict
        A dictionary where keys are tuples of (hedged_fx_return, equity_fx_corr) and values are dictionaries with keys
        'Global', 'Pre-hedge', 'Post-hedge' and values as arrays of mean returns.
    vol_return: dict
        A dictionary where keys are tuples of (hedged_fx_return, equity_fx_corr) and values are dictionaries with keys
        'Global', 'Pre-hedge', 'Post-hedge' and values as arrays of return volatility.
    optimal_portfolio_mean_return: dict
        A dictionary where keys are tuples of (hedged_fx_return, equity_fx_corr) and values are the mean return of the
        optimal portfolio.
    optimal_portfolio_vol_return: dict
        A dictionary where keys are tuples of (hedged_fx_return, equity_fx_corr) and values are the volatility of the
        optimal portfolio.
    sample_mean_return: dict
        A dictionary where keys are 'Global', 'Pre-hedge', 'Post-hedge' and values are dictionaries with keys as tuples
        of (hedged_fx_return, equity_fx_corr) and values as arrays of mean returns in samples.
    sample_vol_return: dict
        A dictionary where keys are 'Global', 'Pre-hedge', 'Post-hedge' and values are dictionaries with keys as tuples
        of (hedged_fx_return, equity_fx_corr) and values as arrays of return volatility in samples.

    """

    methods = config.methods
    names = config.method_names
    colors = config.method_colors
    scenarios = list(mean_return.keys())

    for idx, scenario in enumerate(scenarios):
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(20, 6))
        axes = axes.flatten()
        for i, method in enumerate(methods):
            # True efficient frontier
            axes[i].plot(
                vol_return[scenario]['Global'],
                mean_return[scenario]['Global'],
                color='black',
                linewidth=2,
                label='Efficient rand',
                zorder=-1
            )
            # Optimal chosen portfolio
            axes[i].scatter(
                optimal_portfolio_vol_return[scenario],
                optimal_portfolio_mean_return[scenario],
                color='yellow',
                edgecolor='black',
                label='Optimal portefølje',
                marker='*',
                zorder=10,
                s=150
            )
            # Sampled optimal portfolios
            axes[i].scatter(
                sample_vol_return[method][scenario],
                sample_mean_return[method][scenario],
                alpha=.5,
                label=f'{names[method]} (estimeret)',
                color=colors[method],
                s=9,
                edgecolors='none'
            )
            axes[i].legend(loc='upper left')
            axes[i].set_facecolor('#E6E6E6')
            axes[i].grid(True, linestyle='--', alpha=0.1)
            axes[i].set_ylim((0.025, 0.06))
            axes[i].set_xlim((0.06, 0.19))
            axes[i].set_title(f"{names[method]}")

            all_y = []
            all_x = []
            for m in methods:
                all_y.extend(mean_return[scenario][m])
                all_x.extend(vol_return[scenario][m])
                all_y.extend(sample_mean_return[m][scenario])
                all_x.extend(sample_vol_return[m][scenario])
            all_y.append(optimal_portfolio_mean_return[scenario])
            all_x.append(optimal_portfolio_vol_return[scenario])
            y_min, y_max = min(all_y), max(all_y)
            x_min, x_max = min(all_x), max(all_x)
            # Add a small margin
            y_range = y_max - y_min
            x_range = x_max - x_min
            y_min -= 0.05 * y_range
            y_max += 0.05 * y_range
            x_min -= 0.05 * x_range
            x_max += 0.05 * x_range
            for ax in axes:
                ax.set_ylim((y_min, y_max))
                ax.set_xlim((x_min, x_max))
                ax.set_ylabel('Forventet afkast')
                ax.set_xlabel('Volatilitet')
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.1%}'.replace('.', ',')))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.1%}'.replace('.', ',')))

        plt.tight_layout()
        # Create a safe scenario string for the filename
        scenario_str = f"{scenario[0]:.3f}_{scenario[1]:.3f}".replace('.', 'p').replace(',', '_')
        fig.savefig(f"resampled_portfolio_{scenario_str}.svg")
        plt.close(fig)


def plot_opportunity_cost_violins(scenarios: list, opportunity_cost: dict):

    """
    Plot violin plots of opportunity costs for different hedged FX returns and equity/FX correlations.

    Parameters
    ----------
    scenarios: list
        A list of tuples representing different scenarios of (hedged_fx_return, equity_fx_corr).
    opportunity_cost: dict
        A dictionary where keys are tuples of (hedged_fx_return, equity_fx_corr) and values are dictionaries with keys
        'Global', 'Pre-hedge', 'Post-hedge' and values as arrays of opportunity costs.

    """

    # Prepare data for plotting
    methods = config.methods
    colors = config.method_colors
    names = config.method_names
    op_cost_data = {method: [] for method in methods}
    width = 0.2

    for scenario in scenarios:
        for method in methods:
            # Flatten if opportunity cost is an array, else keep as is
            val = opportunity_cost[scenario][method]
            if isinstance(val, np.ndarray):
                op_cost_data[method].append(val)
            else:
                op_cost_data[method].append(np.array([val]))

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = []
    violin_data = []
    violin_colors = []
    for i, method in enumerate(methods):
        for j, vals in enumerate(op_cost_data[method]):
            # Remove NaN values for violin plot
            clean_vals = vals[~np.isnan(vals)]
            if clean_vals.size == 0:
                # If all values are NaN, add a single NaN to avoid errors
                clean_vals = np.array([np.nan])
            positions.append(j + i * width)
            violin_data.append(clean_vals)
            violin_colors.append(colors[method])

    parts = ax.violinplot(violin_data, positions=positions, widths=width * 0.9, showmeans=True, showmedians=False,
                          showextrema=False)

    for pc, color in zip(parts['bodies'], violin_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.8)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)

    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(1.5)

    ax.set_facecolor('#E6E6E6')

    ax.set_ylabel("Alternativomkostning")
    ax.set_xlabel("(Forventet afkast på afdækning, Aktie/valuta korrelation) i pct.")
    ax.set_xticks([j + width for j in range(len(scenarios))])

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.1%}'.replace('.', ',')))

    ax.set_xticklabels(
        [f"({fx_ret * 100:.1f}".replace('.', ',') + f"; {eq_corr * 100:.0f})".replace('.', ',')
         for fx_ret, eq_corr in scenarios],
        rotation=0)

    legend_handles = [plt.Line2D([0], [0], color=colors[method], lw=8, label=names[method]) for i, method in
                      enumerate(methods)]
    ax.legend(handles=legend_handles)
    for lh in ax.get_legend().legend_handles:
        lh.set_alpha(0.5)
    plt.tight_layout()
    plt.savefig("opportunity_cost_violins.svg")
    plt.close()