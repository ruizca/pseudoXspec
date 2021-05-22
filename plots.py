import numpy as np
import matplotlib.gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
import sherpa.astro.ui as shp
from scipy.stats import binned_statistic


# plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("axes.formatter", limits=(-3, 3))
plt.style.use("bmh")


def _get_plot_data(ids, emin, emax):
    all_model = []
    all_emodel = []
    all_data = []
    all_dataxerr = []
    all_datayerr = []
    all_edata = []
    all_ratio = []
    all_ratioerr = []

    # Get data and model for each spectrum
    for sid in ids:
        d = shp.get_data_plot(sid)
        m = shp.get_model_plot(sid)
        e = (m.xhi + m.xlo) / 2
        bins = np.concatenate((d.x - d.xerr / 2, [d.x[-1] + d.xerr[-1]]))

        model = m.y
        model_de = model * (m.xhi - m.xlo)
        model_binned, foo1, foo2 = binned_statistic(
            e, model_de, bins=bins, statistic="sum"
        )
        model_binned = model_binned / d.xerr

        # delchi = resid/d.yerr
        ratio = d.y / model_binned

        mask_data = np.logical_and(d.x + d.xerr / 2 >= emin, d.x - d.xerr / 2 <= emax)
        mask_model = np.logical_and(e >= emin, e <= emax)

        all_model.append(model[mask_model])
        all_emodel.append(e[mask_model])
        all_data.append(d.y[mask_data])
        all_dataxerr.append(d.xerr[mask_data])
        all_datayerr.append(d.yerr[mask_data])
        all_edata.append(d.x[mask_data])
        all_ratio.append(ratio[mask_data])
        all_ratioerr.append(d.yerr[mask_data] / model_binned[mask_data])

    return (
        all_model,
        all_emodel,
        all_data,
        all_dataxerr,
        all_datayerr,
        all_edata,
        all_ratio,
        all_ratioerr,
    )


def _ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value / 10 ** exp
    if exp == 0 or exp == 1:
        return r"${0:d}$".format(int(value))
    if exp == -1:
        return r"${0:.1f}$".format(value)
    else:
        if base == 1:
            return r"$10^{{{0:d}}}$".format(int(exp))
        else:
            return r"${0:d}\\times10^{{{1:d}}}$".format(int(base), int(exp))


def spectra_bestfit(ids, emin=0, emax=np.inf, prefix=None):
    (
        all_model,
        all_emodel,
        all_data,
        all_dataxerr,
        all_datayerr,
        all_edata,
        all_ratio,
        all_ratioerr,
    ) = _get_plot_data(ids, emin, emax)

    # Show all spectra in one plot
    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    ax0 = plt.subplot(gs[0])
    for model, emodel, data, dataxerr, datayerr, edata in zip(
        all_model, all_emodel, all_data, all_dataxerr, all_datayerr, all_edata,
    ):
        ax0.errorbar(
            edata,
            data,
            xerr=dataxerr / 2,
            yerr=datayerr,
            fmt="o",
            ms=5,
            elinewidth=1.25,
            capsize=2,
            ls="None",
            zorder=1000,
        )
        ax0.loglog(emodel, model, c="red", alpha=0.5)

    ax0.set_ylabel(r"count rate / $\mathrm{s}^{-1}\:\mathrm{keV}^{-1}$")
    ax0.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_ticks_format))

    ax1 = plt.subplot(gs[1], sharex=ax0)
    for edata, dataxerr, ratio, ratioerr in zip(
        all_edata, all_dataxerr, all_ratio, all_ratioerr,
    ):
        ax1.errorbar(
            edata,
            ratio,
            xerr=dataxerr / 2,
            yerr=ratioerr,
            elinewidth=1.25,
            capsize=2,
            ls="None",
            zorder=1000,
        )

    ax1.axhline(1, ls="--", c="gray")

    plt.setp(ax0.get_xticklabels(), visible=False)

    ax1.set_yscale("log")
    ax1.set_xlabel("Energy / keV")
    ax1.set_ylabel("ratio")

    ax1.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_ticks_format))
    #ax1.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 3.0, 5.0)))
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_ticks_format))
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    plt.xlim(emin, emax)
    plt.tight_layout()

    if prefix:
        fig.savefig(f"{prefix}_srcspec_ALL.png")
    else:
        fig.show()

    plt.close(fig)
