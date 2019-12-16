import logging
from pprint import pprint as pp

import click

import coloredlogs
import matplotlib.style as mp_style
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

LOG = logging.getLogger(__name__)

####### Helper functions #######


def despine_plot(offset=None, trim=False, left=False):
    """Alter the frames of the plot"""
    if trim:
        offset = 10

    LOG.info(
        "Despining plot with offset: {0}, trim: {1}, left: {2}".format(
            offset, trim, left
        )
    )

    sns.despine(offset=offset, trim=trim, left=left)


def produce_plot(outfile=None, axis=None, context=None):
    """Either show the plot or save to a file

    Args:
        outfile(str): Path to the file to plot
        axis(matplotlib.Axis)
        context(dict)
    """
    if axis:
        axis.set(
            xlabel=context.get("xlabel"),
            ylabel=context.get("ylabel"),
            title=context.get("title"),
        )

    if context.get("despine"):
        despine_plot(
            context.get("offset"),
            context.get("trim", False),
            context.get("left", False),
        )

    if outfile:
        LOG.info("Saving plot to %s", outfile)
        plt.savefig(outfile)
        return

    LOG.info("Producing plot")
    plt.show()


@click.command()
@click.option(
    "-k",
    "--kind",
    type=click.Choice(["point", "bar", "strip", "swarm", "box", "violin", "boxen"]),
    default="strip",
    help="Cat plots can have different styles",
)
@click.option("-x", "--x-axis", help="Specify the x-axis")
@click.option("-y", "--y-axis", help="Specify the x-axis")
@click.option("--hue", help="Specify the hue")
@click.option("--jitter/--no-jitter", default=True)
@click.pass_context
def catplot(ctx, kind, x_axis, y_axis, hue, jitter):
    """Plots different kinds of category plots"""
    outfile = ctx.obj.get("outfile")
    data = ctx.obj.get("data")
    if data is None:
        data = sns.load_dataset("tips")
    kwargs = {}
    if x_axis:
        kwargs["x"] = x_axis
        if y_axis:
            kwargs["y"] = y_axis
    if hue:
        kwargs["hue"] = hue
    kwargs["data"] = data
    kwargs["kind"] = kind
    if not kind in ["box", "bar", "swarm", "boxen"]:
        kwargs["jitter"] = jitter
    LOG.info("Creating catplot with kind %s", kind)

    ax = sns.catplot(**kwargs)

    produce_plot(outfile, axis=ax, context=ctx.obj)


@click.command()
@click.option("-f", "--flip", default=1)
@click.pass_context
def sinplot(ctx, flip):
    """Plots a sinplot"""
    outfile = ctx.obj.get("outfile")
    x = np.linspace(0, 14, 100)
    LOG.info("Creating sinplot")
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * 0.5) * (7 - i) * flip)

    if ctx.obj.get("despine"):
        despine_plot(
            ctx.obj.get("offset"),
            ctx.obj.get("trim", False),
            ctx.obj.get("left", False),
        )

    produce_plot(outfile)


@click.command()
@click.option("-x", "--x-axis", help="Specify the x-axis")
@click.pass_context
def countplot(ctx, x_axis):
    """Plots a countplot for categorical data"""
    outfile = ctx.obj.get("outfile")
    data = ctx.obj.get("data")
    if data is None:
        data = sns.load_dataset("titanic")
        x_axis = "class"
    if not x_axis:
        LOG.warning("Please specify what axis to plot")
        return

    LOG.info("Creating countplot")

    ax = sns.countplot(x=x_axis, data=data)

    produce_plot(outfile, axis=ax, context=ctx.obj)


@click.command()
@click.pass_context
def boxplot(ctx):
    """Plots a boxplot"""
    outfile = ctx.obj.get("outfile")
    LOG.info("Creating boxplot")
    data = ctx.obj.get("data")
    if data is None:
        data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
    ax = sns.boxplot(data=data)

    produce_plot(outfile, axis=ax, context=ctx.obj)


@click.command()
@click.option("--scale", type=click.Choice(["area", "width", "count"]), default="area")
@click.pass_context
def violinplot(ctx, scale):
    """Plots a violinplot"""
    outfile = ctx.obj.get("outfile")
    LOG.info("Creating violinplot")
    data = ctx.obj.get("data")
    if data is None:
        data = np.random.normal(size=(20, 6)) + np.arange(6) / 2

    ax = sns.violinplot(data=data, scale=scale)

    produce_plot(outfile, axis=ax, context=ctx.obj)


@click.command()
@click.option("-c", "--columns", multiple=True, help="Specify name of columns to plot")
@click.pass_context
def distplot(ctx, columns):
    """Plots a distplot"""
    outfile = ctx.obj.get("outfile")
    LOG.info("Creating distplot")
    data = ctx.obj.get("data")
    if data is None:
        data = pd.DataFrame({"x": np.random.normal(size=100)})
        columns = "x"
    f, ax = plt.subplots()
    if not columns:
        LOG.warning("Please specify what columns to plot")
        ctx.abort()
    try:
        for i in columns:
            sns.distplot(data[i])
    except KeyError as err:
        LOG.warning("Column %s does not exist in data", i)
        LOG.info("Existing columns: %s", ", ".join(list(data)))
        ctx.abort()

    produce_plot(outfile, axis=ax, context=ctx.obj)


@click.command()
@click.option("-n", "--nr-colors", default=10)
@click.option("-s", "--saturation", type=float)
@click.option("-l", "--lightness", type=float)
@click.pass_context
def palplot(ctx, nr_colors, saturation, lightness):
    """Plots a color palette"""
    outfile = ctx.obj.get("outfile")
    LOG.info("Creating palplot")
    palette = ctx.obj.get("palette")
    kwargs = {}
    if saturation:
        kwargs["s"] = saturation
    if lightness:
        kwargs["l"] = lightness
    if not palette:
        current_palette = sns.color_palette()
    else:
        if palette == "hls":
            current_palette = sns.hls_palette(nr_colors, **kwargs)
        else:
            current_palette = sns.color_palette(palette, nr_colors, **kwargs)
    ax = sns.palplot(current_palette)

    produce_plot(outfile, axis=ax, context=ctx.obj)


@click.command()
@click.option("-x", "--x-axis", help="Specify the x-axis")
@click.option("-y", "--y-axis", help="Specify the x-axis")
@click.option("--hue", help="Specify the hue")
@click.option(
    "--err-style",
    type=click.Choice(["band", "bars"]),
    default="band",
    show_default=True,
    help="Specify the error styles",
)
@click.pass_context
def lineplot(ctx, x_axis, y_axis, hue, err_style):
    """Plots a line plot"""
    outfile = ctx.obj.get("outfile")
    LOG.info("Creating lineplot")
    kwargs = {}

    if ctx.obj.get("despine"):
        despine_plot(
            ctx.obj.get("offset"),
            ctx.obj.get("trim", False),
            ctx.obj.get("left", False),
        )

    data = ctx.obj.get("data")
    if data is None:
        data = sns.load_dataset("fmri")
        x_axis = "timepoint"
        y_axis = "signal"
        hue = "event"

    ax = sns.lineplot(x=x_axis, y=y_axis, hue=hue, data=data, err_style=err_style)

    produce_plot(outfile, axis=ax, context=ctx.obj)


@click.command()
@click.option("-x", "--x-axis", help="Specify the x-axis")
@click.option("-y", "--y-axis", help="Specify the y-axis")
@click.option(
    "--truncate",
    is_flag=True,
    help="If regression line should be truncated to last point of data or not",
)
@click.option(
    "--x-estimator",
    type=click.Choice(["mean"]),
    help="If regression line should be truncated to last point of data or not",
)
@click.option("--x-jitter", type=float)
@click.option("--logx", is_flag=True)
@click.pass_context
def regplot(ctx, x_axis, y_axis, x_estimator, x_jitter, truncate, logx):
    """Plots a line plot"""
    outfile = ctx.obj.get("outfile")
    LOG.info("Creating regplot")
    kwargs = {}

    if ctx.obj.get("despine"):
        despine_plot(
            ctx.obj.get("offset"),
            ctx.obj.get("trim", False),
            ctx.obj.get("left", False),
        )

    data = ctx.obj.get("data")
    if data is None:
        data = sns.load_dataset("tips")
        x_axis = "size"
        y_axis = "total_bill"

    if x_estimator:
        if x_estimator == "mean":
            x_estimator = np.mean

    ax = sns.regplot(
        x=x_axis,
        y=y_axis,
        data=data,
        x_estimator=x_estimator,
        x_jitter=x_jitter,
        logx=logx,
        truncate=truncate,
    )

    produce_plot(outfile, axis=ax, context=ctx.obj)


@click.command()
@click.option("-x", "--x-axis", help="Specify the x-axis")
@click.option("-y", "--y-axis", help="Specify the x-axis")
@click.option("--hue", help="Specify the hue")
@click.option(
    "--err-style",
    type=click.Choice(["band", "bars"]),
    default="band",
    show_default=True,
    help="Specify the error styles",
)
@click.pass_context
def lmplot(ctx, x_axis, y_axis, hue, err_style):
    """Plots a line plot"""
    outfile = ctx.obj.get("outfile")
    LOG.info("Creating lineplot")
    kwargs = {}

    if ctx.obj.get("despine"):
        despine_plot(
            ctx.obj.get("offset"),
            ctx.obj.get("trim", False),
            ctx.obj.get("left", False),
        )

    data = ctx.obj.get("data")
    if data is None:
        data = sns.load_dataset("fmri")
        x_axis = "timepoint"
        y_axis = "signal"
        hue = "event"

    ax = sns.lmplot(x=x_axis, y=y_axis, hue=hue, data=data)

    produce_plot(outfile, axis=ax, context=ctx.obj)


@click.command()
@click.pass_context
@click.option("-c", "--columns", multiple=True, help="Specify name of columns to plot")
def kdeplot(ctx, columns):
    """Fit and plot a univariate or bivariate kernel density estimate"""
    outfile = ctx.obj.get("outfile")
    LOG.info("Creating kdeplot")
    x, y = np.random.multivariate_normal([0, 0], [[1, -0.5], [-0.5, 1]], size=300).T
    data = ctx.obj.get("data")
    if data is None:
        data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
        data = pd.DataFrame(data, columns=["x", "y"])
        columns = ("x", "y")

    # cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    # sns.kdeplot(x, y, cmap=cmap, shade=True)

    if not columns:
        LOG.warning("Please specify what columns to plot")
        ctx.abort()

    try:
        for col in columns:
            sns.kdeplot(data[col], shade=True)
    except KeyError as err:
        LOG.warning("Column %s does not exist in data", col)
        LOG.info("Existing columns: %s", ", ".join(list(data)))
        ctx.abort()

    produce_plot(outfile, context=ctx.obj)


@click.group()
@click.option(
    "-s",
    "--style",
    type=click.Choice(["darkgrid", "whitegrid", "dark", "white", "ticks"]),
    default="whitegrid",
    help="Figure Style",
)
@click.option(
    "-c",
    "--scaling",
    type=click.Choice(["paper", "notebook", "talk", "poster"]),
    default="notebook",
    help="Change relative style depending on context",
)
@click.option(
    "--dataset",
    type=click.Choice(["tips", "titanic", "diamonds"]),
    help="If one of the datasets that comes with the distributions should be used",
)
@click.option(
    "--palette",
    type=click.Choice(
        [
            "deep",
            "muted",
            "bright",
            "pastel",
            "dark",
            "colorblind",
            "hls",
            "husl",
            "Blues",
            "BuGn",
            "GnBu",
            "BrBG",
            "RdBu",
            "PuBuGn",
            "coolwarm",
            "cubehelix",
        ]
    ),
    help="Choose color palette",
)
@click.option(
    "--reverse-palette", is_flag=True, help="Reverse the current color palette"
)
@click.option("--dark-palette", is_flag=True, help="Darken the color palette")
@click.option(
    "-f",
    "--file-path",
    type=click.Path(exists=True),
    help="Specify the path to a csv file with data",
)
@click.option("-d", "--despine", is_flag=True, help="Removes spines")
@click.option("-t", "--trim", is_flag=True, help="Trims the spines")
@click.option("--ggstyle", is_flag=True, help="If plot should be ggplot style")
@click.option("-l", "--despine-left", is_flag=True, help="Removes left spine")
@click.option("--xlabel", help="Choose a label for the X axis")
@click.option("--ylabel", help="Choose a label for the Y axis")
@click.option("--title", help="Choose a title for the plot")
@click.option(
    "-o",
    "--outfile",
    # type=click.Path(exists=True),
    help="If plot should be saved to a file",
)
@click.pass_context
def cli(
    ctx,
    style,
    scaling,
    dataset,
    palette,
    reverse_palette,
    dark_palette,
    file_path,
    despine,
    trim,
    despine_left,
    xlabel,
    ylabel,
    title,
    outfile,
    ggstyle,
):
    """Plot results.
    Info from https://seaborn.pydata.org/tutorial/aesthetics.html and
    https://seaborn.pydata.org/tutorial/color_palettes.html#palette-tutorial
    """
    loglevel = "INFO"
    coloredlogs.install(level=loglevel)

    LOG.info("Using style %s", style)
    sns.set_style(style)
    if ggstyle:
        mp_style.use("ggplot")
    LOG.info("Using scaling %s", scaling)
    sns.set_context(scaling)
    if palette:
        LOG.info("Using palette %s", palette)
        if palette not in ["deep", "muted", "bright", "pastel", "dark", "colorblind"]:
            if reverse_palette:
                LOG.info("Using reverse colors")
                palette += "_r"
            elif dark_palette:
                LOG.info("Using dark colors")
                palette += "_d"
    sns.set_palette(palette)
    ctx.obj = {}
    ctx.obj["outfile"] = outfile
    ctx.obj["palette"] = palette
    ctx.obj["xlabel"] = xlabel
    ctx.obj["ylabel"] = ylabel
    ctx.obj["title"] = title

    data = None
    if file_path:
        LOG.info("Loading data frame from %s", file_path)
        data = pd.read_csv(file_path, index_col=0)
    if dataset:
        data = sns.load_dataset(dataset)
    ctx.obj["data"] = data

    ctx.obj["despine"] = despine
    ctx.obj["trim"] = trim
    ctx.obj["left"] = despine_left


cli.add_command(catplot)
cli.add_command(sinplot)
cli.add_command(boxplot)
cli.add_command(violinplot)
cli.add_command(palplot)
cli.add_command(kdeplot)
cli.add_command(distplot)
cli.add_command(lineplot)
cli.add_command(countplot)
cli.add_command(lmplot)
cli.add_command(regplot)

if __name__ == "__main__":
    cli()
