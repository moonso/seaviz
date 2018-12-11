import logging

import click
import coloredlogs

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

LOG = logging.getLogger(__name__)

####### Helper functions #######

def despine_plot(offset = None, trim = False, left = False):
    """Alter the frames of the plot"""
    if trim:
        offset = 10

    LOG.info("Despining plot with offset: {0}, trim: {1}, left: {2}".format(
        offset, trim, left
    ))

    sns.despine(offset=offset, trim=trim, left=left)

def produce_plot(outfile):
    """Either show the plot or save to a file"""
    if outfile:
        LOG.info("Saving plot to %s", outfile)
        plt.savefig(outfile)
        return
    LOG.info("Producing plot")
    plt.show()


@click.command()
@click.option('-k', '--kind', 
    type=click.Choice(['point', 'bar', 'strip', 'swarm',
    'box', 'violin', 'boxen']),
    default='strip',
    help="Cat plots can have different styles",
)
@click.option('-x', '--x-axis', 
    help="Specify the x-axis",
)
@click.option('-y', '--y-axis',
    help="Specify the x-axis",
)
@click.option('--hue',
    help="Specify the hue",
)
@click.option('--jitter/--no-jitter',
    default=True
)
@click.pass_context
def catplot(ctx, kind, x_axis, y_axis, hue, jitter):
    """Plots different kinds of category plots"""
    outfile = ctx.obj.get('outfile')
    data = ctx.obj.get('data')
    if data is None:
        data = sns.load_dataset("tips")
    kwargs = {}
    if x_axis:
        kwargs['x'] = x_axis
        if y_axis:
            kwargs['y'] = y_axis
    if hue:
        kwargs['hue'] = hue
    kwargs['data'] = data
    kwargs['kind'] = kind
    if not kind in ['box', 'bar', 'swarm', 'boxen']:
        kwargs['jitter'] = jitter
    LOG.info("Creating catplot with kind %s", kind)
    
    sns.catplot(**kwargs)
    
    if ctx.obj.get('despine'):
        despine_plot(ctx.obj.get('offset'), ctx.obj.get('trim',False), ctx.obj.get('left',False))

    produce_plot(outfile)

@click.command()
@click.option('-f', '--flip', 
    default=1,
)
@click.pass_context
def sinplot(ctx, flip):
    """Plots a sinplot"""
    outfile = ctx.obj.get('outfile')
    x = np.linspace(0, 14, 100)
    LOG.info("Creating sinplot")
    for i in range(1,7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

    if ctx.obj.get('despine'):
        despine_plot(ctx.obj.get('offset'), ctx.obj.get('trim',False), ctx.obj.get('left',False))

    produce_plot(outfile)

@click.command()
@click.pass_context
def boxplot(ctx):
    """Plots a boxplot"""
    outfile = ctx.obj.get('outfile')
    LOG.info("Creating boxplot")
    data = np.random.normal(size=(20,6)) + np.arange(6) / 2
    sns.boxplot(data=data)

    if ctx.obj.get('despine'):
        despine_plot(ctx.obj.get('offset'), ctx.obj.get('trim',False), ctx.obj.get('left',False))

    produce_plot(outfile)

@click.command()
@click.pass_context
def violinplot(ctx):
    """Plots a violinplot"""
    outfile = ctx.obj.get('outfile')
    LOG.info("Creating violinplot")
    data = np.random.normal(size=(20,6)) + np.arange(6) / 2
    f, ax = plt.subplots()
    sns.violinplot(data=data)

    if ctx.obj.get('despine'):
        despine_plot(ctx.obj.get('offset'), ctx.obj.get('trim',False), ctx.obj.get('left',False))

    produce_plot(outfile)

@click.command()
@click.option('-n', '--nr-colors', 
    default=10,
)
@click.option('-s', '--saturation', 
    type=float,
)
@click.option('-l', '--lightness', 
    type=float,
)
@click.pass_context
def palplot(ctx, nr_colors, saturation, lightness):
    """Plots a color palette"""
    outfile = ctx.obj.get('outfile')
    LOG.info("Creating palplot")
    palette = ctx.obj.get('palette')
    kwargs = {}
    if saturation:
        kwargs['s'] = saturation
    if lightness:
        kwargs['l'] = lightness
    if not palette:
        current_palette = sns.color_palette()
    else:
        if palette == 'hls':
            current_palette = sns.hls_palette(nr_colors, **kwargs)
        else:
            current_palette = sns.color_palette(palette, nr_colors, **kwargs)
    sns.palplot(current_palette)

    if ctx.obj.get('despine'):
        despine_plot(ctx.obj.get('offset'), ctx.obj.get('trim',False), ctx.obj.get('left',False))

    produce_plot(outfile)

@click.command()
@click.pass_context
def kdeplot(ctx):
    """Fit and plot a univariate or bivariate kernel density estimate"""
    outfile = ctx.obj.get('outfile')
    LOG.info("Creating kdeplot")
    x, y = np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=300).T
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.kdeplot(x, y, cmap=cmap, shade=True)

    if ctx.obj.get('despine'):
        despine_plot(ctx.obj.get('offset'), ctx.obj.get('trim',False), ctx.obj.get('left',False))

    produce_plot(outfile)



@click.group()
@click.option('-s', '--style', 
    type=click.Choice(['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']),
    default='whitegrid',
    help="Figure Style",
)
@click.option('-c', '--scaling', 
    type=click.Choice(['paper', 'notebook', 'talk', 'poster']),
    default='notebook',
    help="Change relative style depending on context",
)
@click.option('-p', '--plot', 
    type=click.Choice(['sinplot', 'boxplot', 'violinplot', 'palplot', 'kdeplot', 'catplot']),
    help="What kind of plot to check"
)
@click.option('--dataset', 
    type=click.Choice(['tips', 'titanic', 'diamonds']),
    help="If one of the datasets that comes with the distributions should be used"
)
@click.option('--palette', 
    type=click.Choice(['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind', 'hls', 'husl',
                       'Blues', 'BuGn', 'GnBu', 'BrBG', 'RdBu', 'PuBuGn', 'coolwarm','cubehelix']),
    help="Choose color palette"
)
@click.option('--reverse-palette', 
    is_flag=True,
    help="Reverse the current color palette"
)
@click.option('--dark-palette', 
    is_flag=True,
    help="Darken the color palette"
)
@click.option('-f', '--file-path', 
    type=click.Path(exists=True),
    help="Specify the path to a csv file with data"
)
@click.option('-d', '--despine', 
    is_flag=True,
    help="Removes spines"
)
@click.option('-t', '--trim', 
    is_flag=True,
    help="Trims the spines"
)
@click.option('-l', '--despine-left', 
    is_flag=True,
    help="Removes left spine"
)
@click.option('-o', '--outfile', 
    # type=click.Path(exists=True),
    help="If plot should be saved to a file"
)
@click.pass_context
def cli(ctx, style, scaling, plot, dataset, palette, reverse_palette, dark_palette, 
                 file_path, despine, trim, despine_left, outfile):
    """Plot results.
    Info from https://seaborn.pydata.org/tutorial/aesthetics.html and
    https://seaborn.pydata.org/tutorial/color_palettes.html#palette-tutorial
    """
    loglevel = "INFO"
    coloredlogs.install(level=loglevel)
    
    LOG.info("Using style %s", style)
    sns.set_style(style)
    LOG.info("Using scaling %s", scaling)
    sns.set_context(scaling)
    if palette:
        LOG.info("Using palette %s", palette)
        if palette not in ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']:
            if reverse_palette:
                LOG.info("Using reverse colors")
                palette += '_r'
            elif dark_palette:
                LOG.info("Using dark colors")
                palette += '_d'
    sns.set_palette(palette)
    ctx.obj = {}
    ctx.obj['outfile'] = outfile
    ctx.obj['palette'] = palette
    data = None
    if file_path:
        LOG.info("Loading data frame from %s", file_path)
        data = pd.read_csv(file_path, index_col=0)
    if dataset:
        data = sns.load_dataset(dataset)
    ctx.obj['data'] = data

    ctx.obj['despine'] = despine
    ctx.obj['trim'] = trim
    ctx.obj['left'] = despine_left


cli.add_command(catplot)
cli.add_command(sinplot)
cli.add_command(boxplot)
cli.add_command(violinplot)
cli.add_command(palplot)
cli.add_command(kdeplot)

if __name__=='__main__':
    cli()
