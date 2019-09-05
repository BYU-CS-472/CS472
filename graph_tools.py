import matplotlib.pyplot as plt
import numpy as np

def graph_function(func,save_path=None):
    """ Simple demonstration of graphing a function
    Args:
        slope (int)
        y_intercept (int)
    Returns:
        Graph of line
    """
    x = np.linspace(-1, 1, 100)
    plt.plot(x, func(x))

    if save_path:
        plt.savefig(save_path) # don't call plt.savefig after plt.show!
    plt.show()

def graph(x, y, labels=None, title="Untitled Graph", xlabel="X", ylabel="Y", points=True, style='fivethirtyeight',
          xlim=None, ylim=None, legend=True, save_path=None):

    """ Graph results
    Args:
        x (array-like): a list of x-coordinates
        y (array-like): a list of y-coordinates
        labels (array-like): a list of integers corresponding to classes
        title (str): Title of graph
        xlabel (str): X-axis title
        ylabel (str): Y-axis title, .5
        points (bool): True will plot points, False a line
        style: Plot style (e.g. ggplot, fivethirtyeight, classic etc.)
        xlim (2-tuple): x-min, x-max
        ylim (2-tuple): y-min, y-max
        save_path (str): where graph should be saved
    Returns:
        (graph)
    """

    # prep data
    x = np.asarray(x)
    y = np.asarray(y)
    labels = np.asarray(labels)

    # set style, create plot
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(8,6))

    # create labels
    point_style = 'o' if points else ''
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    # set graph limits
    if not xlim is None:
        plt.xlim(xlim)
    if not ylim is None:
        plt.ylim(ylim)

    plot_list = []

    for l in np.unique(labels.astype(int)):
        idx = np.where(labels==l)
        plot_list.append(ax.plot(x[idx],y[idx], point_style, label = str(l))[0])

    # Put legend below
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=5, handles=plot_list,
                      facecolor = 'white', edgecolor = 'black')
    plt.show()

    if save_path:
        fig.savefig(save_path)


if __name__ == "__main__":
    y = lambda x: 5 * x**2 + 1 # equation of a parabola
    graph_function(y)
