import matplotlib.pyplot as plt

def set_fontsize(label=None, xlabel=None, ylabel=None, tick=None, xtick=None, ytick=None, title=None, set=None):

    fig = plt.gcf()

    if set == 'default':
        label = 14
        tick = 12
        title = 16

    for ax in fig.axes:
        if xlabel is not None:
            ax.xaxis.label.set_size(xlabel)
        elif label is not None:
            ax.xaxis.label.set_size(label)
        if ylabel is not None:
            ax.yaxis.label.set_size(ylabel)
        elif label is not None:
            ax.yaxis.label.set_size(label)

        if xtick is not None:
            for ticklabel in (ax.get_xticklabels()):
                ticklabel.set_fontsize(xtick)
        elif tick is not None:
            for ticklabel in (ax.get_xticklabels()):
                ticklabel.set_fontsize(tick)
        if ytick is not None:
            for ticklabel in (ax.get_yticklabels()):
                ticklabel.set_fontsize(ytick)
        elif tick is not None:
            for ticklabel in (ax.get_yticklabels()):
                ticklabel.set_fontsize(tick)

        if title is not None:
            ax.title.set_fontsize(title)


