import matplotlib.pyplot as plt

class Animator:
    """For plotting data in animation without relying on IPython."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        plt.rcParams['figure.figsize'] = figsize
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: self._set_axes(xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        plt.ion()  # Turn on interactive mode

    def _set_axes(self, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        ax = self.axes[0]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid()

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_vals, y_vals, fmt)
        self.config_axes()
        plt.draw()         # Update the plot
        plt.pause(0.001)   # Small pause for the figure to update

if __name__ == '__main__':
    # Testing the Animator class by simulating loss and accuracy curves.
    animator = Animator(xlabel='Epoch', ylabel='Value', legend=['Loss', 'Accuracy'],
                        xlim=(0, 10), ylim=(0, 1), figsize=(5, 3))
    
    import time
    # Simulated data: loss decreases, accuracy increases over epochs
    for epoch in range(11):
        loss = 1.0 / (epoch + 1)
        accuracy = epoch / 10.0
        animator.add(epoch, [loss, accuracy])
        time.sleep(0.5)  # pause to visualize animation update
    plt.ioff()  # Turn off interactive mode
    plt.show()
