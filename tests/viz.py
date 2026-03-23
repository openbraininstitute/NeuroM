import pylab as plt

import neurom
from neurom.view.matplotlib_utils import get_figure, plot_style
from neurom.view import plot_morph

neuron = neurom.load_morphology("./morph.swc")

f, ax = get_figure()

plot_morph(neuron, ax=ax, plane="yz")
plot_style(f, ax, white_space=5)

plt.show()
