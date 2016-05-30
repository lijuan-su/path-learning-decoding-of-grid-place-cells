import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

# plt.figure('aa')
# plt.plot(np.arange(10))

# plt.figure(33)
# plt.plot(-np.arange(3, 50), 'r-')

# plt.figure()
# plt.plot(np.random.randn(10000), 'g-', rasterized=True)

def plot3D(): 
    np.random.seed(0)
    n = 100000
    x = np.random.standard_normal(n)
    y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    plt.figure('bb')
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(121)
    plt.hexbin(x, y, cmap=plt.cm.YlOrRd_r)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title("Hexagon binning")
    cb = plt.colorbar()
    cb.set_label('counts')

    plt.subplot(122)
    plt.hexbin(x, y, bins='log', cmap=plt.cm.YlOrRd_r)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title("With a log color scale")
    cb = plt.colorbar()
    cb.set_label('log10(N)')

# plot3D()
# print (plt.get_fignums())
# print (plt.get_figlabels())

# for i in plt.get_fignums():
#     print ('figure_%d.pdf' % i)
#     # print (i)
#     # print (plt.figure(i))
#     # plt.savefig('figure%d.pdf' % i)

# for i in plt.get_figlabels():
#     print ('figure_%s.pdf' % i)
