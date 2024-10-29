from matplotlib import colors, cm, colormaps
import matplotlib.pyplot as plt
plt.ion()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


a_range = np.linspace(0.1,10,50)
x = np.linspace(0,20,500)
y_list = np.array([a*x for a in a_range])

cmap = colormaps['coolwarm']
# or cmap = truncate_colormap(cm.get_cmap('coolwarm'), 0.5, 1.0)
norm = colors.Normalize(vmin=np.min(a_range), vmax=np.max(a_range)) #can also use LogNorm

plt.figure()
for i, a in enumerate(a_range):
    plt.plot(x,y_list[i,:],color=cmap(norm(a)))
plt.xlabel(r'$x$');plt.ylabel(r'$y$')
sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
sm.set_array([])
c_bar=plt.colorbar(sm,fraction=0.05)#, ticks=np.linspace(0,1,11))
c_bar.set_label(r'Slope')
plt.tight_layout()
plt.savefig('example_colourscale.png')