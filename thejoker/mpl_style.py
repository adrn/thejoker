from astropy.visualization import astropy_mpl_style
mpl_style = astropy_mpl_style.copy()
mpl_style['figure.figsize'] = (8,6)
mpl_style['axes.grid'] = False
mpl_style['lines.markeredgewidth'] = 0

# bigger fonts
mpl_style['axes.titlesize'] = 26
mpl_style['axes.labelsize'] = 22
mpl_style['xtick.labelsize'] = 18
mpl_style['ytick.labelsize'] = 18
