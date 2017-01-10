from astropy.visualization import astropy_mpl_style
mpl_style = astropy_mpl_style.copy()
mpl_style['figure.figsize'] = (8,6)
mpl_style['axes.grid'] = False
mpl_style['savefig.facecolor'] = 'none'
mpl_style['savefig.bbox'] = 'tight'
mpl_style['font.family'] = 'serif'
mpl_style['font.serif'] = 'cmr10'
