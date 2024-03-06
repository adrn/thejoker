import importlib.metadata
import os

project = "thejoker"
copyright = "2024, Adrian Price-Whelan"
author = "Adrian Price-Whelan"
version = release = importlib.metadata.version("thejoker")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "nbsphinx",
    # "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "rtds_action",
    "matplotlib.sphinxext.plot_directive",
    "matplotlib.sphinxext.figmpl_directive",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

# HTML theme
html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "thejoker"
html_logo = "_static/thejoker.png"
html_favicon = "_static/icon.ico"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/adrn/thejoker",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "classic",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable", None),
    "h5py": ("http://docs.h5py.org/en/latest/", None),
    "twobody": ("https://twobody.readthedocs.io/en/latest/", None),
    "schwimmbad": ("https://schwimmbad.readthedocs.io/en/latest/", None),
    "numpy": (
        "https://numpy.org/doc/stable/",
        (None, "http://data.astropy.org/intersphinx/numpy.inv"),
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/",
        (None, "http://data.astropy.org/intersphinx/scipy.inv"),
    ),
    "matplotlib": (
        "https://matplotlib.org/stable/",
        (None, "http://data.astropy.org/intersphinx/matplotlib.inv"),
    ),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

plot_srcset = ["2.0x"]  # for retina displays
plot_rcparams = {"font.size": 16, "font.family": "serif", "figure.figsize": (6, 4)}
plot_apply_rcparams = True

always_document_param_types = True

# We execute the tutorial notebooks using GitHub Actions and upload to RTD:
nbsphinx_execute = "never"

# The name of your GitHub repository
rtds_action_github_repo = "adrn/thejoker"

# The path where the artifact should be extracted
# Note: this is relative to the conf.py file!
rtds_action_path = "examples"

# The "prefix" used in the `upload-artifact` step of the action
rtds_action_artifact_prefix = "notebooks-for-"

# A GitHub personal access token is required, more info below
rtds_action_github_token = os.environ.get("GITHUB_TOKEN", "")

# Whether or not to raise an error on Read the Docs if the
# artifact containing the notebooks can't be downloaded (optional)
rtds_action_error_if_missing = False
