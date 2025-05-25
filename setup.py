from setuptools import setup

APP = ['app.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'iconfile': 'datalens_logo.ico',
    'packages': ['numpy', 'matplotlib', 'spectral', 'sklearn'],
}

setup(
    app=APP,
    name='DataLens_PCA_Viewer',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
