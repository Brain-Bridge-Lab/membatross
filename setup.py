import setuptools

setuptools.setup(
    name="membatross_functions",
    version="0.1.0",
    author="Fiona Lee",
    author_email="mlee26@uchicago.edu",
    description="Function package for Membatross",
    url="https://github.com/Brain-Bridge-Lab/membatross",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'matplotlib',
        'numpy',
        'scipy',
        'scikit-learn',
        'seaborn',
    ],
)

