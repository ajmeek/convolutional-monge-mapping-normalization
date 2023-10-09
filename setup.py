from setuptools import setup, find_packages

setup(
    name='cmm',
    version='0',
    description='Convolutional Monge Mapping Normalization',

    # The project's main homepage.
    # url='https://github.com/tgnassou/da-toolbox',

    # Author details
    author='Théo Gnassounou',
    author_email='theo.gnassounou@inria.fr',

    # Choose your license
    license='MIT-License',
    # What does your project relate to?
    keywords='monge mapping optimal transport',

    packages=find_packages(),
    install_requires=[
        'mne',
        'numpy',
        'scipy',
        'seaborn',
        'matplotlib',
        'scikit-learn',
    ],
)
