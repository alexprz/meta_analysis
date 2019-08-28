from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
    'numpy',
    'pandas',
    'scipy',
    'nilearn',
    'sklearn',
    'matplotlib',
    'seaborn',
    'colorama',
    'python-dotenv'
]

setup(
    name="meta_analysis",
    version="0.0.1",
    author="Alexandre PEREZ",
    author_email="alexperez@wanadoo.fr",
    description="A package to perform meta analysis on brain studies.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/alexprz/brain_mapping",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
