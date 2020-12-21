import setuptools

test_packages = [
    "pytest>=5.4.3",
    "pytest-cov>=2.6.1"
]

docs_packages = [
    "mkdocs>=1.1",
    "mkdocs-material>=4.6.3",
    "mkdocstrings>=0.8.0",
]


base_packages = [
    "tqdm>=4.41.1",
    "numpy>=1.18.5",
    "umap-learn>=0.4.6",
    "hdbscan>=0.8.26",
    "pandas>=1.1.5",
    "scipy>=1.3.1",
    "scikit-learn>=0.22.2.post1",
    "joblib==0.17.0",
    "torch>=1.4.0",
    "sentence-transformers>=0.3.9"
]

visualization_packages = [
    "matplotlib>=3.2.2",
    "plotly>=4.7.0"
]

dev_packages = docs_packages + test_packages + visualization_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bertopic",
    packages=["bertopic"],
    version="0.4.0",
    author="Maarten Grootendorst",
    author_email="maartengrootendorst@gmail.com",
    description="BERTopic performs topic Modeling with state-of-the-art transformer models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaartenGr/BERTopic",
    keywords="nlp bert topic modeling embeddings",
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=base_packages,
    extras_require={
        "test": test_packages,
        "docs": docs_packages,
        "dev": dev_packages,
        "visualization": visualization_packages
    },
    python_requires='>=3.6',
)
