import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


base_packages = [
    "tqdm>=4.49.0",
    "numpy>=1.19.2",
    "umap-learn>=0.4.6",
    "hdbscan>=0.8.26",
    'pandas>=1.1.2',
    "scikit_learn>=0.23.2",
    "sentence_transformers>=0.3.6",
    "joblib>=0.15.1",
    "scipy>=1.5.2"
]

test_packages = [
    "pytest>=5.4.3",
    "pytest-cov>=2.6.1"
]

setuptools.setup(
    name="bertopic",
    packages=["bertopic"],
    version="0.2.0",
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
        },
    python_requires='>=3.6',
)
