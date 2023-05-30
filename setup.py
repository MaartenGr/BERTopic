from setuptools import setup, find_packages

test_packages = [
    "pytest>=5.4.3",
    "pytest-cov>=2.6.1"
]

docs_packages = [
    "mkdocs==1.1",
    "mkdocs-material==4.6.3",
    "mkdocstrings==0.8.0",
]

base_packages = [
    "numpy>=1.20.0",
    "hdbscan>=0.8.29",
    "umap-learn>=0.5.0",
    "pandas>=1.1.5",
    "scikit-learn>=0.22.2.post1",
    "tqdm>=4.41.1",
    "sentence-transformers>=0.4.1",
    "plotly>=4.7.0",
]

flair_packages = [
    "transformers>=3.5.1",
    "torch>=1.4.0",
    "flair>=0.7"
]

spacy_packages = [
    "spacy>=3.0.1"
]

use_packages = [
    "tensorflow",
    "tensorflow_hub",
    "tensorflow_text"
]

gensim_packages = [
    "gensim>=4.0.0"
]

vision_packages = [
    "Pillow>=9.2.0",
    "accelerate>=0.19.0"  # To prevent "cannot import name 'PartialState' from 'accelerate'"
]

extra_packages = flair_packages + spacy_packages + use_packages + gensim_packages

dev_packages = docs_packages + test_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bertopic",
    packages=find_packages(exclude=["notebooks", "docs"]),
    version="0.15.0",
    author="Maarten P. Grootendorst",
    author_email="maartengrootendorst@gmail.com",
    description="BERTopic performs topic Modeling with state-of-the-art transformer models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaartenGr/BERTopic",
    project_urls={
        "Documentation": "https://maartengr.github.io/BERTopic/",
        "Source Code": "https://github.com/MaartenGr/BERTopic/",
        "Issue Tracker": "https://github.com/MaartenGr/BERTopic/issues",
    },
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=base_packages,
    extras_require={
        "test": test_packages,
        "docs": docs_packages,
        "dev": dev_packages,
        "flair": flair_packages,
        "spacy": spacy_packages,
        "use": use_packages,
        "gensim": gensim_packages,
        "vision": vision_packages
    },
    python_requires='>=3.7',
)
