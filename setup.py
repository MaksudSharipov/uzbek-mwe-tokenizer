from setuptools import setup, find_packages

setup(
    name="uzbek-mwe-tokenizer",
    version="0.1.0",
    author="Maksud Sharipov, Abdusobir Saidov, Azizbek Ruzmetov, Shukurla Bekchanov, Ogabek Sobirov, Surayyo Sattarova",
    author_email="maqsbek72@gmail.com",
    description="A Python library for Transformer-based multi-word expression identification in Uzbek.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MaksudSharipov/uzbek-mwe-tokenizer",
    project_urls={
        "Homepage": "https://github.com/MaksudSharipov/uzbek-mwe-tokenizer",
        "Repository": "https://github.com/MaksudSharipov/uzbek-mwe-tokenizer",
        "PyPI": "https://pypi.org/project/uzbek-mwe-tokenizer/",
        "Model": "https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT",
        "Issues": "https://github.com/MaksudSharipov/uzbek-mwe-tokenizer/issues",
    },
    packages=find_packages(include=["uzbek_mwe_tokenizer", "uzbek_mwe_tokenizer.*"]),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "UzTransliterator",
        "sentencepiece>=0.1.96",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
)
