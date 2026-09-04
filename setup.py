from setuptools import setup, find_packages

setup(
    name="uzbek-mwe-tokenizer",
    version="0.1.0",
    author="Maksud Sharipov",
    author_email="maqsbek72@gmail.com",
    description="Transformer-based intelligent model for identifying multi-word lexical units in Uzbek.",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "UzTransliterator",
        "sentencepiece"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
