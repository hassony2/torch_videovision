import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchvideotransforms",
    version="0.1.2",
    author="Yana Hasson",
    # author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hassony2/torch_videovision",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
    ],
    install_requires=[
        'torch',
        'torchvision',
        'scikit-image',
    ],
    python_requires='>=3.6',
)
