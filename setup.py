import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SCNN",
    version="0.0.1",
    author="Demetres Kostas",
    author_email="demetres.kostas@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SPOClab-ca/SCNN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPL V3",
        "Operating System :: OS Independent",
    ],
)
