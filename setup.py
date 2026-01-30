from setuptools import find_packages, setup

setup(
    name="oa-classifier",
    version="1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "pipeline=oa_classifier.main:main",
        ],
    },
)
