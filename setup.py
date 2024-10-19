from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requires = [req.strip() for req in f if req]

setup(
    name="persona_understanding",
    version="0.0.1",
    description="Research for LLM's persona understanding",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Zhong Qishuai",
    author_email="ferdinandzhong@gmail.com",
    url="git@github.com:FerdinandZhong/persona_understanding",
    packages=find_packages(exclude=["tests*", "example*"]),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">3.9",
    install_requires=requires,
    extras_require={
        "dev": [
            "pytest>=6",
            "flake8>=3.8",
            "black>=20.8b1",
            "isort>=5.6",
            "autoflake>=1.4",
        ],
    },
    zip_safe=False,
    entry_points={
        "console_scripts": [],
    },
)