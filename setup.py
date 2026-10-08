"""Packaging configuration for kimi-delta-attention.

The distribution version is read from ``kda/__init__.py`` so that the
installed package metadata and ``kda.__version__`` can never drift apart.
"""

import re
from pathlib import Path

from setuptools import setup, find_packages

_ROOT = Path(__file__).parent.resolve()

with open(_ROOT / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def _read_version() -> str:
    """Return the version string declared by ``kda/__init__.py``.

    The package ``__init__`` is the single source of truth for the version.
    Importing it here would require torch to already be installed, so the
    declaration is parsed statically instead.

    Returns:
        The version string assigned to ``__version__`` in ``kda/__init__.py``.

    Raises:
        RuntimeError: If ``kda/__init__.py`` declares no ``__version__``.
    """
    init_source = (_ROOT / "kda" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(
        r"^__version__\s*=\s*[\"']([^\"']+)[\"']", init_source, re.MULTILINE
    )
    if match is None:
        raise RuntimeError("kda/__init__.py does not declare __version__")
    return match.group(1)


setup(
    name="kimi-delta-attention",
    version=_read_version(),
    author="hwilner",
    description="Educational implementation of Kimi Delta Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hwilner/kimi-delta-attention",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
)
