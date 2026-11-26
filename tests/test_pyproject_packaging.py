"""Consistency guards for pyproject.toml packaging (issue #16).

The repository ships PEP 621 metadata in ``pyproject.toml`` *alongside* the
legacy ``setup.py``. These tests pin the two metadata sources together so
they cannot drift, and pin the centralized pytest configuration.

TOML parsing uses the stdlib ``tomllib`` (Python 3.11+). On Python 3.10 the
parse-based tests skip with an explicit reason; the plain-text guards still
run everywhere the package supports.
"""

import ast
import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: tomllib entered the stdlib in 3.11
    tomllib = None

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
SETUP_PY_PATH = REPO_ROOT / "setup.py"

requires_tomllib = pytest.mark.skipif(
    tomllib is None,
    reason="tomllib is only available in the stdlib on Python 3.11+",
)


def _load_pyproject() -> dict:
    assert tomllib is not None  # guarded by the skipif marker
    with PYPROJECT_PATH.open("rb") as fh:
        return tomllib.load(fh)


def _setup_py_kwargs() -> dict:
    """Extract the literal keyword arguments of the ``setup()`` call.

    Only literal values are captured (``long_description`` is computed from
    README.md at build time and is intentionally skipped; the pyproject
    ``readme`` field is checked separately).
    """
    tree = ast.parse(SETUP_PY_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            kwargs = {}
            for kw in node.keywords:
                try:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    pass  # non-literal value (e.g. long_description) -- skipped
            return kwargs
    raise AssertionError("setup.py does not contain a setup() call")


# ---------------------------------------------------------------------------
# Plain-text guards (run on every supported Python, no TOML parser needed)
# ---------------------------------------------------------------------------


def test_pyproject_toml_exists_with_required_sections():
    assert PYPROJECT_PATH.is_file(), "pyproject.toml is missing"
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    for section in (
        "[build-system]",
        "[project]",
        "[project.optional-dependencies]",
        "[tool.setuptools.packages.find]",
        "[tool.pytest.ini_options]",
    ):
        assert section in text, f"pyproject.toml is missing section {section}"


def test_setup_py_retained_alongside_pyproject():
    # Issue #16: pyproject.toml is added *alongside* the existing setup.py.
    assert SETUP_PY_PATH.is_file(), "setup.py must be retained"
    assert "setup(" in SETUP_PY_PATH.read_text(encoding="utf-8")


def test_pytest_config_is_centralized_no_competing_files():
    # pytest configuration must live only in pyproject.toml.
    for competing in ("pytest.ini", "tox.ini", "setup.cfg"):
        assert not (REPO_ROOT / competing).exists(), (
            f"{competing} must not exist: pytest config is centralized in pyproject.toml"
        )


# ---------------------------------------------------------------------------
# Parse-based guards (Python 3.11+)
# ---------------------------------------------------------------------------


@requires_tomllib
def test_build_system_uses_setuptools_backend():
    build = _load_pyproject()["build-system"]
    assert build["build-backend"] == "setuptools.build_meta"
    assert any(
        re.match(r"^setuptools\b", req) for req in build["requires"]
    ), "build-system.requires must include setuptools"


@requires_tomllib
def test_project_metadata_matches_setup_py():
    project = _load_pyproject()["project"]
    legacy = _setup_py_kwargs()

    assert project["name"] == legacy["name"] == "kimi-delta-attention"
    assert project["version"] == legacy["version"]
    assert project["description"] == legacy["description"]
    assert project["requires-python"] == legacy["python_requires"]
    assert project["dependencies"] == legacy["install_requires"]
    assert project["optional-dependencies"]["dev"] == legacy["extras_require"]["dev"]
    assert project["classifiers"] == legacy["classifiers"]
    assert project["urls"]["Homepage"] == legacy["url"]
    assert project["authors"] == [{"name": legacy["author"]}]


@requires_tomllib
def test_project_version_matches_package_version():
    import kda

    assert _load_pyproject()["project"]["version"] == kda.__version__


@requires_tomllib
def test_readme_field_points_at_existing_file():
    project = _load_pyproject()["project"]
    assert project["readme"] == "README.md"
    assert (REPO_ROOT / project["readme"]).is_file()


@requires_tomllib
def test_packages_find_mirrors_find_packages():
    from setuptools import find_packages

    find_config = _load_pyproject()["tool"]["setuptools"]["packages"]["find"]
    assert find_config["include"] == ["kda*"]
    # The declared include pattern must resolve to exactly what
    # setup.py's bare find_packages() discovers (cwd-independent).
    discovered = find_packages(where=str(REPO_ROOT))
    assert find_packages(where=str(REPO_ROOT), include=find_config["include"]) == discovered
    assert "kda" in discovered


@requires_tomllib
def test_pytest_config_centralized_values():
    config = _load_pyproject()["tool"]["pytest"]["ini_options"]
    assert config["testpaths"] == ["tests"]
    assert "--strict-markers" in config["addopts"]
    # testpaths must point at the real test directory.
    assert (REPO_ROOT / config["testpaths"][0]).is_dir()
