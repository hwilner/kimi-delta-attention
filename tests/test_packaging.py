"""Tests for packaging metadata consistency.

The distribution version in ``setup.py`` is parsed from ``kda/__init__.py``
so the installed metadata and ``kda.__version__`` cannot drift apart. These
tests pin that contract, along with the repository's other doc/code
consistency guarantees.
"""

import re
from pathlib import Path

import pytest

import kda

REPO_ROOT = Path(__file__).resolve().parent.parent
SETUP_PY = REPO_ROOT / "setup.py"


def _setup_declared_version() -> str:
    """Return the version ``setup.py`` hands to ``setuptools.setup()``.

    ``setup.py`` is read by executing it with ``setuptools.setup`` patched to
    a recorder, so the module's own version logic runs exactly as it would
    at build time without invoking a real build. The patch is undone before
    returning so it cannot leak into other tests.

    Returns:
        The version string the distribution would be built with.
    """
    import setuptools

    recorded: dict = {}
    original_setup = setuptools.setup
    setuptools.setup = lambda **kwargs: recorded.update(kwargs)
    try:
        namespace: dict = {"__file__": str(SETUP_PY), "__name__": "setup_under_test"}
        exec(  # noqa: S102 - test-only, executes trusted repo source
            compile(SETUP_PY.read_text(encoding="utf-8"), str(SETUP_PY), "exec"),
            namespace,
        )
    finally:
        setuptools.setup = original_setup

    assert "version" in recorded, "setup.py did not pass a version to setup()"
    return recorded["version"]


class TestPackagingMetadata:
    """Distribution metadata must agree with the package's own declarations."""

    def test_setup_version_matches_package_version(self):
        """``setup.py`` and ``kda.__version__`` declare the same version."""
        assert _setup_declared_version() == kda.__version__

    def test_setup_version_is_not_hardcoded(self):
        """The version is derived from the package, not a duplicated literal.

        A hardcoded literal is how the version drifted from 0.1.0 to 0.2.0
        without either file noticing.
        """
        source = SETUP_PY.read_text(encoding="utf-8")
        literal = re.search(r'version\s*=\s*["\'][^"\']+["\']', source)
        assert literal is None, (
            "setup.py hardcodes a version literal; it should read "
            "kda.__version__ via _read_version()"
        )

    def test_version_is_valid_pep440(self):
        """The declared version parses as a PEP 440 public version."""
        from packaging.version import Version

        version = Version(kda.__version__)
        assert not version.is_prerelease
        assert version.release


class TestDocumentationMatchesRepository:
    """Documented files must exist in the tree."""

    @pytest.mark.parametrize(
        "relative_path",
        ["examples/basic_usage.py", "examples/benchmark.py"],
    )
    def test_documented_example_exists(self, relative_path):
        """Scripts advertised in the README and METHODS.md are present."""
        assert (REPO_ROOT / relative_path).is_file(), (
            f"{relative_path} is documented but does not exist"
        )

    def test_readme_tree_lists_no_missing_files(self):
        """Every file path in the README's structure block exists.

        The block is a nested tree whose first line is the repository root
        itself, so the root prefix is dropped and the directory stack is
        rebuilt from indentation. Entries carry trailing ``#`` comments,
        which are stripped. Only fenced blocks are scanned, since matching
        the whole file would also pick up unrelated ``*.py`` text from URLs
        and prose.
        """
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        structure_blocks = [
            block for block in re.findall(r"```\n(.*?)```", readme, re.DOTALL)
            if "├" in block or "└" in block
        ]
        assert structure_blocks, "README has no fenced structure block to check"

        referenced: set = set()
        for block in structure_blocks:
            # Stack of (name column, path prefix); an entry at column <= the
            # top's column is a sibling, so deeper prefixes are popped first.
            stack: list = []
            for line in block.splitlines()[1:]:  # Skip the repository root line.
                if "─" not in line and "│" not in line:
                    continue
                name = re.sub(r"^[^A-Za-z0-9_.-]*", "", line).split("#")[0].strip()
                if not name:
                    continue
                indent = len(line) - len(
                    re.sub(r"^[^A-Za-z0-9_.-]*", "", line)
                )
                while stack and stack[-1][0] >= indent:
                    stack.pop()
                if name.endswith("/"):
                    prefix = f"{stack[-1][1]}/{name[:-1]}" if stack else name[:-1]
                    stack.append((indent, prefix))
                elif name.endswith(".py"):
                    prefix = f"{stack[-1][1]}/{name}" if stack else name
                    referenced.add(prefix)

        assert referenced, "no Python file paths found in README structure block"
        for candidate in sorted(referenced):
            assert (REPO_ROOT / candidate).exists(), (
                f"README references {candidate}, which is not in the repository"
            )
