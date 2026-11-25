"""Consistency checks for the public API reference.

The API reference is documentation, but it should not drift from the actual
package exports, constructor signatures, tensor shapes, or variant registry.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import kda
from kda import KDACore, KDAChunkwise, KimiDeltaAttention, ShortConv1d
from kda import variants


ROOT = Path(__file__).resolve().parents[1]
API_REFERENCE = ROOT / "docs" / "API_REFERENCE.md"


def _doc_text() -> str:
    return API_REFERENCE.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    marker = f"## `{heading}`"
    assert marker in text, f"missing API reference section for {heading}"
    return text.split(marker, maxsplit=1)[1].split("\n## ", maxsplit=1)[0]


def test_api_reference_documents_all_top_level_exports():
    text = _doc_text()

    for export_name in kda.__all__:
        assert export_name in text

    assert "`__version__`" not in text  # metadata, not part of the behavior API


def test_core_class_constructor_arguments_are_documented():
    text = _doc_text()

    for cls in (KDACore, KDAChunkwise, KimiDeltaAttention, ShortConv1d):
        section = _section(text, cls.__name__)
        parameters = list(inspect.signature(cls.__init__).parameters.values())[1:]
        for parameter in parameters:
            assert f"`{parameter.name}`" in section


def test_core_shapes_and_state_equation_are_documented():
    text = _doc_text()

    expected_shapes = [
        "(B, L, d_h)",
        "(B, L, d_v)",
        "(B, L, 1)",
        "(B, d_h, d_v)",
        "(B, L, D)",
        "(B, H, d_h, d_v)",
    ]
    for shape in expected_shapes:
        assert shape in text

    assert "S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T" in text
    assert "o_t = S_t^T q_t" in text


def test_variant_registry_api_and_registered_names_are_documented():
    text = _doc_text()

    for name in (
        "DeltaMemoryOutput",
        "DeltaMemoryVariant",
        "VariantNotFoundError",
        "register_variant",
        "available_variants",
        "create_variant",
    ):
        assert name in text

    for registered_name in variants.available_variants():
        assert f"`{registered_name}`" in text


def test_variant_constructor_arguments_are_documented():
    text = _doc_text()

    for name in variants.available_variants():
        variant_cls = type(variants.create_variant(name, head_dim=4, value_dim=3))
        parameters = list(inspect.signature(variant_cls.__init__).parameters.values())[1:]
        for parameter in parameters:
            assert parameter.name in text, (
                f"{variant_cls.__name__} parameter {parameter.name} is not documented"
            )


def test_missing_utils_module_is_explicitly_scoped():
    text = _doc_text()

    assert not (ROOT / "kda" / "utils.py").exists()
    assert "no `kda/utils.py` module" in text
    assert "Research module map" in text


def test_api_reference_includes_minimal_examples():
    text = _doc_text()

    assert text.count("### Minimal example") >= 3
    assert "### Minimal examples" in text
    assert "### Minimal variant example" in text
    assert text.count("```python") >= 8
