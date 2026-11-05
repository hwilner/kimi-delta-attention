"""Cross-experiment consistency checks for the Phase 4 results report (#34).

``docs/PHASE4_RESULTS.md`` is generated from the pre-registered #28/#34/#38
default configs, while the follow-up sweeps — calibration robustness (#50,
:mod:`kda.experiments.calibration_robustness`), the #33 hyperparameter
ablation (#51, :mod:`kda.experiments.precision_gated_ablation`), routing
robustness (#52, :mod:`kda.experiments.routing_robustness`), and the #36
hyperparameter ablation (#54,
:mod:`kda.experiments.rls_lite_ablation`) — each contain a
*default cell* that is documented and regression-tested to reproduce the
corresponding base pipeline bit-for-bit. That means the same numbers appear
in two places, and they must agree: the report's capacity table must equal
the ablations' default cells, the report's calibration AUROC table must equal
the robustness sweep's ``(8, 0.9)`` cell, and the report's routing section
must equal the routing sweep's default cell.

The report also embeds the two ablation sweeps *in full* (the robustness
section of the report): every grid cell of both ablations is rendered as a
table row. Those
tables are cross-checked cell-by-cell against a fresh run of the checked-in
full-grid ablation configs (the grids are small enough that the
recomputation takes seconds), so a hand edit to any ablation row — not just
the default cell — is caught.

This module automates those cross-checks: it parses the checked-in report's
markdown tables, recomputes the overlapping numbers from the checked-in
configs, and compares the *formatted strings* — the exact text a reader sees
in the report, in the generator's fixed three-decimal style. Every
comparison is recorded as an explicit check entry with ``expected``,
``actual``, and ``match``; mismatches are reported, never hidden or
corrected (the honest-negatives discipline applied to the report itself).

Scope: only the overlapping numbers are checked — the #51 ablation runs KDA
(reference) and precision-gated KDA, the #54 ablation runs KDA (reference)
and KDA-RLS-lite, and the robustness sweeps run KDA,
KDA-RLS-lite, and precision-gated KDA, so GDN-2/EDA rows and non-default
regime cells are out of scope here (the per-sweep regression tests already
cover their own pipelines). The KDA reference rows of the two ablations are
two renderings of the same pipeline as well; they are cross-checked via the
#51 expectations, so the #54 side contributes its RLS-lite default cell. The
ablation reference *lines* in the report's robustness section are prose
(under the tables, outside the pipe tables) and are intentionally not parsed;
the same numbers are already cross-checked in the capacity table.
"""

from __future__ import annotations

__all__ = [
    "parse_pipe_tables",
    "format_report_float",
    "format_report_ci",
    "default_calibration_robustness_config",
    "default_routing_robustness_config",
    "precision_gated_ablation_heading",
    "rls_lite_ablation_heading",
    "precision_gated_ablation_rows",
    "rls_lite_ablation_rows",
    "capacity_expectations",
    "rls_lite_capacity_expectations",
    "calibration_expectations",
    "routing_expectations",
    "check_report_consistency",
    "PRECISION_GATED_ABLATION_HEADER",
    "RLS_LITE_ABLATION_HEADER",
]

from .calibration_robustness import CalibrationRobustnessConfig
from .routing_robustness import RoutingRobustnessConfig

# Headings of the checked-in report sections that carry cross-checked
# tables (see docs/PHASE4_RESULTS.md; the capacity sweep runs at the fixed
# collision cosine 0.900, which is the ablation's operating-point cosine).
CAPACITY_HEADING = "### Capacity sweep (collision cosine 0.900)"
FAILURE_HEADING = "### Failure rates"
AUROC_HEADING = "### AUROC of each uncertainty signal"
ENDPOINTS_HEADING = "### Endpoints"
ROUTING_GAINS_HEADING = "### Best / worst adaptive gain over matched fixed-ratio routing"


def parse_pipe_tables(markdown: str) -> dict[str, list[dict]]:
    """Parse GitHub pipe tables out of a markdown document.

    Returns a mapping from the nearest preceding heading line (``## ...`` or
    ``### ...``) to the list of tables under it, in document order. Each
    table is ``{"header": [...], "rows": [[...], ...]}`` with cell text
    stripped. Lines inside fenced code blocks (the report embeds config JSON)
    are ignored so JSON content is never mistaken for a table.
    """

    tables: dict[str, list[dict]] = {}
    heading = ""
    in_fence = False
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            index += 1
            continue
        if in_fence:
            index += 1
            continue
        if stripped.startswith("#"):
            heading = stripped
            index += 1
            continue
        if stripped.startswith("|"):
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index].strip())
                index += 1
            cells = [
                [cell.strip() for cell in row.strip("|").split("|")]
                for row in block
            ]
            header, body = cells[0], [
                row for row in cells[1:]
                if not all(set(cell) <= set("-: ") for cell in row)
            ]
            tables.setdefault(heading, []).append(
                {"header": header, "rows": body}
            )
            continue
        index += 1
    return tables


def format_report_float(value: float) -> str:
    """Fixed three-decimal formatting, mirroring the report generator."""

    return f"{value:.3f}"


def format_report_ci(entry: dict) -> str:
    """Format a ``{value, ci}`` metric as ``0.750 [0.688, 0.806]``."""

    low, high = entry["ci"]
    return (
        f"{format_report_float(entry['value'])} "
        f"[{format_report_float(low)}, {format_report_float(high)}]"
    )


def default_calibration_robustness_config() -> CalibrationRobustnessConfig:
    """Calibration robustness config restricted to the #28 default cell."""

    return CalibrationRobustnessConfig(
        num_pairs_grid=(8,),
        collision_cosine_grid=(0.9,),
    )


def default_routing_robustness_config() -> RoutingRobustnessConfig:
    """Routing robustness config restricted to the #38 default cell."""

    return RoutingRobustnessConfig(
        num_pairs_grid=(8,),
        collision_cosine_grid=(0.9,),
    )


def capacity_expectations(ablation_report: dict) -> dict[tuple[str, int], str]:
    """Expected capacity-table cells from the ablation default cell.

    Maps ``(variant, num_pairs)`` to the formatted ``value [ci]`` string the
    report's capacity sweep must show at collision cosine 0.900: the KDA
    reference row and the precision-gated default cell, per operating point.
    """

    expectations: dict[tuple[str, int], str] = {}
    for point in ablation_report["operating_points"]:
        num_pairs = int(point["num_pairs"])
        for variant, section in point["reference"].items():
            expectations[(variant, num_pairs)] = format_report_ci(
                section["recall_accuracy"]
            )
        default_cells = [c for c in point["cells"] if c["is_default"]]
        if len(default_cells) != 1:
            raise ValueError(
                "ablation report must contain exactly one default cell per "
                f"operating point, found {len(default_cells)} at "
                f"num_pairs={num_pairs}"
            )
        expectations[("precision-gated-kda", num_pairs)] = format_report_ci(
            default_cells[0]["recall_accuracy"]
        )
    return expectations


def rls_lite_capacity_expectations(
    rls_lite_report: dict,
) -> dict[tuple[str, int], str]:
    """Expected capacity-table cells from the RLS-lite ablation default cell.

    Maps ``("kda-rls-lite", num_pairs)`` to the formatted ``value [ci]``
    string the report's capacity sweep must show at collision cosine 0.900,
    per operating point of the #54 ablation. The ablation's KDA reference
    rows are not included: they reproduce the same pipeline as the #51
    reference rows, which :func:`capacity_expectations` already checks.
    """

    expectations: dict[tuple[str, int], str] = {}
    for point in rls_lite_report["operating_points"]:
        num_pairs = int(point["num_pairs"])
        default_cells = [c for c in point["cells"] if c["is_default"]]
        if len(default_cells) != 1:
            raise ValueError(
                "RLS-lite ablation report must contain exactly one default "
                f"cell per operating point, found {len(default_cells)} at "
                f"num_pairs={num_pairs}"
            )
        expectations[("kda-rls-lite", num_pairs)] = format_report_ci(
            default_cells[0]["recall_accuracy"]
        )
    return expectations


# --- ablation robustness-section tables (full-grid cross-checks) -----------

# Header rows of the report's ablation tables; the row builders below and
# the report generator both render exactly these columns, in this order.
PRECISION_GATED_ABLATION_HEADER = [
    "lambda",
    "forgetting",
    "process_noise",
    "default",
    "recall accuracy",
    "mean recall cosine",
]
RLS_LITE_ABLATION_HEADER = [
    "evidence_decay",
    "beta_min",
    "beta_max",
    "default",
    "recall accuracy",
    "mean recall cosine",
]


def precision_gated_ablation_heading(num_pairs: int, collision_cosine: float) -> str:
    """Heading of the report's #51 ablation table at one operating point."""

    return (
        f"### Precision-gated KDA ablation (pairs={num_pairs}, "
        f"collision cosine {format_report_float(collision_cosine)})"
    )


def rls_lite_ablation_heading(num_pairs: int, collision_cosine: float) -> str:
    """Heading of the report's #54 ablation table at one operating point."""

    return (
        f"### KDA-RLS-lite ablation (pairs={num_pairs}, "
        f"collision cosine {format_report_float(collision_cosine)})"
    )


def precision_gated_ablation_rows(report: dict) -> dict[str, list[list[str]]]:
    """Expected #51 ablation-table rows, keyed by table heading.

    Each operating point maps to its grid cells rendered as formatted row
    strings in the report's fixed three-decimal style, in the config's
    deterministic Cartesian cell order; the default cell carries the ``*``
    marker. The generator and this checker iterate the same
    ``point["cells"]`` order, so rows are compared positionally.
    """

    tables: dict[str, list[list[str]]] = {}
    for point in report["operating_points"]:
        heading = precision_gated_ablation_heading(
            int(point["num_pairs"]), float(point["collision_cosine"])
        )
        tables[heading] = [
            [
                format_report_float(cell["lambda_"]),
                format_report_float(cell["forgetting"]),
                format_report_float(cell["process_noise"]),
                "*" if cell["is_default"] else "",
                format_report_ci(cell["recall_accuracy"]),
                format_report_float(cell["mean_recall_cosine"]),
            ]
            for cell in point["cells"]
        ]
    return tables


def rls_lite_ablation_rows(report: dict) -> dict[str, list[list[str]]]:
    """Expected #54 ablation-table rows, keyed by table heading.

    Same conventions as :func:`precision_gated_ablation_rows`.
    """

    tables: dict[str, list[list[str]]] = {}
    for point in report["operating_points"]:
        heading = rls_lite_ablation_heading(
            int(point["num_pairs"]), float(point["collision_cosine"])
        )
        tables[heading] = [
            [
                format_report_float(cell["evidence_decay"]),
                format_report_float(cell["beta_min"]),
                format_report_float(cell["beta_max"]),
                "*" if cell["is_default"] else "",
                format_report_ci(cell["recall_accuracy"]),
                format_report_float(cell["mean_recall_cosine"]),
            ]
            for cell in point["cells"]
        ]
    return tables


def calibration_expectations(
    robustness_report: dict,
) -> dict[str, dict[str, dict[str, str]]]:
    """Expected calibration-table cells from the robustness default cell.

    Returns ``{"failure": {(variant, column): text}, "auroc": {(signal,
    variant): text}}`` where ``text`` is the formatted string (or the literal
    ``"unavailable"`` for signals a variant does not expose) that the
    report's calibration tables must show at the #28 default operating point.
    """

    cells = robustness_report["cells"]
    if len(cells) != 1:
        raise ValueError(
            "expected a single-cell robustness report (the #28 default "
            f"cell), found {len(cells)} cells"
        )
    cell = cells[0]
    failure: dict[tuple[str, str], str] = {}
    auroc: dict[tuple[str, str], str] = {}
    for variant, section in cell["variants"].items():
        failure[(variant, "failure rate")] = format_report_ci(
            section["failure_rate"]
        )
        failure[(variant, "mean recall cosine")] = format_report_float(
            section["mean_recall_cosine"]
        )
        for signal, entry in section["signals"].items():
            auroc[(signal, variant)] = (
                format_report_ci(entry["auroc"])
                if entry["available"]
                else "unavailable"
            )
    return {"failure": failure, "auroc": auroc}


def routing_expectations(
    routing_report: dict,
) -> dict[str, dict[tuple[str, str], str]]:
    """Expected routing-section cells from the routing default cell.

    Returns ``{"endpoints": {(variant, column): text}, "gains": {(signal,
    variant): text}}``; gain cells reproduce the report's
    ``best_gain @ best_rate / worst_gain @ worst_rate`` summary over the
    pre-registered threshold grid (flat and negative operating points
    included, exactly as the report renders them).
    """

    cells = routing_report["cells"]
    if len(cells) != 1:
        raise ValueError(
            "expected a single-cell routing robustness report (the #38 "
            f"default cell), found {len(cells)} cells"
        )
    cell = cells[0]
    endpoints: dict[tuple[str, str], str] = {}
    gains: dict[tuple[str, str], str] = {}
    for variant, section in cell["variants"].items():
        endpoints[(variant, "compressed accuracy")] = format_report_float(
            section["compressed_accuracy"]
        )
        endpoints[(variant, "exact accuracy")] = format_report_float(
            section["exact_accuracy"]
        )
        endpoints[(variant, "headroom")] = format_report_float(
            section["exact_headroom"]
        )
        for signal, entry in section["signals"].items():
            if not entry["available"]:
                gains[(signal, variant)] = "unavailable"
                continue
            points = entry["operating_points"]
            best = max(points, key=lambda p: p["gain_vs_fixed_ratio"])
            worst = min(points, key=lambda p: p["gain_vs_fixed_ratio"])
            gains[(signal, variant)] = (
                f"{format_report_float(best['gain_vs_fixed_ratio'])} @ "
                f"{format_report_float(best['fallback_rate'])} / "
                f"{format_report_float(worst['gain_vs_fixed_ratio'])} @ "
                f"{format_report_float(worst['fallback_rate'])}"
            )
    return {"endpoints": endpoints, "gains": gains}


def _table_by_heading(tables: dict[str, list[dict]], heading: str) -> dict | None:
    """The first pipe table under ``heading``, or ``None`` if absent."""

    found = tables.get(heading)
    return found[0] if found else None


def _compare_cells(
    checks: list[dict],
    *,
    table_name: str,
    table: dict | None,
    expectations: dict[tuple[str, str], str],
) -> None:
    """Record one check per expected cell against the parsed report table.

    The first column of each row is the row label; the header row names the
    remaining columns. A missing table or a missing row/column is recorded as
    a failed check with an explanatory note rather than raised, so one
    structural problem does not hide the remaining comparisons.
    """

    for (row_label, column_label), expected in sorted(expectations.items()):
        check = {
            "table": table_name,
            "row": row_label,
            "column": column_label,
            "expected": expected,
            "actual": None,
            "match": False,
        }
        if table is None:
            check["note"] = f"table '{table_name}' not found in the report"
        else:
            header = table["header"]
            rows = {row[0]: row for row in table["rows"] if row}
            if column_label not in header:
                check["note"] = f"column '{column_label}' not found"
            elif row_label not in rows:
                check["note"] = f"row '{row_label}' not found"
            else:
                actual = rows[row_label][header.index(column_label)]
                check["actual"] = actual
                check["match"] = actual == expected
        checks.append(check)


def _compare_rows(
    checks: list[dict],
    *,
    table_name: str,
    table: dict | None,
    expected_header: list[str],
    expected_rows: list[list[str]],
) -> None:
    """Record positional row/cell checks of a whole table against expectations.

    Used for the ablation tables, whose rows are hyperparameter cells in a
    deterministic Cartesian order rather than label-keyed rows. The header is
    compared as a single ``(header)`` check; data rows are compared
    positionally, one check per cell, with ``row`` = ``row {i}`` (0-based
    data-row index) and ``column`` = the header name. A missing table, a
    header mismatch, or a row-count mismatch is recorded as failed checks
    with explanatory notes rather than raised, so one structural problem does
    not hide the remaining comparisons.
    """

    if table is None:
        for i in range(len(expected_rows)):
            checks.append(
                {
                    "table": table_name,
                    "row": f"row {i}",
                    "column": "(all)",
                    "expected": expected_rows[i],
                    "actual": None,
                    "match": False,
                    "note": f"table '{table_name}' not found in the report",
                }
            )
        return
    checks.append(
        {
            "table": table_name,
            "row": "(header)",
            "column": "(header)",
            "expected": expected_header,
            "actual": table["header"],
            "match": table["header"] == expected_header,
        }
    )
    if len(table["rows"]) != len(expected_rows):
        checks.append(
            {
                "table": table_name,
                "row": "(row count)",
                "column": "(row count)",
                "expected": len(expected_rows),
                "actual": len(table["rows"]),
                "match": False,
                "note": (
                    f"expected {len(expected_rows)} data rows, found "
                    f"{len(table['rows'])}"
                ),
            }
        )
    for i, expected_row in enumerate(expected_rows):
        if i >= len(table["rows"]):
            checks.append(
                {
                    "table": table_name,
                    "row": f"row {i}",
                    "column": "(all)",
                    "expected": expected_row,
                    "actual": None,
                    "match": False,
                    "note": "row missing from the report table",
                }
            )
            continue
        actual_row = table["rows"][i]
        for j, expected_cell in enumerate(expected_row):
            column = (
                expected_header[j] if j < len(expected_header) else f"col {j}"
            )
            actual_cell = actual_row[j] if j < len(actual_row) else None
            checks.append(
                {
                    "table": table_name,
                    "row": f"row {i}",
                    "column": column,
                    "expected": expected_cell,
                    "actual": actual_cell,
                    "match": actual_cell == expected_cell,
                }
            )


def check_report_consistency(
    markdown: str,
    *,
    ablation_report: dict,
    calibration_robustness_report: dict,
    routing_robustness_report: dict,
    rls_lite_ablation_report: dict,
) -> dict:
    """Cross-check the report's tables against recomputed sweep results.

    Args:
        markdown: Text of the checked-in ``docs/PHASE4_RESULTS.md``.
        ablation_report: Result of
            :func:`kda.experiments.precision_gated_ablation.run_precision_gated_ablation`
            on the checked-in *full-grid* #51 default config
            (:class:`~kda.experiments.PrecisionGatedAblationConfig`). The
            default cells feed the capacity-table expectations; the full
            grids feed the ablation-table row checks.
        calibration_robustness_report: Result of
            :func:`kda.experiments.calibration_robustness.run_calibration_robustness`
            on :func:`default_calibration_robustness_config`.
        routing_robustness_report: Result of
            :func:`kda.experiments.routing_robustness.run_routing_robustness`
            on :func:`default_routing_robustness_config`.
        rls_lite_ablation_report: Result of
            :func:`kda.experiments.rls_lite_ablation.run_rls_lite_ablation`
            on the checked-in *full-grid* #54 default config
            (:class:`~kda.experiments.RLSLiteAblationConfig`).

    Returns:
        A JSON-serializable dict with one entry per compared cell
        (``expected``/``actual``/``match``) and an ``all_match`` summary.
        Mismatches are listed, never hidden.
    """

    tables = parse_pipe_tables(markdown)
    checks: list[dict] = []

    capacity = capacity_expectations(ablation_report)
    capacity.update(rls_lite_capacity_expectations(rls_lite_ablation_report))
    _compare_cells(
        checks,
        table_name=CAPACITY_HEADING,
        table=_table_by_heading(tables, CAPACITY_HEADING),
        expectations={
            (variant, f"pairs={num_pairs}"): text
            for (variant, num_pairs), text in capacity.items()
        },
    )

    calibration = calibration_expectations(calibration_robustness_report)
    _compare_cells(
        checks,
        table_name=FAILURE_HEADING,
        table=_table_by_heading(tables, FAILURE_HEADING),
        expectations=calibration["failure"],
    )
    _compare_cells(
        checks,
        table_name=AUROC_HEADING,
        table=_table_by_heading(tables, AUROC_HEADING),
        expectations=calibration["auroc"],
    )

    routing = routing_expectations(routing_robustness_report)
    _compare_cells(
        checks,
        table_name=ENDPOINTS_HEADING,
        table=_table_by_heading(tables, ENDPOINTS_HEADING),
        expectations=routing["endpoints"],
    )
    _compare_cells(
        checks,
        table_name=ROUTING_GAINS_HEADING,
        table=_table_by_heading(tables, ROUTING_GAINS_HEADING),
        expectations=routing["gains"],
    )

    for heading, expected_rows in precision_gated_ablation_rows(
        ablation_report
    ).items():
        _compare_rows(
            checks,
            table_name=heading,
            table=_table_by_heading(tables, heading),
            expected_header=PRECISION_GATED_ABLATION_HEADER,
            expected_rows=expected_rows,
        )
    for heading, expected_rows in rls_lite_ablation_rows(
        rls_lite_ablation_report
    ).items():
        _compare_rows(
            checks,
            table_name=heading,
            table=_table_by_heading(tables, heading),
            expected_header=RLS_LITE_ABLATION_HEADER,
            expected_rows=expected_rows,
        )

    mismatches = [c for c in checks if not c["match"]]
    return {
        "experiment": "phase4-report-consistency",
        "issue": 34,
        "description": (
            "Cross-experiment consistency: the checked-in #34 report tables "
            "must equal the recomputed #50/#51/#52/#54 extension sweeps. The "
            "sweeps' default cells reproduce the #28/#34/#36/#38 pipelines "
            "bit-for-bit (capacity/calibration/routing cross-checks), and "
            "the report's ablation tables embed the #51/#54 full grids, "
            "which are cross-checked cell-by-cell against a fresh full-grid "
            "run. Compared as formatted strings in the report's fixed "
            "three-decimal style; mismatches are reported, never corrected."
        ),
        "num_checks": len(checks),
        "num_mismatches": len(mismatches),
        "all_match": not mismatches,
        "checks": checks,
    }
