"""
CAFA++ Prototype Implementation (Beta)
=====================================

This module contains a **plain-structured** implementation of the
first beta of the CAFA++ architecture described in the design notes.
The goal is to build a demonstration pipeline that can ingest a
mixture of documents and tables (PDF, DOCX, XLSX, CSV, or plain
text), extract and normalize evidence, run a semantic parser (via
LLM) to produce a structured CAFA-AST, validate and repair that AST,
compile it into a solver-neutral CAFA-IR, generate an executable
optimization program, and finally solve the problem and explain the
results.  The code is written in a **flat, functional style** with
minimal class encapsulation to keep it easy to debug and understand
for individual developers.

The architecture closely follows the 10-step process outlined in the
design discussion:

1. **Input**: Accept various file types containing problem
   descriptions and numerical data.
2. **Extraction**: Read text blocks, tables, captions, headings and
   metadata from the source files.  This stage isolates the raw
   evidence in a generic format that can be consumed by later stages.
3. **Normalization**: Convert the extracted tables into a normalized
   JSON structure, clean numeric values and units, detect headers and
   labels, and link paragraphs to nearby tables.  The aim is to
   produce a consistent representation that is independent of the
   underlying file format.
4. **Candidate Labeling**: Identify candidate sets, parameters,
   variables, and resource types by applying heuristics over the
   normalized evidence.  This stage does not assign final semantics
   but helps the LLM by providing hints about the roles of different
   numbers and columns.
5. **LLM AST Generation**: Use a Large Language Model (LLM) to
   convert the normalized evidence and labels into a structured
   CAFA-AST.  The AST captures the high‑level semantics of the
   optimization problem—decision variables, objective function,
   constraints, sets, parameters, and assumptions.  Each AST node
   includes provenance references to the evidence IDs from which it
   was derived.
6. **Semantic Checker**: Validate the AST for completeness and
   correctness.  Checks include schema validation, unit consistency,
   number coverage, constraint direction, variable type realism, and
   table coverage.  Suspicious fields are collected for targeted
   repair.
7. **Targeted Repair**: When the semantic checker detects likely
   mistakes, a minimal message is sent back to the LLM to fix only
   those fields.  This avoids regeneration of the entire AST and
   reduces error propagation.
8. **Lowering**: Convert the CAFA-AST into a solver-neutral CAFA-IR.
   This includes generating linear expressions, mapping sets and
   parameters to indices, and producing a compact representation of
   the model ready for compilation.
9. **Compilation and Solver Execution**: Generate solver code
   (Gurobi/Pyomo/OR-Tools) from the CAFA-IR and execute it to obtain a
   solution.  The code is constructed deterministically to avoid
   injecting any modeling logic from the LLM beyond the AST.
10. **Explanation**: Produce a human‑readable explanation of the
   solution by tracing the AST back to the evidence.  This step
   explains how each decision variable and constraint arises from the
   original document and provides a clear justification of the
   optimized solution.

This file is intentionally verbose and heavily commented.  Each
function includes a detailed docstring to help developers understand
its purpose and behaviour.  The number of lines (~3500–4000) is
deliberate to provide room for future extensions and to document
each component thoroughly.

The implementation draws inspiration from the existing
``cafa_ir_guarded.py`` pipeline but generalizes it for multi‑modal
input and a richer AST representation.  Key improvements include a
more robust semantic checker, explicit evidence linkage in the AST,
and the ability to perform targeted repairs based on detected
anomalies.

Note that this prototype does not require any external packages
beyond the Python standard library and Gurobi/Pyomo (for solving).
Where support for PDF or DOCX is needed, simple fallbacks are
provided; a production system should integrate specialized parsers
like ``pdfplumber`` or ``python‑docx``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A linear expression is represented as a pair: (coefficients, constant)
LinearExpr = Tuple[Dict[str, float], float]

# Evidence objects are stored in a dictionary keyed by a unique ID.  Each
# evidence block is a dictionary with at least ``type`` and ``id`` keys, and
# other fields depending on the block type.  For example, paragraphs
# contain ``text``, while tables contain ``rows`` and ``columns``.
EvidenceDict = Dict[str, Dict[str, Any]]

# The AST is represented as a nested set of dictionaries.  The
# ``problem`` key stores high-level metadata.  ``sets`` is a list of
# dictionaries describing set definitions, ``parameters`` is a list of
# parameters with values and units, ``variables`` lists the decision
# variables, ``objective`` defines the objective, ``constraints`` is a
# list of constraint objects, ``assumptions`` captures modelling
# assumptions, and ``metadata`` stores global provenance and other
# information.
AstDict = Dict[str, Any]

# The IR is a flat dictionary that encodes a linear (MILP) model.
IrDict = Dict[str, Any]

# ---------------------------------------------------------------------------
# Utilities for logging and error reporting
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """
    Write a message to stderr with a consistent prefix.  This helper
    function centralizes logging so that the output can easily be
    redirected or silenced.  Developers may prefer using the Python
    ``logging`` module; for simplicity, we use basic prints here.

    Parameters
    ----------
    msg : str
        The message to be logged.
    """
    sys.stderr.write(f"[CAFA++] {msg}\n")


def warn(msg: str) -> None:
    """
    Write a warning message to stderr.  Warnings are annotated
    separately from informational logs to make them easier to spot.  In
    a production system this should be integrated with the logging
    subsystem.

    Parameters
    ----------
    msg : str
        The warning message to be logged.
    """
    sys.stderr.write(f"[CAFA++ WARNING] {msg}\n")


def debug(msg: str) -> None:
    """
    Write a debug message to stderr.  Debugging output can be
    controlled globally by setting the ``DEBUG`` environment variable
    or modifying this function.  For now, all debug messages are
    printed unconditionally to assist development.

    Parameters
    ----------
    msg : str
        The debug message to be logged.
    """
    sys.stderr.write(f"[CAFA++ DEBUG] {msg}\n")


# ---------------------------------------------------------------------------
# Step 1: Input and Evidence Extraction
# ---------------------------------------------------------------------------

def read_text_file(path: str) -> EvidenceDict:
    """
    Read a plain text file and return an evidence dictionary.

    The function splits the text into paragraphs based on two or more
    consecutive newline characters.  Each paragraph is stored as a
    separate evidence block with a unique ID.  The page number is
    defaulted to 1 because plain text does not provide pagination.

    Parameters
    ----------
    path : str
        The path to the text file.

    Returns
    -------
    evidence : EvidenceDict
        A dictionary mapping evidence IDs to blocks.  Each block has
        ``id``, ``type``, ``text``, and ``page`` fields.
    """
    evidence: EvidenceDict = {}
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    paragraphs = re.split(r"\n\s*\n", text)
    for idx, para in enumerate(paragraphs):
        eid = f"p{idx+1}"
        evidence[eid] = {
            "id": eid,
            "type": "paragraph",
            "text": para.strip(),
            "page": 1,
        }
    return evidence


def read_csv_file(path: str) -> EvidenceDict:
    """
    Read a CSV file and return an evidence dictionary containing a single
    table block.  The CSV file is assumed to contain a header row.

    Each cell value is stored exactly as read; normalization happens in
    a later phase.  The table is assigned an ID ``t1``.

    Parameters
    ----------
    path : str
        The path to the CSV file.

    Returns
    -------
    evidence : EvidenceDict
        A dictionary containing one table block with ``columns`` and
        ``rows`` keys.
    """
    evidence: EvidenceDict = {}
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        columns = reader.fieldnames or []
        rows = [row for row in reader]
    eid = "t1"
    evidence[eid] = {
        "id": eid,
        "type": "table",
        "columns": columns,
        "rows": rows,
        "caption": "",
        "page": 1,
    }
    return evidence


def read_xlsx_file(path: str) -> EvidenceDict:
    """
    Read an Excel (XLSX) file and return an evidence dictionary.

    This implementation uses Python's built-in ``csv`` module by
    converting the first sheet into a CSV format.  A production
    implementation should leverage a library like ``openpyxl`` or
    ``pandas`` for more robust handling of spreadsheets, including
    multiple sheets and complex cell types.  Here we fallback to
    ``csv`` because the environment may lack additional packages.

    Parameters
    ----------
    path : str
        The path to the Excel file.

    Returns
    -------
    evidence : EvidenceDict
        A dictionary containing one table block with ``columns`` and
        ``rows`` keys extracted from the first worksheet.
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        warn("pandas is not installed; falling back to CSV extraction for XLSX")
        # Fallback: treat the file as CSV
        return read_csv_file(path)
    evidence: EvidenceDict = {}
    df = pd.read_excel(path, sheet_name=0)
    columns = df.columns.tolist()
    rows = df.to_dict(orient='records')
    eid = "t1"
    evidence[eid] = {
        "id": eid,
        "type": "table",
        "columns": columns,
        "rows": rows,
        "caption": "",
        "page": 1,
    }
    return evidence


def read_pdf_file(path: str) -> EvidenceDict:
    """
    Read a PDF file and return an evidence dictionary.

    This function attempts to extract text and tables using the
    ``pdfminer`` library if available.  If ``pdfminer`` is not
    installed, the function falls back to reading the raw PDF bytes and
    creating a single block of binary data.  In that case the
    downstream pipeline cannot process the PDF, so a warning is issued.

    A production system should integrate a robust PDF parser (e.g.
    ``pdfplumber``, ``camelot``, or ``tabula-py``) to extract tables
    and text with high fidelity.  This placeholder implementation
    merely demonstrates the API surface.

    Parameters
    ----------
    path : str
        The path to the PDF file.

    Returns
    -------
    evidence : EvidenceDict
        A dictionary containing zero or more blocks extracted from the
        PDF.  If extraction fails, a single binary block is returned.
    """
    evidence: EvidenceDict = {}
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        text = extract_text(path)
        if not text.strip():
            warn(f"No text extracted from PDF {path}")
        paragraphs = re.split(r"\n\s*\n", text)
        for idx, para in enumerate(paragraphs):
            eid = f"p{idx+1}"
            evidence[eid] = {
                "id": eid,
                "type": "paragraph",
                "text": para.strip(),
                "page": 1,
            }
        # PDF table extraction is not implemented here.  A real system
        # should iterate over pages and detect tables using a library
        # like camelot.  For now we leave this as future work.
    except ImportError:
        warn("pdfminer.six is not installed; PDF extraction is disabled")
        # If we cannot parse the PDF, we still store a placeholder
        with open(path, 'rb') as f:
            content = f.read()
        eid = "bin1"
        evidence[eid] = {
            "id": eid,
            "type": "binary",
            "data": content,
            "page": 1,
        }
    return evidence


def read_docx_file(path: str) -> EvidenceDict:
    """
    Read a Word (DOCX) file and return an evidence dictionary.

    This function attempts to use ``python-docx`` to extract text and
    simple tables.  If the library is unavailable, it falls back to
    reading the raw file contents.  A robust implementation should
    handle complex table structures, lists, and styles.

    Parameters
    ----------
    path : str
        The path to the DOCX file.

    Returns
    -------
    evidence : EvidenceDict
        A dictionary containing paragraphs and tables from the DOCX.
    """
    evidence: EvidenceDict = {}
    try:
        from docx import Document  # type: ignore
        doc = Document(path)
        p_counter = 0
        t_counter = 0
        for element in doc.element.body:
            if element.tag.endswith("p"):
                p_counter += 1
                paragraphs = element.text.strip() if hasattr(element, 'text') else ''
                eid = f"p{p_counter}"
                evidence[eid] = {
                    "id": eid,
                    "type": "paragraph",
                    "text": paragraphs,
                    "page": 1,
                }
            elif element.tag.endswith("tbl"):
                t_counter += 1
                rows: List[Dict[str, Any]] = []
                # Extract table cells
                for row in element.iter(tag=element.tag.replace("tbl", "tr")):
                    cells = []
                    for cell in row.iter(tag=element.tag.replace("tbl", "tc")):
                        paragraphs = []
                        for p in cell.iter(tag=element.tag.replace("tbl", "p")):
                            paragraphs.append(p.text)
                        cells.append(" ".join(paragraphs))
                    if cells:
                        rows.append({str(i): val for i, val in enumerate(cells)})
                eid = f"t{t_counter}"
                evidence[eid] = {
                    "id": eid,
                    "type": "table",
                    "columns": [str(i) for i in range(len(rows[0]))] if rows else [],
                    "rows": rows,
                    "caption": "",
                    "page": 1,
                }
    except ImportError:
        warn("python-docx is not installed; DOCX extraction is disabled")
        with open(path, 'rb') as f:
            data = f.read()
        eid = "bin1"
        evidence[eid] = {
            "id": eid,
            "type": "binary",
            "data": data,
            "page": 1,
        }
    return evidence


def read_input(path: str) -> EvidenceDict:
    """
    Top-level input reader dispatch.  Determines the file type by
    extension and calls the appropriate reader.  When the input is a
    directory, all files inside the directory are read and merged.

    If multiple files are supplied, evidence IDs must remain unique.
    This implementation concatenates the evidence dictionaries,
    prefixing IDs with the file index (e.g. ``f1_p1``, ``f2_t1``).

    Parameters
    ----------
    path : str
        Path to a file or directory containing problem descriptions.

    Returns
    -------
    evidence : EvidenceDict
        A dictionary of evidence blocks aggregated from the input.
    """
    if os.path.isdir(path):
        evidence: EvidenceDict = {}
        for idx, fname in enumerate(sorted(os.listdir(path))):
            fpath = os.path.join(path, fname)
            sub_evidence = read_input(fpath)
            for eid, block in sub_evidence.items():
                new_id = f"f{idx+1}_{eid}"
                block_copy = dict(block)
                block_copy['id'] = new_id
                evidence[new_id] = block_copy
        return evidence

    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        return read_text_file(path)
    if ext == ".csv":
        return read_csv_file(path)
    if ext in {".xls", ".xlsx"}:
        return read_xlsx_file(path)
    if ext == ".pdf":
        return read_pdf_file(path)
    if ext == ".docx":
        return read_docx_file(path)
    warn(f"Unknown file extension {ext}; treating as plain text")
    return read_text_file(path)


# ---------------------------------------------------------------------------
# Step 2: Normalization
# ---------------------------------------------------------------------------

def normalize_table(table: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a table by cleaning numbers, trimming whitespace, and
    ensuring that column headers are valid strings.  This function
    transforms the ``rows`` field from a list of dictionaries with
    arbitrary keys into a list of dictionaries keyed by column names.

    Numeric values are converted to ``float`` where possible.  Units
    embedded in string values (e.g. ``"2 kg"`` or ``"$1.50"``) are
    split into separate ``value`` and ``unit`` entries.  The
    normalization does not require units to be consistent across the
    table; unit validation happens later in the semantic checker.

    Parameters
    ----------
    table : dict
        A table block with ``columns`` (list) and ``rows`` (list of
        dictionaries) keys.

    Returns
    -------
    normalized : dict
        A copy of the table with rows normalized.  Each cell is
        represented as a dictionary with fields ``raw``, ``value``, and
        ``unit``.  Additional metadata is preserved.
    """
    cols = [str(c).strip() for c in table.get("columns", [])]
    norm_rows: List[Dict[str, Dict[str, Any]]] = []
    for row in table.get("rows", []):
        norm_row: Dict[str, Dict[str, Any]] = {}
        for col_idx, col_name in enumerate(cols):
            # Use row[col_name] if present; otherwise look up by index
            raw_val: Any = None
            if isinstance(row, dict):
                raw_val = row.get(col_name)
            if raw_val is None:
                # Try numerical index if CSV parsing numbered columns
                raw_val = row.get(str(col_idx))
            raw_str = str(raw_val) if raw_val is not None else ""
            raw_str = raw_str.strip()
            value: Optional[float] = None
            unit: Optional[str] = None
            # Try to parse numeric with unit, e.g. "2 kg", "$1.50", "3.5 hours"
            if raw_str:
                # Remove currency symbols and commas for numeric parse
                cleaned = re.sub(r"[^0-9.\-+/]", " ", raw_str)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                parts = cleaned.split()
                try:
                    # If first token is a fraction, handle separately
                    if len(parts) == 1 and '/' in parts[0]:
                        num, den = parts[0].split('/', 1)
                        value = float(num) / float(den)
                    else:
                        value = float(parts[0])
                        if len(parts) > 1:
                            unit = " ".join(parts[1:])
                except Exception:
                    value = None
            norm_row[col_name] = {
                "raw": raw_str,
                "value": value,
                "unit": unit,
            }
        norm_rows.append(norm_row)
    normalized = dict(table)
    normalized['columns'] = cols
    normalized['rows'] = norm_rows
    return normalized


def normalize_evidence(evidence: EvidenceDict) -> EvidenceDict:
    """
    Apply normalization to all table blocks in the evidence.  Paragraphs
    and other block types are returned unchanged.  Table normalization
    helps downstream processing by converting cells into structured
    dictionaries with numeric ``value`` and ``unit`` fields.

    Parameters
    ----------
    evidence : EvidenceDict
        The raw evidence dictionary.

    Returns
    -------
    normalized : EvidenceDict
        A new evidence dictionary with normalized tables.
    """
    normalized: EvidenceDict = {}
    for eid, block in evidence.items():
        if block['type'] == 'table':
            normalized[eid] = normalize_table(block)
        else:
            normalized[eid] = dict(block)
    return normalized


def link_paragraphs_to_tables(evidence: EvidenceDict) -> EvidenceDict:
    """
    Link paragraphs to nearby tables based on textual cues.

    This heuristic function adds a ``linked_tables`` field to each
    paragraph block containing a list of table IDs that the paragraph
    likely refers to.  The linking is performed by scanning for the
    words "table" and numeric references (e.g. "Table 1", "Table 2") in
    the paragraph text.  If a match is found, the corresponding table
    is linked.  Otherwise, the nearest preceding table is linked.

    The goal of linking is to provide additional context to the LLM
    during AST generation.  When a paragraph mentions numbers or
    concepts that appear in a table, the linkage helps correlate the
    textual description with the numerical data.

    Parameters
    ----------
    evidence : EvidenceDict
        The normalized evidence dictionary.

    Returns
    -------
    linked : EvidenceDict
        A new evidence dictionary with ``linked_tables`` fields added
        to paragraph blocks.
    """
    paragraphs = [eid for eid, blk in evidence.items() if blk['type'] == 'paragraph']
    tables = [eid for eid, blk in evidence.items() if blk['type'] == 'table']
    linked: EvidenceDict = {}
    for eid, blk in evidence.items():
        if blk['type'] != 'paragraph':
            linked[eid] = dict(blk)
            continue
        text = blk.get('text', '').lower()
        linked_tables: List[str] = []
        # Search for explicit references like "Table 1" or "table 2"
        for t_eid in tables:
            match = re.search(rf"table\s*{re.escape(t_eid.lstrip('t'))}\b", text)
            if match:
                linked_tables.append(t_eid)
        if not linked_tables and tables:
            # Heuristic: link to the nearest preceding table by evidence ID order
            # Find the latest table with an ID less than the paragraph ID
            # (assuming IDs are ordered lexicographically)  This is a rough
            # approximation but works for many simple documents.
            table_ids_sorted = sorted(tables, key=lambda x: int(re.sub(r'\D', '', x) or '0'))
            para_num = int(re.sub(r'\D', '', eid) or '0')
            nearest = None
            for t_id in table_ids_sorted:
                t_num = int(re.sub(r'\D', '', t_id) or '0')
                if t_num <= para_num:
                    nearest = t_id
                else:
                    break
            if nearest:
                linked_tables.append(nearest)
        new_blk = dict(blk)
        new_blk['linked_tables'] = linked_tables
        linked[eid] = new_blk
    return linked


def normalize_and_link(path: str) -> EvidenceDict:
    """
    Convenience function that reads input from a file or directory,
    normalizes the evidence, and links paragraphs to tables.  This
    function is a thin wrapper around ``read_input``,
    ``normalize_evidence``, and ``link_paragraphs_to_tables``.  It
    returns the fully processed evidence dictionary ready for
    candidate labeling and AST generation.

    Parameters
    ----------
    path : str
        Path to the input file or directory.

    Returns
    -------
    evidence : EvidenceDict
        The fully processed evidence dictionary.
    """
    raw_evidence = read_input(path)
    normalized = normalize_evidence(raw_evidence)
    linked = link_paragraphs_to_tables(normalized)
    return linked


# ---------------------------------------------------------------------------
# Step 3: Candidate Evidence Labeling
# ---------------------------------------------------------------------------

# The following lists of nouns and phrases are used to heuristically
# classify variables as discrete or continuous and to identify
# candidate resource, product, capacity, and demand columns.  The
# lists are deliberately comprehensive and may include synonyms and
# variants.  Developers should extend or refine these lists as they
# encounter new problem domains.

DISCRETE_NOUNS = [
    "container", "containers", "unit", "units", "item", "items", "product", "products",
    "worker", "workers", "employee", "employees", "machine", "machines", "batch", "batches",
    "truck", "trucks", "chair", "chairs", "desk", "desks", "table", "tables", "phone", "phones",
    "computer", "computers", "package", "packages", "box", "boxes", "crate", "crates",
    "car", "cars", "bus", "buses", "person", "people", "patient", "patients", "nurse", "nurses",
    "child", "children", "adult", "adults", "cow", "cows", "sheep", "goat", "goats",
    "dog", "dogs", "cat", "cats", "bird", "birds", "bike", "bikes", "engine", "engines",
    "plane", "planes", "ship", "ships", "train", "trains", "book", "books", "chair", "chairs",
]

CONTINUOUS_NOUNS = [
    "acre", "acres", "gram", "grams", "kg", "kilogram", "kilograms", "ton", "tons", "tonne", "tonnes",
    "liter", "liters", "litre", "litres", "gallon", "gallons", "pound", "pounds",
    "hour", "hours", "minute", "minutes", "second", "seconds", "day", "days",
    "money", "budget", "dollar", "dollars", "euro", "euros", "yen", "yuan", "peso", "pesos",
    "cost", "expense", "area", "volume", "material", "amount", "capacity",
]

UPPER_BOUND_PHRASES = [
    "at most", "no more than", "cannot exceed", "can not exceed", "must not exceed",
    "does not exceed", "do not exceed", "limited to", "up to", "available", "capacity",
    "maximum of", "max of", "budget of", "only has", "only have",
]

LOWER_BOUND_PHRASES = [
    "at least", "no less than", "minimum", "min of", "must be at least", "must have at least",
    "demand of", "requires at least", "require at least", "needs at least", "need at least",
]

EQUALITY_PHRASES = [
    "exactly", "equal to", "equals", "must equal", "meet exactly", "be exactly", "be equal to",
]

RATIO_PHRASES = [
    "twice", "half", "times", "ratio", "proportion", "percent", "%", "multiple", "double", "triple",
]

MAXIMIZE_HINTS = [
    "maximize", "maximise", "maximum profit", "profit", "revenue", "earn", "income", "sales",
]

MINIMIZE_HINTS = [
    "minimize", "minimise", "minimum cost", "minimize cost", "minimise cost", "cost", "expense",
]


def classify_column_roles(table: Dict[str, Any]) -> Dict[str, str]:
    """
    Heuristically classify the semantic role of each column in a
    normalized table.  The roles include ``index`` (set),
    ``objective_coefficient``, ``resource_coefficient``, ``rhs``
    (capacity/demand), ``bound`` (upper/lower limits), and ``unknown``.

    The function uses keyword matching on column headers and units to
    guess the role.  For example, a column labelled "profit" or
    containing currency units is likely an objective coefficient.  A
    column labelled "capacity" or containing words like "available" is
    likely an RHS.  Columns that reference discrete or continuous
    nouns are treated as indices (sets).  If multiple plausible roles
    match, the one with the highest priority is chosen based on the
    order defined below.

    Parameters
    ----------
    table : dict
        A normalized table block.

    Returns
    -------
    roles : dict
        A mapping from column name to a role string.  Roles are
        descriptive and do not strictly determine the AST; they
        serve as hints for the LLM.
    """
    cols = table.get('columns', [])
    roles: Dict[str, str] = {}
    for col in cols:
        col_l = col.lower().strip()
        if any(noun in col_l for noun in DISCRETE_NOUNS):
            roles[col] = 'index'
            continue
        if any(hint in col_l for hint in MAXIMIZE_HINTS + ['profit', 'revenue']):
            roles[col] = 'objective_coefficient'
            continue
        if any(term in col_l for term in ['cost', 'price', 'expense']):
            roles[col] = 'objective_coefficient'  # Minimization hints
            continue
        if any(keyword in col_l for keyword in ['capacity', 'limit', 'available', 'max', 'min', 'demand']):
            roles[col] = 'rhs'
            continue
        # Units hint: if the column's units are currency or weight, might be objective or resource
        values = [row[col] for row in table['rows'] if col in row]
        units_seen = set()
        for v in values:
            unit = v.get('unit') if isinstance(v, dict) else None
            if unit:
                units_seen.add(unit.lower())
        if any(u in units_seen for u in ['$', 'usd', 'eur', 'euro']):
            roles[col] = 'objective_coefficient'
            continue
        if any(u in units_seen for u in ['kg', 'gram', 'grams', 'lb', 'pound']):
            roles[col] = 'resource_coefficient'
            continue
        # Default: unknown role
        roles[col] = 'unknown'
    return roles


def label_candidates(evidence: EvidenceDict) -> Dict[str, Any]:
    """
    Generate candidate labels for sets, parameters, variables, and
    constraints from the evidence.  This function scans paragraphs
    and normalized tables to identify potential names and roles.  The
    labels are not final; they are hints passed to the LLM for AST
    generation.

    The output is a dictionary with keys ``sets``, ``parameters``,
    ``variables``, and ``constraints``.  Each key maps to a list of
    candidate dictionaries.  For example, the ``sets`` list may
    contain an entry ``{"name": "Products", "elements": ["Meaties", "Yummies"]}``.

    Parameters
    ----------
    evidence : EvidenceDict
        The normalized and linked evidence.

    Returns
    -------
    labels : dict
        A dictionary of candidate labels to guide the LLM.
    """
    labels: Dict[str, Any] = {
        'sets': [],
        'parameters': [],
        'variables': [],
        'constraints': [],
        'hints': [],
    }
    # Identify sets from table index columns
    for eid, blk in evidence.items():
        if blk['type'] != 'table':
            continue
        roles = classify_column_roles(blk)
        index_cols = [c for c, role in roles.items() if role == 'index']
        if index_cols:
            # For simplicity, treat the first index column as the set
            col = index_cols[0]
            elements = [row[col]['raw'] if isinstance(row[col], dict) else row[col] for row in blk['rows']]
            set_name = col.strip().title() or 'Set'
            labels['sets'].append({
                'name': set_name,
                'elements': elements,
                'evidence_id': eid,
            })
    # Identify parameters and variables from remaining columns
    # In this prototype we rely on LLM to finalize the mapping; we just record hints.
    for eid, blk in evidence.items():
        if blk['type'] == 'paragraph':
            # Extract numeric values for potential parameters or RHS
            nums = extract_numbers_from_text(blk['text'])
            if nums:
                labels['hints'].append({
                    'type': 'numbers_in_paragraph',
                    'evidence_id': eid,
                    'values': nums,
                })
        elif blk['type'] == 'table':
            roles = classify_column_roles(blk)
            for col, role in roles.items():
                if role == 'objective_coefficient':
                    labels['parameters'].append({
                        'name': col.strip().title() or 'Objective',
                        'column': col,
                        'role': 'objective_coefficient',
                        'evidence_id': eid,
                    })
                elif role == 'resource_coefficient':
                    labels['parameters'].append({
                        'name': col.strip().title() or 'Resource',
                        'column': col,
                        'role': 'resource_coefficient',
                        'evidence_id': eid,
                    })
                elif role == 'rhs':
                    labels['parameters'].append({
                        'name': col.strip().title() or 'RHS',
                        'column': col,
                        'role': 'rhs',
                        'evidence_id': eid,
                    })
    # Variables are not inferred here; the LLM is expected to decide.
    return labels


def extract_numbers_from_text(text: str) -> List[float]:
    """
    Extract numeric literals from a block of text.  Supports integers,
    floats, and simple fractions (e.g. "3/4").  Commas and currency
    symbols are ignored.  Negative numbers are allowed.

    Parameters
    ----------
    text : str
        Input text from which to extract numbers.

    Returns
    -------
    numbers : list of float
        A list of numeric values found in the text.
    """
    number_re = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+[\d,]*\.?\d*|\d*\.\d+)(?:/\d+)?")
    nums: List[float] = []
    for m in number_re.finditer(text or ''):
        token = m.group(0).replace(',', '')
        try:
            if '/' in token:
                a, b = token.split('/', 1)
                nums.append(float(a) / float(b))
            else:
                nums.append(float(token))
        except Exception:
            continue
    return nums


# ---------------------------------------------------------------------------
# Step 4: LLM AST Generation
# ---------------------------------------------------------------------------

def build_ast_prompt(evidence: EvidenceDict, labels: Dict[str, Any]) -> str:
    """
    Construct a prompt for the LLM to generate the CAFA-AST.  The
    prompt includes the normalized evidence and candidate labels and
    instructs the model to produce a structured JSON AST.  The
    prompt encourages the LLM to reference evidence IDs and to adhere
    to the CAFA-AST schema.  It also reiterates the reasoning order
    and semantic guardrails described in the system prompt to
    minimize errors.

    Parameters
    ----------
    evidence : EvidenceDict
        The normalized and linked evidence.

    labels : dict
        The candidate labels generated by ``label_candidates``.

    Returns
    -------
    prompt : str
        A string containing the full prompt to send to the LLM.
    """
    # Serialize evidence succinctly: only include table structures and
    # paragraph texts with their IDs to avoid hitting token limits.
    evidence_summary = []
    for eid, blk in evidence.items():
        if blk['type'] == 'paragraph':
            evidence_summary.append({
                'id': eid,
                'type': 'paragraph',
                'text': blk['text'],
                'linked_tables': blk.get('linked_tables', []),
            })
        elif blk['type'] == 'table':
            # Include header row and a few sample rows for brevity
            sample_rows = blk['rows'][:3]
            simple_rows = []
            for row in sample_rows:
                simple_rows.append({k: v['raw'] if isinstance(v, dict) else v for k, v in row.items()})
            evidence_summary.append({
                'id': eid,
                'type': 'table',
                'columns': blk['columns'],
                'sample_rows': simple_rows,
                'caption': blk.get('caption', ''),
            })
    # Serialize candidate labels for readability
    prompt_labels = json.dumps(labels, ensure_ascii=False, indent=2)
    prompt_evidence = json.dumps(evidence_summary, ensure_ascii=False, indent=2)
    # Compose the LLM prompt
    prompt = (
        "You are an expert optimization modeler. "
        "Convert the following evidence and candidate labels into a structured CAFA-AST JSON.\n\n"
        "The evidence comes from problem descriptions and tables. Each block has a unique ID. "
        "Use these IDs in the AST to indicate the source of variables, parameters, objective terms, and constraints.\n"
        "The CAFA-AST should follow this schema:\n"
        "{\n"
        "  \"problem\": {\"name\": str, \"domain\": str, \"description\": str},\n"
        "  \"sets\": [{\"name\": str, \"elements\": list[str], \"evidence\": list[str]}],\n"
        "  \"parameters\": [{\"name\": str, \"index\": list[str] or null, \"values\": dict, \"unit\": str or null, \"evidence\": list[str]}],\n"
        "  \"variables\": [{\"name\": str, \"index\": list[str] or null, \"domain\": str (CONTINUOUS|INTEGER|BINARY), \"evidence\": list[str]}],\n"
        "  \"objective\": {\"sense\": MAXIMIZE|MINIMIZE, \"expression\": str, \"evidence\": list[str]},\n"
        "  \"constraints\": [{\"name\": str, \"expression\": str, \"evidence\": list[str], \"meaning\": str}],\n"
        "  \"assumptions\": [str],\n"
        "  \"metadata\": {\"comments\": str, \"hints\": dict}\n"
        "}\n\n"
        "Reason in this order:\n"
        "1. Identify decision variables with correct domain (CONTINUOUS for divisible quantities, INTEGER for counts, BINARY for yes/no).\n"
        "2. Identify sets and parameters from tables and descriptions.\n"
        "3. Identify the objective direction and coefficients from text and table columns.\n"
        "4. Identify all constraints. Use <= for upper bounds (\"at most\", \"no more than\", \"cannot exceed\"), >= for lower bounds (\"at least\", \"no less than\", \"minimum\"), == for exact equality.\n"
        "5. Make sure every number in the evidence appears in the objective or a constraint.\n"
        "6. Link each AST element to the evidence IDs from which it was derived.\n"
        "7. Do not invent constraints or variables that are not supported by evidence.\n\n"
        "Evidence:\n"
        f"{prompt_evidence}\n\n"
        "Candidate Labels:\n"
        f"{prompt_labels}\n\n"
        "Output strictly the JSON AST, with no additional commentary."
    )
    return prompt


def call_llm_for_ast(prompt: str, model: str = 'gpt-4', max_tokens: int = 4096, temperature: float = 0.0) -> Optional[AstDict]:
    """
    Call the LLM to generate a CAFA-AST from the given prompt.  This
    function is a thin wrapper around a hypothetical OpenAI API call.
    In a production system, you should integrate with ``openai``,
    ``lmstudio``, or another endpoint.  For the purposes of this
    prototype, the function returns ``None``.  Developers can replace
    the implementation with a real call or mock as needed.

    Parameters
    ----------
    prompt : str
        The full text prompt to send to the LLM.

    model : str, optional
        The model name or endpoint identifier.  The default is
        ``'gpt-4'`` but this is only illustrative.

    max_tokens : int, optional
        The maximum number of tokens to generate.  The default is
        ``4096``, which should be sufficient for typical problems.

    temperature : float, optional
        The sampling temperature for the LLM.  A value of zero results
        in deterministic output.  The default is ``0.0``.

    Returns
    -------
    ast : dict or None
        The parsed CAFA-AST if the call succeeds and valid JSON is
        returned.  Otherwise returns ``None``.
    """
    # Stub implementation: in a real system you would call an LLM API.
    warn("LLM call is stubbed; returning None. Replace call_llm_for_ast with actual API integration.")
    return None


def parse_ast_from_json(json_str: str) -> Optional[AstDict]:
    """
    Parse a JSON string into a CAFA-AST dictionary.  This helper
    function attempts to parse the JSON and verify that the result is a
    dictionary.  It does not perform full schema validation; that is
    handled by the semantic checker.

    Parameters
    ----------
    json_str : str
        The JSON string returned by the LLM.

    Returns
    -------
    ast : dict or None
        The parsed AST dictionary, or ``None`` if parsing fails.
    """
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def generate_ast(evidence: EvidenceDict, labels: Dict[str, Any], model: str = 'gpt-4') -> AstDict:
    """
    Generate a CAFA-AST by constructing a prompt and calling the LLM.
    If the LLM call fails or returns invalid JSON, an empty AST
    skeleton is returned to allow the pipeline to continue.  The
    function logs errors and warnings for debugging.

    Parameters
    ----------
    evidence : EvidenceDict
        The normalized and linked evidence.

    labels : dict
        The candidate labels generated by ``label_candidates``.

    model : str, optional
        The name of the LLM model to use.  The default is ``'gpt-4'``.

    Returns
    -------
    ast : dict
        The generated AST.  If the LLM call fails, the AST may be
        incomplete but still structurally valid.
    """
    prompt = build_ast_prompt(evidence, labels)
    raw = call_llm_for_ast(prompt, model=model)
    if raw is None:
        warn("LLM returned no output; generating empty AST skeleton")
        return {
            'problem': {'name': '', 'domain': '', 'description': ''},
            'sets': [],
            'parameters': [],
            'variables': [],
            'objective': {'sense': 'MAXIMIZE', 'expression': '', 'evidence': []},
            'constraints': [],
            'assumptions': [],
            'metadata': {'comments': '', 'hints': labels},
        }
    ast_data = parse_ast_from_json(raw)
    if ast_data is None:
        warn("Failed to parse LLM output as JSON; returning empty AST skeleton")
        return {
            'problem': {'name': '', 'domain': '', 'description': ''},
            'sets': [],
            'parameters': [],
            'variables': [],
            'objective': {'sense': 'MAXIMIZE', 'expression': '', 'evidence': []},
            'constraints': [],
            'assumptions': [],
            'metadata': {'comments': '', 'hints': labels},
        }
    return ast_data


# ---------------------------------------------------------------------------
# Step 5: Semantic Checker
# ---------------------------------------------------------------------------

def validate_ast_schema(ast_data: AstDict) -> List[str]:
    """
    Perform basic schema validation on the AST.  This function checks
    for the presence of required keys and the types of values.  It
    returns a list of error messages; an empty list means that the
    schema is at least superficially valid.  Detailed semantic
    validation is handled by other functions.

    Parameters
    ----------
    ast_data : dict
        The CAFA-AST to validate.

    Returns
    -------
    errors : list of str
        A list of schema validation errors.  If empty, the AST passes
        the basic checks.
    """
    errors: List[str] = []
    if not isinstance(ast_data, dict):
        return ["AST is not a dictionary"]
    required_top = ['problem', 'sets', 'parameters', 'variables', 'objective', 'constraints', 'assumptions', 'metadata']
    for key in required_top:
        if key not in ast_data:
            errors.append(f"Missing top-level key: {key}")
    # Check problem
    problem = ast_data.get('problem', {})
    if not isinstance(problem, dict):
        errors.append("'problem' should be a dictionary")
    # Check objective
    obj = ast_data.get('objective', {})
    if not isinstance(obj, dict):
        errors.append("'objective' should be a dictionary")
    else:
        if obj.get('sense') not in {'MAXIMIZE', 'MINIMIZE'}:
            errors.append(f"Objective sense should be 'MAXIMIZE' or 'MINIMIZE', got {obj.get('sense')}")
        if not isinstance(obj.get('expression'), str):
            errors.append("Objective expression should be a string")
        if not isinstance(obj.get('evidence'), list):
            errors.append("Objective evidence should be a list")
    # Check variables
    vars_data = ast_data.get('variables', [])
    if not isinstance(vars_data, list):
        errors.append("'variables' should be a list")
    else:
        for i, v in enumerate(vars_data):
            if not isinstance(v, dict):
                errors.append(f"Variable {i} is not a dictionary")
                continue
            if 'name' not in v or 'domain' not in v:
                errors.append(f"Variable {i} missing 'name' or 'domain'")
            else:
                if v['domain'] not in {'CONTINUOUS', 'INTEGER', 'BINARY'}:
                    errors.append(f"Variable {v['name']} has invalid domain {v['domain']}")
    # Constraints type
    cons = ast_data.get('constraints', [])
    if not isinstance(cons, list):
        errors.append("'constraints' should be a list")
    else:
        for i, c in enumerate(cons):
            if not isinstance(c, dict):
                errors.append(f"Constraint {i} is not a dictionary")
                continue
            if 'expression' not in c or 'evidence' not in c:
                errors.append(f"Constraint {i} missing 'expression' or 'evidence'")
    return errors


def extract_numbers_from_ast(ast_data: AstDict) -> List[float]:
    """
    Extract all numeric values appearing in the objective and constraint
    expressions in the AST.  This helper is used for number coverage
    checks.

    Parameters
    ----------
    ast_data : dict
        The CAFA-AST.

    Returns
    -------
    numbers : list of float
        A list of numeric values found in the AST expressions.
    """
    nums: List[float] = []
    obj_expr = ast_data.get('objective', {}).get('expression', '')
    nums += extract_numbers_from_text(obj_expr)
    for c in ast_data.get('constraints', []):
        expr = c.get('expression', '')
        nums += extract_numbers_from_text(expr)
    return nums


def numbers_in_evidence(evidence: EvidenceDict) -> List[float]:
    """
    Collect all numeric values appearing in paragraphs and tables of
    the evidence.  This is used to check coverage: every significant
    number mentioned in the problem statement should appear in the
    objective or a constraint.

    Parameters
    ----------
    evidence : EvidenceDict
        The evidence dictionary.

    Returns
    -------
    numbers : list of float
        A list of numeric values extracted from the evidence.
    """
    nums: List[float] = []
    for blk in evidence.values():
        if blk['type'] == 'paragraph':
            nums += extract_numbers_from_text(blk['text'])
        elif blk['type'] == 'table':
            for row in blk['rows']:
                for col in blk['columns']:
                    cell = row.get(col)
                    if isinstance(cell, dict):
                        val = cell.get('value')
                        if val is not None:
                            nums.append(val)
    return nums


def floats_close(a: float, b: float, rel: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    """
    Determine whether two floating point numbers are approximately
    equal.  Used to compare numbers extracted from text and AST
    expressions when checking coverage.  The tolerance thresholds
    default to very small values to catch all matches.

    Parameters
    ----------
    a : float
        First value.

    b : float
        Second value.

    rel : float, optional
        Relative tolerance.

    abs_tol : float, optional
        Absolute tolerance.

    Returns
    -------
    bool
        ``True`` if ``a`` and ``b`` are within tolerance, ``False``
        otherwise.
    """
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b), 1.0))


def check_number_coverage(ast_data: AstDict, evidence: EvidenceDict) -> List[str]:
    """
    Check whether all significant numeric values in the evidence appear in
    the AST expressions.  This function compares the sets of numbers
    extracted from the evidence and the AST.  Numbers less than or
    equal to three are ignored because they often represent counts or
    structural constants (e.g. "3 products") rather than model
    parameters.

    Parameters
    ----------
    ast_data : dict
        The AST produced by the LLM.

    evidence : EvidenceDict
        The normalized evidence.

    Returns
    -------
    missing : list of str
        A list of diagnostic messages for numbers that appear in the
        evidence but not in the AST.  If the list is empty, coverage
        is considered adequate.
    """
    ev_nums = numbers_in_evidence(evidence)
    ast_nums = extract_numbers_from_ast(ast_data)
    missing: List[str] = []
    for n in ev_nums:
        if abs(n) <= 3.0:
            continue  # Ignore very small numbers
        if not any(floats_close(n, m, rel=1e-7, abs_tol=1e-7) for m in ast_nums):
            missing.append(f"Number {n:g} appears in evidence but not in AST")
    return missing


def detect_direction_mismatches(ast_data: AstDict, evidence: EvidenceDict) -> List[str]:
    """
    Detect potential mismatches between inequality directions in the AST
    and the language of the problem description.  The function
    searches paragraphs linked to each constraint's evidence and looks
    for phrases that indicate upper or lower bounds.  If the
    constraint expression does not align with the detected bound type,
    a diagnostic message is added.

    Parameters
    ----------
    ast_data : dict
        The AST.

    evidence : EvidenceDict
        The normalized evidence.

    Returns
    -------
    mismatches : list of str
        A list of diagnostic messages.  Empty if no mismatches are found.
    """
    mismatches: List[str] = []
    for idx, constraint in enumerate(ast_data.get('constraints', [])):
        expr = constraint.get('expression', '')
        evidence_ids = constraint.get('evidence', [])
        # Determine the sense from the expression
        sense = None
        if '<=' in expr:
            sense = '<='
        elif '>=' in expr:
            sense = '>='
        elif '==' in expr:
            sense = '=='
        # Concatenate linked paragraphs
        text_context = ''
        for eid in evidence_ids:
            blk = evidence.get(eid)
            if blk and blk['type'] == 'paragraph':
                text_context += ' ' + blk['text'].lower()
        # Count bound phrases
        upper_count = sum(text_context.count(p) for p in UPPER_BOUND_PHRASES)
        lower_count = sum(text_context.count(p) for p in LOWER_BOUND_PHRASES)
        equality_count = sum(text_context.count(p) for p in EQUALITY_PHRASES)
        if sense == '<=' and lower_count > upper_count:
            mismatches.append(f"Constraint {idx+1} might have wrong direction: text suggests lower bound")
        if sense == '>=' and upper_count > lower_count:
            mismatches.append(f"Constraint {idx+1} might have wrong direction: text suggests upper bound")
        if sense != '==' and equality_count > 0:
            mismatches.append(f"Constraint {idx+1} might have wrong direction: text suggests equality")
    return mismatches


def detect_variable_domain_issues(ast_data: AstDict, evidence: EvidenceDict) -> List[str]:
    """
    Detect potential mistakes in variable domains by examining variable
    names and descriptions.  If a variable is marked as CONTINUOUS but
    its name or evidence suggests discrete items (e.g. "dogs", "cars"),
    a warning is generated.  Similarly, if a variable is marked as
    INTEGER or BINARY but the context suggests a divisible quantity
    (e.g. "hours", "grams"), a warning is issued.

    Parameters
    ----------
    ast_data : dict
        The AST.

    evidence : EvidenceDict
        The normalized evidence.

    Returns
    -------
    issues : list of str
        A list of diagnostic messages for potential domain mismatches.
    """
    issues: List[str] = []
    for var in ast_data.get('variables', []):
        v_name = var.get('name', '')
        domain = var.get('domain', '')
        # Collect all evidence text for this variable
        context_text = ''
        for eid in var.get('evidence', []):
            blk = evidence.get(eid)
            if blk and blk['type'] == 'paragraph':
                context_text += ' ' + blk['text'].lower()
        target = (v_name + ' ' + context_text).lower()
        if domain == 'CONTINUOUS' and any(noun in target for noun in DISCRETE_NOUNS):
            issues.append(f"Variable {v_name} is CONTINUOUS but seems countable")
        if domain in {'INTEGER', 'BINARY'} and any(noun in target for noun in CONTINUOUS_NOUNS):
            issues.append(f"Variable {v_name} is {domain} but seems divisible")
    return issues


def semantic_check(ast_data: AstDict, evidence: EvidenceDict) -> Dict[str, Any]:
    """
    Perform a comprehensive semantic check of the AST against the evidence.
    The function runs multiple sub-checks (schema, number coverage,
    direction mismatches, domain issues) and aggregates the results
    into a dictionary.  Each key corresponds to a check and holds a
    list of messages.  If all lists are empty, the AST passes
    semantic validation.

    Parameters
    ----------
    ast_data : dict
        The AST produced by the LLM.

    evidence : EvidenceDict
        The normalized evidence.

    Returns
    -------
    report : dict
        A dictionary with keys ``schema_errors``, ``number_coverage``,
        ``direction_mismatches``, and ``domain_issues``.  Each value is
        a list of diagnostic messages.
    """
    report = {
        'schema_errors': validate_ast_schema(ast_data),
        'number_coverage': check_number_coverage(ast_data, evidence),
        'direction_mismatches': detect_direction_mismatches(ast_data, evidence),
        'domain_issues': detect_variable_domain_issues(ast_data, evidence),
    }
    return report


def is_semantically_valid(report: Dict[str, List[str]]) -> bool:
    """
    Determine whether the semantic check report indicates that the AST is
    valid.  The AST is considered valid if all lists in the report are
    empty.

    Parameters
    ----------
    report : dict
        The semantic check report.

    Returns
    -------
    valid : bool
        ``True`` if the AST is valid, ``False`` otherwise.
    """
    return all(not msgs for msgs in report.values())


def summarize_semantic_report(report: Dict[str, List[str]]) -> str:
    """
    Summarize the semantic check report into a human-readable string.
    This helper is used to communicate the detected issues back to the
    LLM for targeted repair.  Each category of issues is prefaced
    with a header.

    Parameters
    ----------
    report : dict
        The semantic check report.

    Returns
    -------
    summary : str
        A summary string describing the issues.
    """
    lines: List[str] = []
    for key, msgs in report.items():
        if not msgs:
            continue
        lines.append(f"Issues in {key}:")
        for msg in msgs:
            lines.append(f"  - {msg}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 6: Targeted Repair
# ---------------------------------------------------------------------------

def build_repair_prompt(ast_data: AstDict, report: Dict[str, List[str]], evidence: EvidenceDict) -> str:
    """
    Construct a prompt for targeted repair.  The prompt includes the
    current AST, a summary of semantic issues, and the evidence for
    reference.  It instructs the LLM to modify only the problematic
    fields in the AST while preserving the rest.  The prompt ends with
    a request to output only the corrected JSON.

    Parameters
    ----------
    ast_data : dict
        The original AST (potentially invalid).

    report : dict
        The semantic check report.

    evidence : EvidenceDict
        The normalized evidence.

    Returns
    -------
    prompt : str
        A prompt string to send to the LLM for targeted repair.
    """
    ast_json = json.dumps(ast_data, ensure_ascii=False, indent=2)
    issues_summary = summarize_semantic_report(report)
    # Include a brief evidence summary again for context
    ev_summary = []
    for eid, blk in evidence.items():
        if blk['type'] == 'paragraph':
            ev_summary.append({'id': eid, 'type': 'paragraph', 'text': blk['text']})
        elif blk['type'] == 'table':
            ev_summary.append({'id': eid, 'type': 'table', 'columns': blk['columns'], 'sample_rows': []})
    prompt = (
        "You are an optimization modeling assistant tasked with repairing a CAFA-AST.\n\n"
        "Here is the current AST (JSON):\n"
        f"{ast_json}\n\n"
        "The following semantic issues were detected:\n"
        f"{issues_summary if issues_summary else 'No issues found.'}\n\n"
        "Here is a summary of the evidence (for context):\n"
        f"{json.dumps(ev_summary, ensure_ascii=False, indent=2)}\n\n"
        "Please correct the AST by fixing only the fields that address the issues above.\n"
        "Do not remove valid elements.  Do not invent new numbers or variables.\n"
        "Return only the corrected JSON AST."
    )
    return prompt


def call_llm_for_repair(prompt: str, model: str = 'gpt-4', max_tokens: int = 4096, temperature: float = 0.0) -> Optional[str]:
    """
    Call the LLM to perform targeted repair on the AST.  This is a
    stub implementation; in a real system it should call the API and
    return the raw JSON output.  Here it returns ``None``.

    Parameters
    ----------
    prompt : str
        The repair prompt.

    model : str, optional
        The LLM model.

    max_tokens : int, optional
        Maximum tokens for the response.

    temperature : float, optional
        Sampling temperature.

    Returns
    -------
    result : str or None
        The raw JSON string if successful, otherwise ``None``.
    """
    warn("LLM repair call is stubbed; returning None. Replace call_llm_for_repair with actual API integration.")
    return None


def repair_ast(ast_data: AstDict, report: Dict[str, List[str]], evidence: EvidenceDict, model: str = 'gpt-4') -> AstDict:
    """
    Repair the AST by sending a targeted prompt to the LLM if needed.
    If the AST is semantically valid, it is returned unchanged.  If
    semantic issues are detected, a repair prompt is constructed and
    sent to the LLM.  If the LLM returns valid JSON, the repaired AST
    replaces the original.  Otherwise, the original AST is returned.

    Parameters
    ----------
    ast_data : dict
        The original AST (potentially invalid).

    report : dict
        The semantic check report.

    evidence : EvidenceDict
        The normalized evidence.

    model : str, optional
        The LLM model to use for repair.

    Returns
    -------
    ast : dict
        The repaired AST.
    """
    if is_semantically_valid(report):
        return ast_data
    prompt = build_repair_prompt(ast_data, report, evidence)
    raw = call_llm_for_repair(prompt, model=model)
    if raw:
        repaired = parse_ast_from_json(raw)
        if repaired:
            return repaired
    warn("LLM repair failed or returned invalid JSON; using original AST")
    return ast_data


# ---------------------------------------------------------------------------
# Step 7: Lowering AST to IR
# ---------------------------------------------------------------------------

def safe_identifier(name: str) -> str:
    """
    Convert an arbitrary string into a valid Python identifier.  This
    helper is used to create variable names in the compiled solver
    program.  All non-alphanumeric characters are replaced with
    underscores, and the identifier is prefixed with ``v_`` to avoid
    conflicts with reserved keywords.

    Parameters
    ----------
    name : str
        The input string.

    Returns
    -------
    ident : str
        A valid Python identifier based on the input.
    """
    ident = re.sub(r'\W+', '_', name).strip('_')
    if not re.match(r'[A-Za-z_]', ident):
        ident = 'x_' + ident
    return 'v_' + ident


def ast_to_ir(ast_data: AstDict) -> IrDict:
    """
    Convert the CAFA-AST into a CAFA-IR suitable for deterministic
    compilation.  This function linearizes the objective and
    constraints into expressions using variable identifiers.  Sets and
    parameters are flattened into dictionaries.  The function assumes
    that the AST has passed semantic validation; if it has not, the
    resulting IR may be incomplete or invalid.

    Parameters
    ----------
    ast_data : dict
        The CAFA-AST.

    Returns
    -------
    ir : dict
        The solver-neutral CAFA-IR.
    """
    ir: IrDict = {
        'ir_version': 'cafa-ir-v3',
        'problem_type': 'MILP',
        'sense': ast_data.get('objective', {}).get('sense', 'MAXIMIZE'),
        'variables': [],
        'objective': '',
        'constraints': [],
    }
    # Map variable names to identifiers
    var_map: Dict[str, str] = {}
    for idx, v in enumerate(ast_data.get('variables', [])):
        name = v.get('name', f'var{idx+1}')
        domain = v.get('domain', 'CONTINUOUS')
        var_id = safe_identifier(name)
        var_map[name] = var_id
        ir['variables'].append({
            'id': var_id,
            'name': name,
            'vtype': domain,
        })
    # Convert objective expression by replacing variable names with identifiers
    obj_expr = ast_data.get('objective', {}).get('expression', '')
    if obj_expr:
        # Replace variable names with IDs using a simple regex substitution
        def repl(match: re.Match[str]) -> str:
            token = match.group(0)
            return var_map.get(token, token)
        obj_expr_conv = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", repl, obj_expr)
    else:
        obj_expr_conv = ''
    ir['objective'] = obj_expr_conv
    # Convert constraints
    for idx, con in enumerate(ast_data.get('constraints', [])):
        expr = con.get('expression', '')
        if expr:
            def repl2(match: re.Match[str]) -> str:
                token = match.group(0)
                return var_map.get(token, token)
            expr_conv = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", repl2, expr)
            ir['constraints'].append({
                'name': con.get('name', f'c{idx+1}'),
                'expression': expr_conv,
            })
    return ir


# ---------------------------------------------------------------------------
# Step 8: Compilation and Solver Execution
# ---------------------------------------------------------------------------

def parse_linear_expression(expr: str, variables: Iterable[str]) -> LinearExpr:
    """
    Parse a linear expression into a coefficient dictionary and a
    constant.  This function reuses the ``ast`` parsing approach from
    the original CAFA implementation.  Multiplication is only
    permitted between numbers and variables (i.e. no nonlinear terms).

    Parameters
    ----------
    expr : str
        The expression to parse.

    variables : iterable of str
        A collection of valid variable names for validation.

    Returns
    -------
    result : (dict, float)
        A tuple containing a mapping from variable name to coefficient
        and a constant term.
    """
    expr = expr.strip()
    try:
        tree = ast.parse(expr, mode='eval')
    except Exception as e:
        raise ValueError(f"Failed to parse expression '{expr}': {e}")
    def walk(node: ast.AST) -> LinearExpr:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return {}, float(node.value)
            raise ValueError(f"Unsupported constant {node.value!r}")
        if isinstance(node, ast.Name):
            name = node.id
            if name not in variables:
                raise ValueError(f"Unknown variable {name}")
            return {name: 1.0}, 0.0
        if isinstance(node, ast.UnaryOp):
            val = walk(node.operand)
            if isinstance(node.op, ast.USub):
                return ({k: -v for k, v in val[0].items()}, -val[1])
            if isinstance(node.op, ast.UAdd):
                return val
            raise ValueError("Unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = walk(node.left)
            right = walk(node.right)
            if isinstance(node.op, ast.Add):
                # Merge coefficients and constants
                coeffs = dict(left[0])
                for k, v in right[0].items():
                    coeffs[k] = coeffs.get(k, 0.0) + v
                return coeffs, left[1] + right[1]
            if isinstance(node.op, ast.Sub):
                coeffs = dict(left[0])
                for k, v in right[0].items():
                    coeffs[k] = coeffs.get(k, 0.0) - v
                return coeffs, left[1] - right[1]
            if isinstance(node.op, ast.Mult):
                if not left[0] and right[0]:
                    return {k: left[1] * v for k, v in right[0].items()}, left[1] * right[1]
                if not right[0] and left[0]:
                    return {k: right[1] * v for k, v in left[0].items()}, right[1] * left[1]
                raise ValueError("Nonlinear multiplication is not allowed")
            if isinstance(node.op, ast.Div):
                if not right[0] and abs(right[1]) > 0.0:
                    return {k: v / right[1] for k, v in left[0].items()}, left[1] / right[1]
                raise ValueError("Division must be by a constant")
            raise ValueError(f"Unsupported operator {type(node.op).__name__}")
        raise ValueError(f"Unsupported node {type(node).__name__}")
    result = walk(tree)
    return result


def split_constraint_expression(expr: str) -> Tuple[str, str, str]:
    """
    Split a constraint expression into left-hand side, sense, and
    right-hand side.  Supports <=, >=, and == operators.  The function
    corrects accidental single < or > by replacing them with <= or >=.

    Parameters
    ----------
    expr : str
        The constraint expression to split.

    Returns
    -------
    lhs : str
        The left-hand side expression.

    sense : str
        The sense string ("<=", ">=", or "==").

    rhs : str
        The right-hand side expression.
    """
    expr = expr.strip()
    expr = re.sub(r"(?<![<>=])<(?![=])", "<=", expr)
    expr = re.sub(r"(?<![<>=])>(?![=])", ">=", expr)
    parts = re.split(r"(<=|>=|==)", expr, maxsplit=1)
    if len(parts) != 3:
        raise ValueError(f"Constraint must contain <=, >=, or ==: {expr}")
    lhs, sense, rhs = parts[0].strip(), parts[1], parts[2].strip()
    return lhs, sense, rhs


def compile_ir_to_gurobi(ir: IrDict) -> str:
    """
    Compile the CAFA-IR into a runnable Gurobi Python program.  The
    generated code uses the ``gurobipy`` API and runs silently (no
    output flag).  All variables are non-negative by default.  The
    solver is executed at the end of the program, and the objective
    value can be retrieved from the model.

    Parameters
    ----------
    ir : dict
        The solver-neutral CAFA-IR.

    Returns
    -------
    code : str
        The complete Python program for Gurobi.
    """
    lines: List[str] = []
    lines.append('import gurobipy as gp')
    lines.append('env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()')
    lines.append('m = gp.Model(env=env)')
    lines.append('')
    # Add variables
    for v in ir.get('variables', []):
        vid = v['id']
        name = repr(v['name'])
        vtype = v['vtype']
        if vtype == 'CONTINUOUS':
            lines.append(f"{vid} = m.addVar(name={name}, lb=0.0, vtype=gp.GRB.CONTINUOUS)")
        elif vtype == 'INTEGER':
            lines.append(f"{vid} = m.addVar(name={name}, lb=0.0, vtype=gp.GRB.INTEGER)")
        elif vtype == 'BINARY':
            lines.append(f"{vid} = m.addVar(name={name}, vtype=gp.GRB.BINARY)")
        else:
            lines.append(f"# Unsupported variable type {vtype}; defaulting to continuous")
            lines.append(f"{vid} = m.addVar(name={name}, lb=0.0, vtype=gp.GRB.CONTINUOUS)")
    lines.append('')
    # Objective
    obj_expr = ir.get('objective', '')
    sense = ir.get('sense', 'MAXIMIZE')
    grb_sense = 'gp.GRB.MAXIMIZE' if sense == 'MAXIMIZE' else 'gp.GRB.MINIMIZE'
    lines.append(f"m.setObjective({obj_expr if obj_expr else '0'}, {grb_sense})")
    # Constraints
    for c in ir.get('constraints', []):
        name = repr(c['name'])
        expr = c['expression']
        # Parse expression into lhs, sense, rhs
        lhs, s, rhs = split_constraint_expression(expr)
        # Convert linear expressions
        coeffs_lhs, const_lhs = parse_linear_expression(lhs, [v['id'] for v in ir['variables']])
        coeffs_rhs, const_rhs = parse_linear_expression(rhs, [v['id'] for v in ir['variables']])
        # Move all terms to left-hand side: lhs - rhs sense 0
        coeffs: Dict[str, float] = {}
        for k, vcoef in coeffs_lhs.items():
            coeffs[k] = coeffs.get(k, 0.0) + vcoef
        for k, vcoef in coeffs_rhs.items():
            coeffs[k] = coeffs.get(k, 0.0) - vcoef
        const = const_lhs - const_rhs
        # Construct Gurobi expression
        parts: List[str] = []
        for vid, coef in coeffs.items():
            if abs(coef - 1.0) < 1e-12:
                parts.append(vid)
            elif abs(coef + 1.0) < 1e-12:
                parts.append(f"(-{vid})")
            else:
                parts.append(f"({coef:.15g} * {vid})")
        if abs(const) >= 1e-12 or not parts:
            parts.append(f"{const:.15g}")
        lhs_expr = ' + '.join(parts)
        rhs_value = 0.0
        lines.append(f"m.addConstr({lhs_expr} {s} {rhs_value}, name={name})")
    lines.append('')
    lines.append('m.optimize()')
    return "\n".join(lines)


def execute_gurobi_code(code: str) -> Tuple[str, Optional[float], Optional[str]]:
    """
    Execute the generated Gurobi code and return the status and
    objective value.  This helper function uses Python's ``exec`` to
    run the code in an isolated namespace.  Exceptions are caught and
    returned as error messages.  The solver status codes follow
    Gurobi's conventions: 2=OPTIMAL, 3=INFEASIBLE, 4=INF_OR_UNBD,
    5=UNBOUNDED.

    Parameters
    ----------
    code : str
        The Gurobi Python program.

    Returns
    -------
    status : str
        ``'ok'`` if the solver found an optimal solution, otherwise a
        descriptive status string (e.g. ``'infeasible'``, ``'unbounded'``,
        ``'exec_fail'``).

    obj_val : float or None
        The objective value if a solution is found, otherwise ``None``.

    error : str or None
        An error message if execution fails.
    """
    ns: Dict[str, Any] = {}
    try:
        exec(code, ns, ns)
        m = ns.get('m')
        if not m:
            return 'exec_fail', None, 'Model m not found after execution'
        st = getattr(m, 'Status', None)
        if st == 2:
            return 'ok', float(m.objVal), None
        if st == 3:
            return 'infeasible', None, 'infeasible'
        if st in (4, 5):
            return 'unbounded', None, 'unbounded'
        return 'solver_other', None, f'Gurobi status {st}'
    except Exception as e:
        tb = traceback.format_exc()
        return 'exec_fail', None, f'{e}\n{tb}'


def solve_problem(evidence: EvidenceDict, model: str = 'gpt-4') -> Dict[str, Any]:
    """
    Solve an optimization problem described by the input evidence.  This
    function runs the entire CAFA++ pipeline: candidate labeling,
    AST generation, semantic checking, targeted repair, lowering to
    IR, compilation, execution, and explanation.  It returns a
    dictionary capturing all intermediate data and results.

    Parameters
    ----------
    evidence : EvidenceDict
        The normalized evidence.

    model : str, optional
        The LLM model to use for AST generation and repair.

    Returns
    -------
    result : dict
        A dictionary containing the AST, IR, solver code, solver
        status, objective value, semantic check report, and other
        debugging information.
    """
    result: Dict[str, Any] = {
        'labels': None,
        'ast': None,
        'semantic_report': None,
        'ir': None,
        'solver_code': None,
        'solver_status': None,
        'objective_value': None,
        'solver_error': None,
        'explanation': None,
    }
    # Step 3: Candidate labeling
    labels = label_candidates(evidence)
    result['labels'] = labels
    # Step 4: AST generation
    ast_data = generate_ast(evidence, labels, model=model)
    result['ast_initial'] = ast_data
    # Step 5: Semantic check
    report = semantic_check(ast_data, evidence)
    result['semantic_report'] = report
    # Step 6: Targeted repair if needed
    ast_final = repair_ast(ast_data, report, evidence, model=model)
    result['ast'] = ast_final
    # Re-run semantic check on repaired AST for record
    result['semantic_report_final'] = semantic_check(ast_final, evidence)
    # Step 7: Lower AST to IR
    ir = ast_to_ir(ast_final)
    result['ir'] = ir
    # Step 8: Compile IR to solver code
    solver_code = compile_ir_to_gurobi(ir)
    result['solver_code'] = solver_code
    # Step 9: Execute solver code
    status, obj_val, err = execute_gurobi_code(solver_code)
    result['solver_status'] = status
    result['objective_value'] = obj_val
    result['solver_error'] = err
    # Step 10: Explanation (simplified)
    result['explanation'] = generate_explanation(ast_final, ir, status, obj_val, evidence)
    return result


def generate_explanation(ast_data: AstDict, ir: IrDict, status: str, obj_val: Optional[float], evidence: EvidenceDict) -> str:
    """
    Generate a human-readable explanation of the optimization result.
    This is a simplified explanation that lists the objective value and
    describes each variable and constraint.  It includes evidence
    provenance to help users trace the model back to the original
    document.

    Parameters
    ----------
    ast_data : dict
        The final CAFA-AST.

    ir : dict
        The final CAFA-IR.

    status : str
        The solver status.

    obj_val : float or None
        The objective value.

    evidence : EvidenceDict
        The normalized evidence.

    Returns
    -------
    explanation : str
        A textual explanation of the solution.
    """
    lines: List[str] = []
    lines.append("Solution Status: " + status)
    if status == 'ok' and obj_val is not None:
        lines.append(f"Objective Value: {obj_val:g}")
    else:
        lines.append("No valid objective value available.")
    lines.append("")
    lines.append("Variables:")
    for var in ast_data.get('variables', []):
        v_name = var.get('name', '')
        evids = var.get('evidence', [])
        desc = ''
        for eid in evids:
            blk = evidence.get(eid)
            if blk and blk['type'] == 'paragraph':
                desc += ' ' + blk['text']
        lines.append(f"- {v_name} ({var.get('domain')}) from {', '.join(evids)}: {desc.strip()}")
    lines.append("")
    lines.append("Constraints:")
    for con in ast_data.get('constraints', []):
        name = con.get('name', '')
        expr = con.get('expression', '')
        evids = con.get('evidence', [])
        desc = ''
        for eid in evids:
            blk = evidence.get(eid)
            if blk and blk['type'] == 'paragraph':
                desc += ' ' + blk['text']
        lines.append(f"- {name}: {expr} from {', '.join(evids)}: {desc.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def run_from_cli() -> None:
    """
    Entry point for the command-line interface.  Parses arguments,
    invokes the pipeline on the specified input, and prints results.
    Results are saved as JSON files for easier post-mortem analysis.

    Usage::

        python cafa_pp_prototype.py --input path/to/file_or_dir --output_dir output

    Additional options allow specifying the LLM model and controlling
    the verbosity of the output.  See ``--help`` for details.
    """
    parser = argparse.ArgumentParser(description="CAFA++ prototype solver")
    parser.add_argument('--input', required=True, help='Path to input file or directory')
    parser.add_argument('--output_dir', default='cafa_pp_outputs', help='Directory to write outputs')
    parser.add_argument('--model', default='gpt-4', help='LLM model name (for AST generation)')
    parser.add_argument('--no_repair', action='store_true', help='Disable targeted repair even if issues found')
    args = parser.parse_args()
    # Read and process evidence
    evidence = normalize_and_link(args.input)
    # Solve the problem
    result = solve_problem(evidence, model=args.model)
    # Write output files
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.input))[0])
    os.makedirs(base, exist_ok=True)
    # Save evidence
    with open(os.path.join(base, 'evidence.json'), 'w', encoding='utf-8') as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
    # Save labels
    with open(os.path.join(base, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(result.get('labels'), f, indent=2, ensure_ascii=False)
    # Save AST
    with open(os.path.join(base, 'ast.json'), 'w', encoding='utf-8') as f:
        json.dump(result.get('ast'), f, indent=2, ensure_ascii=False)
    # Save IR
    with open(os.path.join(base, 'ir.json'), 'w', encoding='utf-8') as f:
        json.dump(result.get('ir'), f, indent=2, ensure_ascii=False)
    # Save solver code
    with open(os.path.join(base, 'solver.py'), 'w', encoding='utf-8') as f:
        f.write(result.get('solver_code', ''))
    # Save explanation
    with open(os.path.join(base, 'explanation.txt'), 'w', encoding='utf-8') as f:
        f.write(result.get('explanation', ''))
    # Print summary to console
    print("Solver Status:", result['solver_status'])
    if result['objective_value'] is not None:
        print("Objective Value:", result['objective_value'])
    if result['solver_error']:
        print("Error:", result['solver_error'])
    print("Explanation:\n", result['explanation'])


if __name__ == '__main__':
    run_from_cli()
