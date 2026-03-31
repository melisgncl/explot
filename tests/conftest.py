from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from simulator.proteomics import ProteomicsSimulator
from simulator.scrna import ScrnaSimulator
from simulator.tabular import TabularSimulator


@pytest.fixture()
def tabular_data():
    return TabularSimulator().generate(seed=42)


@pytest.fixture()
def scrna_data():
    return ScrnaSimulator().generate(seed=42)


@pytest.fixture()
def proteomics_data():
    return ProteomicsSimulator().generate(seed=42)


@pytest.fixture()
def workspace_tmp_path():
    base_dir = Path(__file__).resolve().parent.parent / "_tmp_test_cases"
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / f"case_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
