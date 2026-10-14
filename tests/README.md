# GeoRescue AI — Test Suite

This directory contains the automated test suite for GeoRescue AI. All
tests run locally, use synthetic data, and require **no credentials**
and **no network access**.

## How to run the tests

From the project root, with the virtual environment activated:

```powershell
$env:PYTHONPATH = "src"
pytest tests/ -v