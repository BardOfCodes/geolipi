# Installation Guide

## Requirements

- Python 3.10+
- `pip`

## Basic install

```bash
python -m pip install geolipi
```

## Install from source

```bash
git clone https://github.com/BardOfCodes/geolipi.git
cd geolipi
python -m pip install .
```

## Editable install (for development)

```bash
python -m pip install -e .[dev]
```

## Optional dependency groups

```bash
# Documentation toolchain
python -m pip install -e .[docs]

# Plotting/visualization stack
python -m pip install -e .[viz]

# Blender / geometry nodes integration
python -m pip install -e .[blender]
```

## Requirements-file shortcuts

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-docs.txt
python -m pip install -r requirements-blender.txt
```
