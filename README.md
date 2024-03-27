<div align="center">

# Helmholtz-Pulsating-Sphere

Data generation for acoustic pulsating spheres using the helmholtz equation.

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![Linkedin](https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/jakob-wagner-65b9871a9/)
</div>

## Setup

Install the required packages and libraries

```shell
conda env create -f environment.yml
conda env activate helmholtz-solver
pip install -e .
```

## Hooks

Install the git hook scripts

```shell
pre-commit install
```

now `pre-commit` will run automatically on `git commit `.

## Tests

Ensure the `test` optional dependencies are installed.
Run tests with

```shell
pytest test/
```
