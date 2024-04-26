# fish-risk
Modelling stakeholder-agent risk response through metrics

## Setup

Setup is straightforward with [poetry](https://python-poetry.org/docs/).
```bash
# In project root
poetry install
```
This will install `src` modules, i.e. `sim`.  
You can then use our package modules such as `sim.models` and `sim.utils`.

Our package and its dependencies are accessed in a poetry env.
So to start a python session or notebook, simply do
```bash
poetry run python
# or
poetry run jupyter lab
```

### Dev
For devs, if you need to use a new package run
```bash
poetry add <package>
```
which will update the env and `pyproject.toml`.  
Or if you want finer control over the package version for instance,
you can update `pyproject.toml` directly and run `poetry update`.
