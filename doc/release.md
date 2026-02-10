# How to make a realease

1. Increment version number in pyproject.toml
2. Run all tests

```
pytest
```

3. Git tag

4. Build Python package and push to pip

Go to project root

```
python -m build
python -m twine upload dist/*
```