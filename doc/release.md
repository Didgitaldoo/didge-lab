# How to make a realease

1. Increment version number in pyproject.toml
2. Run all tests

Go to project root

```
source .venv/bin/activate
pytest
```

3. Git tag

```
git tag -a vTAGNAME -m "Descriptive message for the tag"
git push origin vTAGNAME
```

4. Build Python package and push to pip

Go to project root

```
python -m build
python -m twine upload dist/*
```