# How to create the python package

```
python setup.py build_ext --inplace
```

```
python setup.py build_ext
python setup.py install --user
```

```
python setup.py sdist bdist_wheel
pip install .
```

[Upload your distribution](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#create-an-account)