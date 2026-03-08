Test wheel builds locally on macOS with Colima:

```bash
# Recommended: run cibuildwheel directly (Docker works; no act/container nesting)
pip install cibuildwheel
CIBW_BEFORE_BUILD="pip install cython numpy" \
CIBW_SKIP="cp36-* cp37-* cp38-* pp*" \
cibuildwheel --platform linux
```

Wheels will be in `./wheelhouse/`.

---

**act** (run full workflow): Docker socket path differs with Colima (`~/.colima/docker.sock` vs `/var/run/docker.sock`), so cibuildwheel inside act often fails. Use the direct cibuildwheel command above instead.
