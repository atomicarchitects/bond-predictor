# A Simple Bond Predictor

Tired of lookup tables?

```bash
pip install git+https://github.com/atomicarchitects/bond-predictor.git
```

Then, run against a single molecule:

```bash
predict-bonds benzene.xyz -o benzene.sdf
```

Or, against a directory of molecules:

```bash
predict-bonds molecules-xyz/ -o molecules-sdf --extension sdf
```