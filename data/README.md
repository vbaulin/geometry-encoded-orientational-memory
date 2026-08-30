# Data

No numerical results are committed to this software repository. Raw
trajectories, derived reports, manifests, and checksums are distributed through
the companion [Zenodo data deposit](https://doi.org/10.5281/zenodo.22174752).
After extracting it, run:

```bash
python scripts/install_zenodo_data.py /path/to/extracted/deposit
```

The installer validates the deposit manifest and maps the archived files into
the paths expected by the simulation, plotting, and audit programs.
