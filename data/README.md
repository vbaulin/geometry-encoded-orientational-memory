# Data

No numerical results are committed to this software repository. Raw
trajectories, derived reports, manifests, and checksums are distributed through
the companion [Zenodo data deposit](https://doi.org/10.5281/zenodo.22544787).
This version includes the complete N=1024 matched-preparation data; it contains
405 checksum-listed files. The all-versions DOI is
[10.5281/zenodo.22173160](https://doi.org/10.5281/zenodo.22173160).
After extracting it, run:

```bash
python scripts/install_zenodo_data.py /path/to/extracted/deposit
```

The installer verifies each manifest file's size and SHA256 and maps the archived files into
the paths expected by the simulation, plotting, and audit programs.
Raw data are linked by default; derived reports are independent working copies.
