
[![PyPI - Version](https://img.shields.io/pypi/v/plate-planner.svg)](https://pypi.org/project/plate-planner)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/plate-planner.svg)](https://pypi.org/project/plate-planner)
[![pages-build-deployment](https://github.com/ssi-dk/CD-MRG-PlatePlanner/actions/workflows/pages/pages-build-deployment/badge.svg?branch=main)](https://github.com/ssi-dk/CD-MRG-PlatePlanner/actions/workflows/pages/pages-build-deployment)
[![.github/workflows/publish-to-pypi](https://github.com/ssi-dk/CD-MRG-PlatePlanner/actions/workflows/publish-to-pypi.yml/badge.svg)](https://github.com/ssi-dk/CD-MRG-PlatePlanner/actions/workflows/publish-to-pypi.yml)
-----

<img src="docs/assets/logo.png" alt="PlatePlannerLogo" title="PlatePlannerLogo" width="300"/>


- **Dynamic QC Sample Patterns**: Create plate layouts with customizable QC sample patterns.
- **Flexible Sample Distribution**: Distribute samples within groups across plates, accommodating variable numbers of samples per group.
- **Non-Split/Split Group Handling**: Control over distributing samples within groups without splitting them over plates or splitting when necessary.
- **Customizable Run Lists & Plate Visualizations**: Generate run lists and visualize plate assignments.
- **Block Randomization**: Perform sample randomizaition or sample block randomization to prevent run order bias.

## Installation

To install PlatePlanner, simply run:

```console
pip install plate-planner
```

## Documentation

For more detailed documentation, visit [PlatePlanner Documentation](https://ssi-dk.github.io/CD-MRG-PlatePlanner/).

## Quick Start Guide

Here's a quick example of how to use PlatePlanner:

```py
from plate-planner import Study, QCPlate

# Create a study and load your file with sample records (csv, xls/xlsx)
study_with_qc = Study(name="cancer")
study_with_qc.load_specimen_records(
    records_file="./data/study_samples.csv",
    sample_group_id_column="pair_ID",
    sample_id_column="specimen_ID"
)

# Block randomize groups
study_with_qc.randomize_order(case_control=True, reproducible=False)

# Distribute samples to a 96-well plate with QC samples as defined in the toml file
qc_plate = QCPlate(plate_dim=(8, 12), QC_config="./data/plate_config_dynamic.toml")
study_with_qc.distribute_samples_to_plates(plate_layout=qc_plate)

# Create visualization for plate 3
fig = study_with_qc[2].as_figure(
    color_metadata_key="organ",
    annotation_metadata_key="date_of_birth",
    rotation=45,
    )

```
![Alt text](docs/assets/qc_plate_layout_example.png "plate visualization example")


```py
#mshow distribution of sample attribute across all plates
fig = study_with_qc.plot_attribute_plate_distributions(attribute="organ", normalize=True, plt_style="fivethirtyeight")

```

<img src="docs/assets/study_attribute_plate_distribution_example.png" alt="attribute distribution across plates visualization example" title="Attribute plate distributions" width="800"/>

<!-- ![Alt text](docs/assets/study_attribute_plate_distribution_example.png "attribute distribution across plates visualization example") -->

## Contributing

We welcome contributions to PlatePlanner! Please read our Contributing Guidelines for more information on how to get involved.

## License
PlatePlanner is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.


