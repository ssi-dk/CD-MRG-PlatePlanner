
[![PyPI - Version](https://img.shields.io/pypi/v/plate-planner.svg)](https://pypi.org/project/plate-planner)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/plate-planner.svg)](https://pypi.org/project/plate-planner)
[![pages-build-deployment](https://github.com/ssi-dk/CD-MRG-PlatePlanner/actions/workflows/pages/pages-build-deployment/badge.svg?branch=main)](https://github.com/ssi-dk/CD-MRG-PlatePlanner/actions/workflows/pages/pages-build-deployment)
-----

<img src="docs/assets/logo.png" alt="PlatePlannerLogo" title="PlatePlannerLogo" width="200"/>


- **Dynamic QC Sample Patterns**: Create plate layouts with customizable QC sample patterns.
- **Flexible Sample Distribution**: Distribute samples within groups across plates, accommodating variable numbers of samples per group.
- **Non-Split/Split Group Handling**: Control over distributing samples within groups without splitting them over plates or splitting when necessary.
- **Customizable Run Lists & Plate Visualizations**: Generate run lists and visualize plate assignments.
- **Block Randomization**: Utilize block randomization to prevent run order bias.
- **Stratified Randomization (Upcoming)**: This feature, once implemented, will ensure balanced distribution of group attributes across LC-MS plates or batches.

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
from plate_layout import Study, QCPlate

# Create a study and load your file with sample records (csv, xls/xlsx)
study = Study(name="cancer")
study.load_specimen_records(
    records_file="./data/study_samples.csv",
    sample_group_id_column="pair_ID",
    sample_id_column="specimen_ID"
)

# Block randomize groups
study.randomize_order(case_control=True, reproducible=False)

# Distribute samples to a 96-well plate with QC samples as defined in the toml file
qc_plate = QCPlate(plate_dim=(8, 12), QC_config="./data/plate_config_dynamic.toml")
study.distribute_samples_to_plates(plate_layout=qc_plate)

# Create visualization for plate 3
fig = study[3].as_figure(
    color_metadata_key="organ",
    annotation_metadata_key="object",
    rotation=45,
    fontsize=8
)

fig = study[3].to_layout_figures(
    annotation_metadata_key="sample_name",
    color_metadata_key="sample_code",
    file_format="png"
)
```

## Visualization Examples
Here are some examples of plate visualizations created using PlatePlanner:

![Alt text](docs/assets/cancer_Plate_7_object_organ.png "plate visualization example")

![Alt text](docs/assets/cancer_Plate_7_sample_name_sample_code.png "plate visualization example")

## Contributing

We welcome contributions to PlatePlanner! Please read our Contributing Guidelines for more information on how to get involved.

## License
PlatePlanner is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.


