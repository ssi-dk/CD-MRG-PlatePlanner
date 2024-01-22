# Easy Plate Layouts for LC-MS Studies and Beyond
PlatePlanner is a Python package designed to simplify plate layout creation for LC-MS studies, and is also applicable for various laboratory applications where sample distribution to plates is required, with or without QC sample patterns. Its user-friendly API supports dynamic QC sample patterns, easy creation of run lists, and plate visualizations.

## Key features 
- **Dynamic QC Sample Patterns**: Effortlessly create plate layouts with customizable QC sample patterns.
- **Flexible Sample Distribution**: Distribute samples within groups across plates, accommodating variable numbers of samples per group. This feature is ideal for studies with uneven sample time points or group sizes.
- **Non-Split/ Split Group Handling**: Choose to distribute samples within groups without splitting them over plates, or opt for splitting when necessary, offering you complete control over your sample layout.
- **Customizable Run Lists & Plate Visualizations**: Generate detailed run lists and visualize plate assignments seamlessly.
- **Block Randomization**: Prevent run order bias by using the block randomization feature. 
- NOT IMPLEMENTED: **Stratified Randomization for Sample Balance**: Prevent bias in sample group characteristics. This feature ensures an even distribution of important group attributes (like gender, age, disease status, etc.) across all LC-MS plates or batches. 

## Installation

```
pip install plate-planner
```

## Example
Here's a quick guide to using PlatePlanner:

```py
from plate_layout import Study, QCPlate

# create a study and load your sfile with sample records (csv, xls/xlsx)
study = Study.(name="cancer")
study.load_specimen_records(
        records_file = "./data/study_samples.csv",
        sample_group_id_column = "pair_ID",
        sample_id_column = "specimen_ID"
)

# block randomize groups
study.randomize_order(case_control=True, reproducible=False)

# distribute samples to a 96-well plate with QC samples as defined in the toml file
qc_plate = QCPlate(plate_dim=(8,12), QC_config="./data/plate_config_dynamic.toml")
study.distribute_samples_to_plates(plate_layout=qc_plate)


# create visualization for plate 3
fig = study[3].as_figure(color_metadata_key="organ", annotation_metadata_key="object", rotation=45, fontsize=8)

fig = study[3].to_layout_figures(annotation_metadata_key="sample_name", color_metadata_key="sample_code", file_format="png")

```


![Alt text](assets/cancer_Plate_7_object_organ.png "Image Title")

![Alt text](assets/cancer_Plate_7_sample_name_sample_code.png "Image Title")


