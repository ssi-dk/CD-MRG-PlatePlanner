# QC Configuration File

## Overview

The QC configuration file is in the TOML (Tom's Obvious, Minimal Language) format. It defines the distribution of Quality Control (QC) and specimen samples within a multiwell plate. This file is structured into several key sections, each tailored for different aspects of the plate layout.

## Example
``` { .toml .copy }
[QC]
# should first well(s) be a QC round?
start_with_QC_round = true
# how many non-QC samples to analyze until we start a round of QC samples? 
run_QC_after_n_specimens = 11 

[QC.names] 
# Describes the key names and descriptions of the QC samples used.
EC = "EC: External_Control_(matrix)"
PB = "PB: Paper_Blank"
PO = "PO: Pooled_specimens"

[QC.patterns]
# start = ["EC", "PB"]
then_alternating = [["EC", "PB"], ["EC", "PO"]]
```

## Sections

### [QC] Section
This section includes general settings for the QC layout.

- `start_with_QC_round` (boolean): Determines whether the first wells should contain QC samples. Set to `true` to start with a QC round; `false` otherwise.
- `run_QC_after_n_specimens` (integer): The number of non-QC (specimen) samples to analyze before starting a new round of QC samples.

### [QC.names] Section
Describes the key names and descriptions of the QC samples used. For example:

- `EC` = "EC: External_Control_(matrix)"
- `PB` = "PB: Paper_Blank"
- `PO` = "PO: Pooled_specimens"

### [QC.patterns] Section
Defines the hierarchy and patterns for QC sample placement throughout the plate.

#### Hierarchy of Patterns:
- **Specific Round Patterns**: Highest priority. Specific instructions for individual rounds (e.g., `round_3`).
- **Start/End Patterns**: Determine patterns for the beginning (`start`) and end (`end`) of the plate.
- **Repeat Pattern**: Specifies a pattern (`repeat_pattern`) to be repeated a certain number of times.
- **Alternating Patterns**: Defines alternating patterns (`then_alternating`) that provide a general structure.
- **Every N Rounds Patterns**: Lowest priority. General patterns applied every N rounds (e.g., `every_4_rounds`).

#### Pattern Examples:
- `round_3 = ["EC", "PO", "PB"]`: Specific pattern for round 3.
- `every_4_rounds = ["EC", "PB"]`: Pattern applied every 4 rounds.
- `start = ["EC", "PB"]`: Pattern for the start of the plate.
- `then_alternating = [["PB", "PO"], ["EC", "PO"]]`: Alternating patterns.
- `end = ["PB"]`: Pattern for the end of the plate.
- `repeat_pattern = { pattern = ["EC", "PB"], times = 3 }`: Repeats the specified pattern 3 times.

!!! note
    Ensure that the sum of all patterns does not exceed the plate capacity and that each pattern is configured correctly to avoid layout errors.