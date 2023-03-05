# plate_layout

[![PyPI - Version](https://img.shields.io/pypi/v/plate-layout.svg)](https://pypi.org/project/plate-layout)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/plate-layout.svg)](https://pypi.org/project/plate-layout)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)
- [Example](#example)

## Installation

```console
pip install plate-layout
```

## License

`plate-layout` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

### TODO
- Add python script and a main function for CLI usage ...

## Example

### Defining the plate type and quality control setup
The plate dimensions and (optional) quality control layouts are are defined in a toml file; see `plate_config.toml' in the config folder for an example. 


```python

import plate_layout as pl
import pandas as pd
import numpy as np
import logging

pl.logger.setLevel(logging.INFO)
```

### Create a plate layout 
Create plate design by specifying the path to a config file directly when instantiating the class, 


### Load study data and randomize order 

### Batches - distributing samples on plates


### Export batch lists to file
You can export batch lists using the `to_file` method and specify desired file format and which columns to export. 

### Plot plate layouts
