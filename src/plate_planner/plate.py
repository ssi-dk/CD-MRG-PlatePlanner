import itertools
import tomli
import glob
import string
from dataclasses import dataclass, field, asdict
from pathlib import Path
import warnings
import json

from typing import Tuple, Union, Optional, Dict, Any, List, Iterator

import numpy as np
import pandas as pd

import plotly.express as px

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch

from .logger import logger

# parameters governing how numpy arrays are printed to console
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


@dataclass
class Well:
    """
    A class to represent a well in a multiwell plate.

    This class provides functionalities to represent and manipulate the properties 
    of a well, including its name, plate ID, coordinate, index, color, and metadata.

    Attributes:
        name (str): The name of the well (default: "A1").
        plate_id (int): The ID of the plate the well belongs to (default: 1).
        coordinate (Tuple[int, int]): The (row, column) coordinate of the well in the plate (default: (0, 0)).
        index (int): The index of the well (optional).
        rgb_color (Tuple[float, float, float]): The RGB color representation of the well (default: (1, 1, 1)).
        metadata (Dict[str, Any]): Additional metadata for the well (default: empty dictionary).

    Example:
        >>> well = Well(name="B2", plate_id=2, coordinate=(1, 6), index=13,)
        >>> well
        Well(name='B2', plate_id=2, coordinate=(1, 6), index=13, empty=True, rgb_color=(1, 1, 1), metadata={})

    """
    name: str = "A1"
    plate_id: int = 1
    coordinate: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    index: int = 0
    empty: bool = True
    rgb_color: Tuple[float, float, float] = field(default_factory=lambda: (1, 1, 1))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """
        Provide an unambiguous string representation of the Well object.

        Returns:
            str: A string representation of the well.

        Example:
            >>> well = Well(name="B3", plate_id=2, coordinate=(1, 2), index=3)
            >>> repr(well)
            "Well(name='B3', plate_id=2, coordinate=(1, 2), index=3, empty=True, rgb_color=(1, 1, 1), metadata={})"
        """
        return (f"Well(name='{self.name}', plate_id={self.plate_id}, "
                f"coordinate={self.coordinate}, index={self.index}, "
                f"empty={self.empty}, rgb_color={self.rgb_color}, metadata={self.metadata})")

    def __eq__(self, other) -> bool:
        """
        Compare this Well object with another for equality.

        Args:
            other (Well): Another Well object to compare with.

        Returns:
            bool: True if both Well objects are considered equal, False otherwise.

        Example:
            >>> well1 = Well(name="A1", plate_id=1)
            >>> well2 = Well(name="A1", plate_id=1)
            >>> well3 = Well(name="B1", plate_id=1)
            >>> well4 = Well(name="A1", plate_id=2)
            >>> well1 == well2
            True
            >>> well1 == well3
            False
            >>> well4 == well1
            False
        """
        if isinstance(other, Well):
            return (self.name == other.name and self.plate_id == other.plate_id
                    and self.coordinate == other.coordinate
                    and self.index == other.index and self.empty == other.empty
                    and self.rgb_color == other.rgb_color and self.metadata == other.metadata)
        return False

    def as_dict(self, flat=True) -> dict:
        """
        Converts the well object to a dictionary.

        The method returns a dictionary representation of the well object with the 
        direct attributes of the well and the keys in the metadata attribute.

        Returns:
            dict: A dictionary representation of the well object.

        Example:
            Convert a Well instance to a dictionary:
            >>> well = Well(name="B2", plate_id=2, coordinate=(1, 6), index=13, rgb_color=(0.5, 0.5, 0.5))
            >>> well_dict = well.as_dict()
            >>> print(well_dict)
            {'name': 'B2', 'plate_id': 2, 'coordinate': (1, 6), 'index': 13, 'empty': True, 'rgb_color': (0.5, 0.5, 0.5)}

        """
        attrib_dict = asdict(self)
        if flat:
            del attrib_dict["metadata"]
            attrib_dict.update(self.metadata)

        return attrib_dict
    
    @staticmethod
    def dict_to_well(well_dict: dict):
        """
        Converts a dictionary into a Well object.

        Args:
            well_dict (dict): A dictionary representing the well's attributes.

        Returns:
            Well: A Well object created from the dictionary.

        Example:
            >>> well_dict = {'name': 'A1', 'plate_id': 1, 'coordinate': (0, 0), 'index': 0, 'empty': True, 'rgb_color': (1, 1, 1), 'metadata': {}}
            >>> Well.dict_to_well(well_dict).name
            'A1'
        """
        return Well(**well_dict)
    
    @staticmethod
    def json_to_well(json_str: str) -> 'Well':
        """
        Deserializes a JSON string back into a Well object.

        Args:
            json_str (str): The JSON string representation of a well.

        Returns:
            Well: The deserialized Well object.

        Example:
            >>> json_str = '{"name": "A1", "plate_id": 1, "coordinate": [0, 0], "index": 0, "empty": true, "rgb_color": [1, 1, 1], "metadata": {}}'
            >>> Well.json_to_well(json_str).name
            'A1'
        """
        well_dict = json.loads(json_str)
        return Well.dict_to_well(well_dict)
    
    def as_json(self) -> str:
            """
            Serializes the Well object to a JSON string, preserving the structure of the metadata.

            Returns:
                str: A JSON string representation of the well object.

            Example:
                >>> well = Well(name="A1", plate_id=1, coordinate=(0, 0), index=0, rgb_color=(1, 1, 1), metadata={})
                >>> json_str = well.as_json()
                >>> isinstance(json_str, str) and "A1" in json_str
                True
            """
            # Directly use asdict for serialization since it preserves the structure of nested objects like metadata
            well_dict = asdict(self)
            return json.dumps(well_dict)
    
    def get_attribute_or_metadata(self, key: str) -> Any:
        """
        Get the value of a direct attribute or a key in the metadata dictionary.

        This method first checks if the provided key corresponds to a direct 
        attribute of the well object. If not, it then checks if the key exists 
        in the metadata dictionary.

        Args:
            key (str): The attribute name or metadata key.

        Returns:
            Any: The value of the attribute or metadata key, if found. Returns 'NaN' if not found.

        Example:
        # Retrieve attribute and metadata values from a Well instance:
        >>> well = Well(name="B2", plate_id=2, coordinate=(1, 1), index=5, rgb_color=(0.5, 0.5, 0.5))
        >>> well.metadata = {"sample_type": "plasma"}    
        >>> well.get_attribute_or_metadata("plate_id")
        2
        >>> well.get_attribute_or_metadata("sample_type")
        'plasma'
        """
        # Check if it's a direct attribute
        if hasattr(self, key):
            return getattr(self, key, "NaN")
            # return getattr(self, key, "")

        # Check if it's a key in metadata
        return self.metadata.get(key, "NaN")
        # return self.metadata.get(key, "")
    
    def set_attribute_or_metadata(self, key: str, value: Any) -> None:
        """
        Set the value of a direct attribute or a key in the metadata dictionary.

        This method first checks if the provided key corresponds to a direct 
        attribute of the well object. If so, it sets the value of that attribute. 
        If not, it then updates or adds the key-value pair in the metadata dictionary.

        Args:
            key (str): The attribute name or metadata key.
            value (Any): The value to be set for the attribute or metadata key.

        Example:
        # Set attribute and metadata values for a Well instance:
        >>> well = Well(name="B2", plate_id=2, coordinate=(1, 1), index=5, rgb_color=(0.5, 0.5, 0.5))
        >>> well.set_attribute_or_metadata("plate_id", 3)
        >>> well.set_attribute_or_metadata("sample_type", "serum")
        """
        # Check if it's a direct attribute
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            # Set/Add a key-value pair in metadata
            self.metadata[key] = value


class Plate:
    """
    A class to represent a multiwell plate.

    This class manages a multiwell plate, with functionalities to access, modify, and visualize
    the wells, along with their metadata.

    Attributes:
        _default_n_rows (int): Default number of rows in the plate.
        _default_n_columns (int): Default number of columns in the plate.
        _default_well_color (Tuple[float, float, float]): Default RGB color of the wells.
        _default_exclude_metadata (list): Default metadata keys to exclude.
        _default_colormap (str): Default colormap for visualizations.

    Parameters:
        plate_dim (Tuple[int, int], optional): The dimensions of the plate as (rows, columns).
        plate_id (int, optional): A unique identifier for the plate.

    """

    _default_n_rows: int = 8
    _default_n_columns: int = 12
    _default_well_color: Tuple[float, float, float] = (1, 1, 1)
    _default_exclude_metadata = ["rgb_color", "coordinate"]
    _default_colormap: str = "tab20"

    def __init__(self, plate_dim: Union[Tuple[int, int], List[int], Dict[str, int], int] = None, plate_id: int = 1):
        """
        Initialize a new Plate instance.

        Args:
            plate_dim (Tuple[int, int], optional): The dimensions of the plate as (rows, columns).
                If None, default dimensions are used.
            plate_id (int, optional): A unique identifier for the plate.

        The constructor initializes the plate with the specified dimensions, generating wells 
        with default properties and assigning them unique coordinates and identifiers.

        Examples:
            Creating a Plate instance with default dimensions and a specific plate ID:

            >>> plate = Plate()
            >>> plate.size
            96

            >>> plate = Plate(plate_dim=(16, 24))
            >>> plate.size
            384

        """

        self._n_rows, self._n_columns = self._parse_plate_dimensions(plate_dim)

        self._rows = list(range(self._n_rows))
        self._columns = list(range(self._n_columns))

        self._alphanumerical_coordinates = self.create_alphanumerical_coordinates(self._rows, self._columns)
        self._coordinates = self.create_index_coordinates(self._rows, self._columns)

        self.size = self._n_rows * self._n_columns

        self.wells = [Well(name=self._alphanumerical_coordinates[index], 
                           coordinate=(row, col), 
                           index=index, 
                           plate_id=plate_id, 
                           rgb_color=self._default_well_color)
                      for index, (row, col) in enumerate(itertools.product(self._rows[::-1], self._columns))]
        
        self.plate_id = plate_id
        
        # dictionary to map well names to indices
        self._name_to_index_map = {well.name: well.index for well in self.wells}
        self._index_to_coordinates_map = {well.index: well.coordinate for well in self.wells}
        self._index_to_name_map = {well.index: well.name for well in self.wells}

        logger.info(f"Created a {self._n_rows}x{self._n_columns} plate with {self.size} wells.")

    def __iter__(self) -> Iterator:
        """
        Return an iterator over the wells of the plate.

        This allows direct iteration over the plate object itself.

        Example:
            >>> plate = Plate(plate_dim=(2, 2))
            >>> [well.name for well in plate]
            ['A1', 'A2', 'B1', 'B2']
        """
        return iter(self.wells)

    def __len__(self) -> int:
        """
        Returns the number of wells in the plate.

        Returns:
            int: The number of wells.

        Example:
            >>> plate = Plate()
            >>> len(plate)
            96
        """
        return len(self.wells)
    
    def __str__(self) -> str:
        plate_summary = f"Plate ID: {self.plate_id}\n"
        plate_summary += f"Dimensions: {self._n_rows} rows x {self._n_columns} columns\n"
        plate_summary += "Plate Layout (Well Names):\n"
        plate_array_str = np.array_str(self.get_metadata_as_numpy_array("name"))
        plate_summary += plate_array_str
        return plate_summary
    
    def __getitem__(self, key: Union[int, Tuple[int,int], str]) -> Well:
        """
        Retrieve a well from the plate based on its index, coordinate, or name.

        Args:
            key (int, tuple, or str): The identifier for the well. Can be an integer index, a tuple indicating row and column coordinates, or a string specifying the well's name.

        Returns:
            Well: The well object corresponding to the given key.

        Raises:
            TypeError: If the key is not an integer, tuple, or string.

        Example:
            >>> plate = Plate()
            >>> plate[0].name
            'A1'
            >>> plate[(0, 0)].name
            'A1'
            >>> plate["A1"].name
            'A1'
        """
        if isinstance(key, int):
            # Access by index
            return self.wells[key]
        elif isinstance(key, tuple):
            # Access by coordinate
            index = self._coordinates_to_index(key)
            return self.wells[index]
        elif isinstance(key, str):
            # Access by name
            index = self._name_to_index_map[key]
            return self.wells[index]
        else:
            raise TypeError("Key must be an integer, tuple, or string")
        
    def __setitem__(self, key, well_object: Well) -> None:
        """
        Set or replace a well in the plate based on its index, coordinate, or name.

        Args:
            key (int, tuple, or str): The identifier for the well to be set or replaced. 
                Can be an integer index, a tuple indicating row and column coordinates, or a string specifying the well's name.
            well_object (Well): The well object to set at the specified key.

        Raises:
            ValueError: If the well_object is not an instance of Well.
            IndexError: If the well index is out of range.
            TypeError: If the key is not a string, integer, or tuple.

        Example:
            >>> plate = Plate(plate_dim=(2, 2))  # Create a small 2x2 plate for simplicity
            >>> new_well = Well(name="C3", plate_id=1, coordinate=(0, 1), metadata={"study_group": "control"})  # Define a new well
            >>> plate[0] = new_well  # Set this well at the first position
            >>> plate[0].name
            'A1'
            >>> plate[0].metadata["study_group"]
            'control'
        """
        if not isinstance(well_object, Well):
            raise ValueError("Value must be an instance of Well")

        if isinstance(key, str):
            index = self._name_to_index_map[key]
            coordinate = self._index_to_coordinates_map[index]
            name = key
        elif isinstance(key, int):
            if key < 0 or key >= len(self.wells):
                raise IndexError("Well index out of range")
            index = key
            coordinate = self._index_to_coordinates_map[index]
            name = self.wells[index].name
        elif isinstance(key, tuple):
            index = self._coordinates_to_index(key)
            coordinate = key
            name = self._index_to_name_map(index)
        else:
            raise TypeError("Key must be a string, integer, or tuple")

        # Update the well object's attributes
        well_object.name = name
        well_object.coordinate = coordinate
        well_object.index = index
        well_object.plate_id = self.plate_id

        # Update the well at the specified index on the plate
        self.wells[index] = well_object
        # Update the name-to-index mapping
        self._name_to_index_map[name] = index

    def __add__(self, other: "Plate") -> "Plate":
        """
        Combine the content of this Plate with another Plate.

        The wells of both plates are combined. If wells at the same coordinates 
        exist in both plates, their metadata is merged.

        Args:
            other (Plate): Another Plate to combine with.

        Returns:
            Plate: A new Plate with combined content from both plates.

        Raises:
            ValueError: If the dimensions of the two plates do not match.

        Example:
            >>> plate1 = Plate(plate_dim=(2, 2))
            >>> plate1.wells[0].metadata = {'sample': 'A'}
            >>> plate2 = Plate(plate_dim=(2, 2))
            >>> plate2.wells[0].metadata = {'volume': 100}
            >>> combined_plate = plate1 + plate2
            >>> combined_plate.wells[0].metadata
            {'sample': 'A', 'volume': 100}
        """
        if (self._n_rows, self._n_columns) != (other._n_rows, other._n_columns):
            raise ValueError("Cannot add plates of different dimensions")

        # Create a new Plate for the combined content
        new_plate = Plate(plate_dim=(self._n_rows, self._n_columns))

        # Iterate through wells and combine metadata
        for (well_self, well_other) in zip(self.wells, other.wells):
            combined_metadata = {**well_self.metadata, **well_other.metadata}
            new_well = Well(name=well_self.name, plate_id=new_plate.plate_id,
                            coordinate=well_self.coordinate, index=well_self.index,
                            rgb_color=well_self.rgb_color, metadata=combined_metadata)
            new_plate.wells[well_self.index] = new_well

        return new_plate

    def _parse_plate_dimensions(self, plate_dim: Union[Tuple[int, int], List[int], Dict[str, int], int]) -> Tuple[int, int]:
        """
        Parse the dimensions of the plate and return the number of rows and columns. This method can handle various 
        formats for specifying the dimensions: as a tuple or list (rows, columns), as a dictionary with 'rows' and 
        'columns' keys, or as an integer representing the total number of wells in a plate. For integer inputs, 
        the method attempts to design a plate with a 2:3 aspect ratio (height to width).

        Args:
            plate_dim (tuple, list, dict, or int): The dimensions of the plate. This can be a tuple or list specifying 
                (rows, columns), a dictionary with 'rows' and 'columns' keys, or an integer specifying the total number 
                of wells, which the method will attempt to fit into a plate with a 2:3 aspect ratio.

        Returns:
            tuple: A tuple containing the number of rows and columns (rows, columns). The method ensures that the 
                resulting plate size can accommodate at least the specified number of wells while trying to maintain 
                the aspect ratio as close to 2:3 as possible.

        Raises:
            ValueError: If the plate_dim format is unsupported or incorrect, or if the number of wells specified by 
                an integer cannot be reasonably fitted into a 2:3 aspect ratio plate.

        Example:
            >>> plate = Plate()
            >>> plate._parse_plate_dimensions((3, 5))
            (3, 5)
            >>> plate._parse_plate_dimensions([4, 6])
            [4, 6]
            >>> plate._parse_plate_dimensions({"rows": 2, "columns": 8})
            (2, 8)
            >>> plate._parse_plate_dimensions(24)  # Assuming 2:3 aspect ratio
            (4, 6)
        """
        if plate_dim is None:
            return self._default_n_rows, self._default_n_columns

        if isinstance(plate_dim, (tuple, list)):
            if len(plate_dim) == 2:
                return plate_dim
            else:
                raise ValueError("Plate dimension must be a tuple or list with two elements (rows, columns).")

        if isinstance(plate_dim, dict):
            return plate_dim.get("rows", self._default_n_rows), plate_dim.get("columns", self._default_n_columns)

        if isinstance(plate_dim, int):
            # Calculate the ideal dimensions for a plate with a 2:3 aspect ratio
            aspect_ratio_width = 3
            aspect_ratio_height = 2

            ideal_height = np.sqrt(plate_dim / (aspect_ratio_width * aspect_ratio_height / aspect_ratio_height**2))
            ideal_width = (aspect_ratio_width / aspect_ratio_height) * ideal_height

            # Round the dimensions and adjust if necessary to accommodate all elements
            rows = int(np.round(ideal_height))
            columns = int(np.round(ideal_width))

            while rows * columns < plate_dim:
                if (rows + 1) * columns <= plate_dim:
                    rows += 1
                elif rows * (columns + 1) <= plate_dim:
                    columns += 1
                else:
                    rows += 1
                    columns += 1

            return rows, columns

        raise ValueError("Unsupported plate format: Must be a tuple, list, dict, or integer.")

    def _coordinates_to_index(self, coordinate: tuple) -> int:
        """
        Convert a well coordinate to its corresponding index in the plate's well list.

        Args:
            coordinate (tuple): The row and column coordinate of the well (row, col).

        Returns:
            int: The index of the well corresponding to the given coordinate.

        Raises:
            IndexError: If the coordinate is out of range of the plate's dimensions.

        Example:
            >>> plate = Plate(plate_dim=(3, 4))  # A 3x4 plate
            >>> plate._coordinates_to_index((0, 0))
            0
            >>> plate._coordinates_to_index((2, 3))
            11
            >>> plate._coordinates_to_index((3, 0))  # This should raise an IndexError
            Traceback (most recent call last):
            ...
            IndexError: Coordinate out of range
        """
        row, col = coordinate
        if row < 0 or row >= self._n_rows or col < 0 or col >= self._n_columns:
            raise IndexError("Coordinate out of range")
        return row * self._n_columns + col
    
    def _to_numpy_array(self, data: list) -> np.ndarray:
        """
        Convert a list of data corresponding to each well into a numpy array matching the plate's layout.

        Args:
            data (list): A list of data values corresponding to each well in the plate.

        Returns:
            numpy.ndarray: A numpy array representing the plate's layout with the provided data.

        Raises:
            Warning: If the number of data elements does not match the plate's size.

        Example:
            # Using a Plate with 4 wells (2x2) for demonstration
            >>> plate = Plate(plate_dim=(2, 2))
            >>> data = [1, 2, 3, 4]  # Sample data corresponding to each well
            >>> array = plate._to_numpy_array(data)
            >>> array.shape
            (2, 2)
            >>> array[0, 1]  # Check the value in the first well (after flipping)
            2
            >>> array[1, 0]  # Check the value in the last well (after flipping)
            3
        """
        # Create an empty array of the right shape
        plate_array = np.empty((self._n_rows, self._n_columns), dtype=object)

        # Check if the data list matches the number of wells
        if len(data) != self.size:
            raise Warning(f"Number of data elements ({len(data)}) does not match the plate's size ({self.size}).")

        # Populate the array with data
        for i, (row, col) in enumerate(self._coordinates):
            plate_array[row, col] = data[i]

        return np.flipud(plate_array)  # Flip to match the physical layout
    
    def summary_dict(self) -> Dict[str, Any]:
        """
        Outputs a summary dictionary with specific details of the plate.

        Returns:
            A dictionary with keys:
            - size: Total size of the plate.
            - dimensions: String representation of plate dimensions.
            - total_number_of_wells_with_sample_code_S: Number of wells with sample_code "S".
            - unique_sample_codes: A list of unique sample codes in the plate.
        """
        summary = {
            "size": self.size,
            "dimensions": f"{self._n_rows}x{self._n_columns}",
            "analytical_samples_capacity": sum(
                1 for well in self.wells if well.metadata.get("sample_code") == "S"
            ),
            "sample_codes": list({
                well.metadata.get("sample_code") for well in self.wells
            }),
        }
        return summary
    
    def get_metadata(self, metadata_key: Optional[str]) -> list:
        """
        Retrieve metadata values for all wells in the plate based on the specified key.

        Args:
            metadata_key (str, optional): The metadata key for which values are to be retrieved. 
                If None, a default value of 'NaN' is returned for each well.

        Returns:
            list: A list of metadata values for each well in the plate.

        Example:
            # Using a Plate with 4 wells and adding metadata for demonstration
            >>> plate = Plate(plate_dim=(2, 2))
            >>> for well in plate.wells:
            ...     well.metadata['sample_type'] = 'RNA'
            >>> plate.get_metadata('sample_type')
            ['RNA', 'RNA', 'RNA', 'RNA']
            >>> plate.get_metadata('non_existing_key')  # Key not present
            ['NaN', 'NaN', 'NaN', 'NaN']
        """
        if metadata_key is None:
            return ["NaN" for _ in self.wells]

        metadata_values = []
        for well in self.wells:
            value = well.get_attribute_or_metadata(metadata_key)
            metadata_values.append(value)

        return metadata_values

    def get_metadata_as_numpy_array(self, metadata_key : str) -> np.ndarray:
        """
        Retrieve metadata values for all wells in a numpy array format based on the specified key.

        Args:
            metadata_key (str): The metadata key for which values are to be retrieved.

        Returns:
            numpy.ndarray: A numpy array representing the metadata values for the plate's layout.

         Example:
            # Using a Plate with 4 wells and adding metadata for demonstration
            >>> plate = Plate(plate_dim=(2, 2))
            >>> for well in plate.wells:
            ...     well.metadata['concentration'] = 10.0
            >>> array = plate.get_metadata_as_numpy_array('concentration')
            >>> array.shape
            (2, 2)
            >>> array[0, 0]  # Value in the first well
            10.0
        """
        metadata = self.get_metadata(metadata_key)

        return self._to_numpy_array(metadata)
    

    def _assign_well_color(self, metadata_key: Optional[str], colormap: str) -> None:
        """
        Assign colors to each well in the plate based on the specified metadata key and colormap.
        
        Args:
            metadata_key (str, optional): The metadata key to use for coloring the wells. 
                If None, a default color is assigned to each well.
        colormap (str): The name of the colormap to use for coloring the wells.
        
        Raises:
            ValueError: If the metadata_key is invalid or not found.
        """

        def is_qualitative_colormap(colormap_name):
            """Check if a given colormap is qualitative."""
            # This list can be expanded with more qualitative colormaps
            qualitative_colormaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 
                                    'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']
            return colormap_name in qualitative_colormaps
        
        if colormap is None:
            colormap = self._default_colormap

        self._metadata_color_map = {}

        if metadata_key is not None:
            
            metadata_values = self.get_metadata(metadata_key)
            unique_values = list(set(metadata_values))
            
            cmap = plt.get_cmap(colormap)

            if is_qualitative_colormap(colormap):
                # Use colors directly for qualitative colormaps
                colors = cmap.colors
                for i, value in enumerate(unique_values):
                    if value is None or value == "NaN":
                        self._metadata_color_map[value] = self._default_well_color
                    else:
                        color_index = i % len(colors)
                        self._metadata_color_map[value] = colors[color_index][0:3]  # RGB color
            else:
                # Use scaling for non-qualitative colormaps
                color_norm = mcolors.Normalize(vmin=0, vmax=len(unique_values) - 1)
                scalar_map = cm.ScalarMappable(norm=color_norm, cmap=cmap)

                for i, value in enumerate(unique_values):
                    if value is None or value == "NaN":
                        self._metadata_color_map[value] = self._default_well_color
                    else:
                        self._metadata_color_map[value] = scalar_map.to_rgba(i)[0:3]  # RGB color

            for well in self.wells:
                metadata_value = well.get_attribute_or_metadata(metadata_key)
                well.rgb_color = self._metadata_color_map.get(metadata_value, self._default_well_color)
        else:
            # Assign default color when metadata_key is None
            for well in self.wells:
                well.rgb_color = self._default_well_color

    def as_records(self) -> List[dict]:
        """
        Convert the plate's well data into a list of dictionaries.

        Each well's attributes are converted into a dictionary, and all these dictionaries
        are compiled into a list, with one dictionary per well.

        Returns:
            list of dict: A list where each element is a dictionary representing a well's attributes.

        Example:
            >>> plate = Plate(plate_dim=(1, 2))
            >>> plate[0].metadata["sample_type"] = "plasma" # set metadata for first well
            >>> records = plate.as_records()
            >>> len(records)  # Number of wells in the plate
            2
            >>> sorted(records[0].keys())  # Show the keys of the first well's dictionary
            ['coordinate', 'empty', 'index', 'name', 'plate_id', 'rgb_color', 'sample_type']
        """
        return [well.as_dict() for well in self]
    
    def as_dict(self) -> dict:
        """
        Converts the Plate object and its contained Well objects into a dictionary.

        Returns:
            dict: A dictionary representation of the Plate object.
        """
        return {
            "plate_id": self.plate_id,
            "n_rows": self._n_rows,
            "n_columns": self._n_columns,
            "wells": [well.as_dict(flat=False) for well in self.wells]  # Ensure Well has an as_dict method
        }
    
    def as_json(self) -> str:
        """
        Serializes the entire Plate object to a JSON string, including all wells. This method
        ensures that the serialized string includes the plate's ID, its dimensions, and a
        full representation of each well within the plate.

        Returns:
            str: A JSON string representation of the plate.

        Example:
            >>> plate = Plate(plate_dim=(2, 2), plate_id=123)
            >>> plate.wells[0].metadata['sample'] = 'Sample A'
            >>> json_str = plate.as_json()
            >>> isinstance(json_str, str) and '"plate_id": 123' in json_str
            True
            >>> '"sample": "Sample A"' in json_str
            True
        """
        
        plate_dict = self.as_dict()  # Use the as_dict method to get a serializable representation

        # Serialize the plate data dictionary to JSON
        return json.dumps(plate_dict, indent=4)
    
    def as_dataframe(self) -> pd.DataFrame:
        """
        Converts the plate data into a Pandas DataFrame.

        Each well and its attributes are represented as a row in the DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame representing the plate's wells and their attributes.
        
        Example:
        >>> plate = Plate()
        >>> df = plate.as_dataframe()
        >>> len(df)
        96
        """
        return pd.DataFrame(self.as_records())
    
    def is_valid_metadata_key(self, key:str) -> bool:
        """
        Check if the provided key is a valid metadata key for the Well instances in the plate.

        This method verifies whether the specified key is either a direct attribute of the Well instances
        or a key within their metadata dictionary.

        Args:
            key (str): The key to check for validity as a metadata key.

        Returns:
            bool: True if the key is a valid metadata key, False otherwise.
        """
        if not key:  # If key is None or empty
            return False

        # Check if the key is a direct attribute or in the metadata dictionary of any well
        for well in self.wells:
            if hasattr(well, key) or key in well.metadata:
                return True

        return False
    
    def as_figure(self, annotation_metadata_key=None, 
                  color_metadata_key=None,
                  fontsize=8,
                  rotation=0,
                  step=10,
                  title_str=None,
                  title_fontsize=14,
                  alpha=0.7,
                  well_size=1200,
                  fig_width=11.69,
                  fig_height=8.27,
                  dpi=100,
                  plt_style="fivethirtyeight",
                  grid_color=(1, 1, 1),
                  edge_color=(0.5, 0.5, 0.5),
                  legend_bb=(0.15, -0.15, 0.7, 1.3),
                  legend_n_columns=6,
                  colormap="tab10",
                  show_grid=True,
                  show_frame=True
                  ) -> 'matplotlib.figure.Figure':
        """
        Create a visual representation of the plate using matplotlib.

        This method generates a figure representing the plate, with options for annotations,
        coloring based on metadata, and various styling adjustments.

        Args:
            annotation_metadata_key (str, optional): Metadata key to use for annotating wells.
            color_metadata_key (str, optional): Metadata key to determine the color of wells.
            fontsize (int, optional): Font size for annotations. Default is 8.
            rotation (int, optional): Rotation angle for annotations. Default is 0.
            step (int, optional): Step size between wells in the grid. Default is 10.
            title_str (str, optional): Title of the figure. If None, a default title is used.
            title_fontsize (str, optional): Font size for title.
            alpha (float, optional): Alpha value for well colors. Default is 0.7.
            well_size (int, optional): Size of the wells in the figure. Default is 1200.
            fig_width (float, optional): Width of the figure. Default is 11.69.
            fig_height (float, optional): Height of the figure. Default is 8.27.
            dpi (int, optional): Dots per inch for the figure. Default is 100.
            plt_style (str, optional): Matplotlib style to use. Default is 'bmh'.
            grid_color (tuple, optional): Color for the grid. Default is (1, 1, 1).
            edge_color (tuple, optional): Color for the edges of wells. Default is (0.5, 0.5, 0.5).
            legend_bb (tuple, optional): Bounding box for the legend. Default is (0.15, -0.15, 0.7, 1.3).
            legend_n_columns (int, optional): Number of columns in the legend. Default is 6.
            colormap (str, optional): Colormap name for coloring wells. Uses default colormap if None.
            show_grid (bool, optional): If True, displays a grid anchored at the well centers; default is True.
            show_grid (bool, optional): If True, plot a rectangle to frame the wells; default is True.

        Returns:
            matplotlib.figure.Figure: A figure object representing the plate.

        Raises:
            ValueError: If provided metadata keys are not valid.
        """
        colormap = colormap if colormap else self._default_colormap

        # Validate metadata keys
        if color_metadata_key and not self.is_valid_metadata_key(color_metadata_key):
            raise ValueError(f"Invalid color_metadata_key: {color_metadata_key}")
        if annotation_metadata_key and not self.is_valid_metadata_key(annotation_metadata_key):
            raise ValueError(f"Invalid annotation_metadata_key: {annotation_metadata_key}")

        # Define title
        if not title_str:
            title_str = f"Plate {self.plate_id}"
            if annotation_metadata_key or color_metadata_key:
                title_str += f", showing {annotation_metadata_key or ''} colored by {color_metadata_key or ''}"

        # Assign colors to wells
        self._assign_well_color(color_metadata_key, colormap)

        # Prepare grid and data for plotting
        minX, maxX, minY, maxY = 0, len(self._columns)*step, 0, len(self._rows)*step
        x = np.arange(minX, maxX, step)
        y = np.arange(minY, maxY, step)

        # Generate grid with columns first (column-major format)
        Xgrid, Ygrid = np.meshgrid(x, y)

        size_grid = np.ones_like(Xgrid) * well_size

        well_colors = np.ravel((self.get_metadata_as_numpy_array("rgb_color")[::-1]))

        # Plot setup
        plt.style.use(plt_style)
        fig = plt.figure(facecolor='white', figsize=(fig_width, fig_height), dpi=dpi,)
        ax = fig.add_subplot(111, facecolor='white')
        # fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi, facecolor="white")
        # Remove the axis lines
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.scatter(Xgrid, Ygrid, s=size_grid, c=well_colors, alpha=alpha, edgecolors=edge_color)

        # Annotations
        if annotation_metadata_key:
            for well in self:
                x_i = Xgrid[well.coordinate]
                y_i = Ygrid[well.coordinate]
                annotation_label = well.get_attribute_or_metadata(annotation_metadata_key)
                ax.annotate(annotation_label, (x_i, y_i), ha='center', va='center', rotation=rotation, fontsize=fontsize, bbox=dict(facecolor='white', alpha=0.5, boxstyle="round,pad=0.25,rounding_size=0.5"))

        # Legends
        if color_metadata_key:
            # Get unique categories and their corresponding colors
            legend_marker_size = np.sqrt(well_size) * 0.5
            unique_categories = set(self.get_metadata(color_metadata_key))
            legend_handles = [plt.Line2D([0], [0], marker='o', color=self._metadata_color_map.get(category, self._default_well_color), alpha=alpha, label=category, markersize=legend_marker_size, linestyle='None') 
                            for category in unique_categories]
            
            ax.legend(handles=legend_handles, bbox_to_anchor=legend_bb, loc='lower center', frameon=False, labelspacing=1, ncol=legend_n_columns)

        # Axis settings
        # Move x-axis ticks to the top
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 

        # Adjust the axis limits to fit the plot tightly
        # Assuming 'step' is the distance between wells
        ax.set_xlim(minX - step/2, maxX - step/2)
        ax.set_ylim(minY - step/2, maxY - step/2)

        # Set x and y tick labels
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in self._columns])
        ax.set_yticks(y)
        ax.set_yticklabels(self.row_labels[::-1])

        # Remove the tick marks but keep the labels
        ax.tick_params(axis='both', length=0) 

        # Grid settings
        if show_grid:
            ax.xaxis.grid(color=grid_color, linestyle='dashed', linewidth=1)
            ax.yaxis.grid(color=grid_color, linestyle='dashed', linewidth=1)
        else:
            ax.xaxis.grid(color=grid_color, linestyle='none',)
            ax.yaxis.grid(color=grid_color, linestyle='none',)

        # ax.set_xlim(minX - maxX*0.08, maxX - maxX*0.035)
        ax.set_ylim(minY - maxY*0.07, maxY - maxY*0.07)

        # # Set tick labels inside the plotting box
        # ax.tick_params(direction='in')

        # # Ugly but works to adjust label padding
        TICK_PADDING = 5
        xticks = [*ax.xaxis.get_major_ticks(), *ax.xaxis.get_minor_ticks()]
        yticks = [*ax.yaxis.get_major_ticks(), *ax.yaxis.get_minor_ticks()]

        for tick in (*xticks, *yticks):
                tick.set_pad(TICK_PADDING)
                
        # ax.set_axisbelow(False)
                # fig.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.15)
        ax.set_title(title_str+"\n", fontsize=title_fontsize)
       

        if show_frame:
           
            x = minX- maxX*0.03  # X-coordinate of the lower-left corner
            y = minY - maxY*0.04 # Y-coordinate of the lower-left corner
            width = maxX*0.975 # Width of the rectangle
            height = maxY*0.955  # Height of the rectangle
            border_radius = 1  # Radius of the rounded corners

            edge_alpha = 0.1
            line_width = 2  

            # Create a rounded rectangle
            rounded_rectangle = FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle=f"round, pad={border_radius}",
                lw=line_width,
                ec=(0, 0, 0,
                edge_alpha),
                fc=(0.95,0.95,0.95),
                zorder=0)

            # Add the rounded rectangle to the axis
            ax.add_patch(rounded_rectangle)

        return fig
    
    def as_plotly_figure(
        self,
        annotation_metadata_key=None, 
        color_metadata_key=None,
        fontsize=14,
        title_str=None,
        title_fontsize=14,
        alpha=0.7,
        well_size=45,  # Adjusted for Plotly marker size
        fig_width=1000,  # Adjusted for Plotly size in pixels
        fig_height=700,  # Adjusted for Plotly size in pixels
        colormap_continuous="Viridis",  # Default colormap in Plotly
        colormap_discrete="D3",  # Default colormap in Plotly
        text_rotation=0,
        show_grid=True,
        theme='plotly',
        dark_mode=False,
        marker_shape='circle'
    ) -> 'plotly.graph_objs._figure.Figure':
        """
        Generates a Plotly scatter plot representing the data of a biological plate.

        This function takes various parameters for customization of the plot such as colors, 
        font sizes, title, and dimensions. It handles both continuous and discrete data types 
        for coloring and allows annotations on each point in the scatter plot.

        Args:
            annotation_metadata_key (str, optional): Metadata key for annotations. 
                Default is None.
            color_metadata_key (str, optional): Metadata key for color mapping.
                Default is None.
            fontsize (int): Font size for annotations. Default is 14.
            title_str (str, optional): Title of the plot. Default is None.
            title_fontsize (int): Font size for the plot title. Default is 14.
            alpha (float): Opacity level for markers. Default is 0.7.
            well_size (int): Marker size. Default is 45.
            fig_width (int): Width of the figure in pixels. Default is 1000.
            fig_height (int): Height of the figure in pixels. Default is 700.
            colormap_continuous (str): Colormap for continuous data. Default is "Viridis".
            colormap_discrete (str): Colormap for discrete data. Default is "D3".
            text_rotation (int): Rotation angle of text annotations. Default is 0.
            show_grid (bool): Whether to show grid lines. Default is True.
            theme (str): Plotly theme. Default is 'plotly'.

        Returns:
            plotly.graph_objs._figure.Figure.Figure: A Plotly scatter plot figure.

        Example:

        ```python
        plate = Plate()
        fig = plate.as_plotly_figure(
            annotation_metadata_key='gene_name',
            color_metadata_key='expression_level',
            fontsize=12,
            title_str='Gene Expression Levels',
            title_fontsize=16,
            alpha=0.8,
            well_size=50,
            fig_width=1200,
            fig_height=800,
            colormap_continuous="Plasma",
            text_rotation=45,
            show_grid=False,
            theme='plotly_dark'
        )
        fig.show()
        ```

        This example generates a scatter plot with gene names as annotations, colors representing
        expression levels, customized font sizes, and a dark theme.

        """
         # Transform the plate data into a DataFrame for easier manipulation
        df = self.as_dataframe()

        if dark_mode:
            annotation_bg_color = 'rgba(10, 10, 10, 0.5)'
            # annotation_font_color = "black"
        else:
            annotation_bg_color = 'rgba(255, 255, 255, 0.5)'

        # Default values if parameters are not provided
        if annotation_metadata_key is None:
            annotation_metadata_key = 'name'
        if color_metadata_key is None:
            color_metadata_key = 'white'

        if color_metadata_key == 'white':
            df[color_metadata_key] = 'white' 


        # Calculate the maximum size for each well
        # Assuming margins are set or default
        margins = dict(l=50, r=50, t=50, b=50, pad=4)  # Default margins, update if changed in your layout
        available_width = fig_width - margins['l'] - margins['r']
        available_height = fig_height - margins['t'] - margins['b']
        
        # Calculate space per well
        space_per_well_x = available_width / self._n_columns
        space_per_well_y = available_height / self._n_rows

        # Set well size to be the minimum of the two, with a certain scaling factor
        scaling_factor = 0.8  # Adjust this factor as needed
        well_size = min(space_per_well_x, space_per_well_y) * scaling_factor

        # Modify the plot based on marker_shape
        marker_symbol = 'square' if marker_shape == 'square' else 'circle'

        # # Calculate the grid dimensions
        step = 1 
       
         # Calculate the axis limits
        x_axis_min = -0.5 * step
        x_axis_max = self._n_columns * step - 0.5 * step
        y_axis_min = -0.5 * step
        y_axis_max = self._n_rows * step - 0.5 * step

        # Generate grid data for plotting, assuming equal spacing between wells
        x = np.arange(0, len(self._columns)*step, step)
        y = np.arange(0, len(self._rows)*step, step)
        Xgrid, Ygrid = np.meshgrid(x, y)

        # Convert coordinate tuples to separate columns for x and y
        df['column'] = df['coordinate'].apply(lambda c: step*c[1])
        df['row'] = df['coordinate'].apply(lambda c: step*c[0])

        # hover_data = ["name"] + list(plate[0].metadata.keys())
        hover_data = ["name"] + list(self[0].metadata.keys())

        # Determine color scale and plot type based on the data type of color_metadata_key
        if df[color_metadata_key].dtype.kind in 'ifc':  # Numeric data - continuous
            color_scale = colormap_continuous
            fig = px.scatter(
                df,
                x='column',
                y='row',
                hover_data=hover_data,
                color=color_metadata_key,
                color_continuous_scale=color_scale,
                # other parameters...
            )
        else:  # Categorical data - discrete
            discrete_color_sequence = px.colors.qualitative.__getattribute__(colormap_discrete)
            fig = px.scatter(
                df,
                x='column',
                y='row',
                hover_data=hover_data,
                color=color_metadata_key,
                color_discrete_sequence=discrete_color_sequence,
                # other parameters...
            )

        # Add annotations to each well in the plate
        for well in self:
            fig.add_annotation(
                x=Xgrid[well.coordinate],
                y=Ygrid[well.coordinate],
                text=str(well.get_attribute_or_metadata(annotation_metadata_key)),
                textangle= -1*text_rotation,
                showarrow=False,
                # font=dict(size=fontsize),
                bgcolor=annotation_bg_color,
                borderpad=2,
                bordercolor=annotation_bg_color
            )

        fig.update_traces(
            marker=dict(
                size=well_size,
                line=dict(width=2),
            opacity=alpha,
            symbol=marker_symbol,
            ),
            selector=dict(mode='markers')
        )
        
        # Adjust plot layout, axes, and other visual elements
        fig.update_layout(
            title=dict(text=title_str, font_size=title_fontsize),
            width=fig_width,
            height=fig_height,
            xaxis=dict(
                title="",
                showgrid=show_grid, 
                zeroline=False, 
                showticklabels=True, 
                tickmode="array",
                tickvals=list(range(0, step*self._n_columns, step)),
                ticktext=self.column_labels,
                side="top",
                tickfont=dict(size=18),
                range=[x_axis_min, x_axis_max]
            ),
            yaxis=dict(
                title="",
                showgrid=show_grid, 
                zeroline=False, 
                showticklabels=True, 
                tickmode="array",
                tickvals=list(range(0, step*step*self._n_rows, step)),
                ticktext=self.row_labels[::-1],
                tickfont=dict(size=18),
                range=[y_axis_min, y_axis_max]
            ),
            template=theme,
            legend=dict(
                orientation="h",  # Horizontal orientation
                yanchor="bottom",
                y=-0.1,  # Adjust this value to move the legend up or down
                xanchor="center",
                x=0.5
            ),
            margin=margins,
        )

        # # Make the layout responsive
        # fig.update_layout(
        #     autosize=True,
        #     margin=dict(l=50, r=50, t=50, b=50, pad=4),  # Adjust margins as needed
        #     # Remove fixed width and height, or set them to None
        #     width=None,
        #     height=None
        # )

        return fig
    
    def to_file(self, file_path : str = None,
                file_format : str = "csv",
                metadata_keys : list = []) -> None:
        """
        Write the plate data to a file in the specified format.

        The method supports various file formats such as CSV, TSV, and Excel. It allows 
        selection of specific metadata keys to be included in the output. If no file path 
        is specified, the file is saved in the current working directory with a default 
        name based on the plate ID.

        Args:
            file_path (str, optional): The path where the file will be saved. 
                If not specified, the file is saved in the current working directory.
            file_format (str, optional): The format of the file ('csv', 'tsv', 'xls').
            metadata_keys (list, optional): A list of metadata keys to include in the file. 
                If empty, all metadata except those in _default_exclude_metadata are included.

        Raises:
            ValueError: If an unsupported file format is specified.
        """
        
        if file_path is None:
            file_name = f"plate_{self.plate_id}.{file_format}"
            file_path = Path.cwd() / file_name
        else:
            file_path = Path(file_path)
            if file_path.is_dir():
                file_name = f"plate_{self.plate_id}.{file_format}"
                file_path = file_path / file_name
            else:
                if file_path.suffix == "":
                    file_path = file_path.with_suffix(f".{file_format}")
                else:
                    file_format = file_path.suffix.lstrip('.')
        
        logger.info(f"Writing to file:\n\t{file_path}")

        df = self.as_dataframe()

        if len(metadata_keys) > 0:
            df = df[metadata_keys]
        else:  # use all metadata except those in default_exclude_metadata
            df = df.drop(columns=self._default_exclude_metadata)
        
        match file_format:
            case "csv":
                df.to_csv(file_path, index=False)

            case "tsv":
                df.to_csv(file_path, sep="\t", index=False)

            case "xls":
                df.to_excel(file_path, index=False)
        
    def add_metadata(self, key, values) -> None:
        """
        Add or update metadata for all wells in the plate. If a list of values is provided,
        assign each value to the corresponding well. If a single value is provided, assign it to all wells.

        Args:
            key (str): The metadata key to add or update.
            values: A single value or a list of values to set for the given metadata key. 

        Example:
            >>> plate = Plate(plate_dim=(2, 2))
            >>> plate.add_metadata('sample_type', ['RNA', 'DNA', 'RNA', 'DNA'])
            >>> [well.metadata['sample_type'] for well in plate.wells]
            ['RNA', 'DNA', 'RNA', 'DNA']
            >>> plate.add_metadata('study', 'oncology')
            >>> all(well.metadata['study'] == 'oncology' for well in plate.wells)
            True

        """
        if isinstance(values, list):
            # Case when values is a list
            if len(values) != len(self.wells):
                raise ValueError("The length of values list does not match the number of wells")

            for well, value in zip(self.wells, values):
                well.metadata[key] = value
        else:
            # Case when a single value is provided
            for well in self.wells:
                well.metadata[key] = values

    @property
    def row_labels(self) -> list:
        """
        Get the row labels for the plate.

        This property generates a list of alphabetical characters representing the row labels
        of the plate, based on the number of rows in the plate.

        Returns:
            list: A list of strings, each representing a row label.

        Example:
            >>> plate = Plate(plate_dim=(8, 12))  # A standard 96-well plate
            >>> plate.row_labels
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        """
        return list(string.ascii_uppercase)[:len(self._rows)]
    
    @property
    def column_labels(self) -> list:
        """
        Get the column labels for the plate.

        This property generates a list of numerical strings representing the column labels
        of the plate, based on the number of columns in the plate.

        Returns:
            list: A list of strings, each representing a column label.

        Example:
            >>> plate = Plate(plate_dim=(8, 12))  # A standard 96-well plate
            >>> plate.column_labels
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        """
        return [str(row_id+1) for row_id in self._columns]
    
    @property
    def capacity(self) -> int:
        """
        Get the number of samples that can be added to the plate, which is the same as the number of wells in this class

        Example:
            >>> plate = Plate(plate_dim=(8, 12))  # A standard 96-well plate
            >>> plate.capacity
            96
        """
        return self.size
    
    @property
    def plate_id(self) -> int:
        """
        Get the plate ID.

        This property returns the unique identifier of the plate.

        Returns:
            int: The plate ID.

        Example:
            >>> plate = Plate()
            >>> plate.plate_id
            1
        """
        return self._plate_id

    @plate_id.setter
    def plate_id(self, new_id) -> None:
        """
        Set a new plate ID.

        This method updates the plate ID and propagates the change to all the wells 
        within the plate.

        Args:
            new_id (int): The new plate ID to be set.

        Example:
            >>> plate = Plate()
            >>> plate.plate_id = 2
            >>> plate.plate_id
            2
        """
        self._plate_id = new_id
        for well in self.wells:
            well.plate_id = new_id

    @staticmethod    
    def create_index_coordinates(rows, columns) -> list:
        """
        Static method to create a list of index coordinates for the wells in a plate.

        The method generates a grid of coordinates, counting from left to right, 
        starting at the well in the top left. It is used to map the wells to their 
        respective positions in the plate.

        Args:
            rows (iterable): An iterable representing the rows of the plate.
            columns (iterable): An iterable representing the columns of the plate.

        Returns:
            list: A list of tuples, each representing the (row, column) index of a well.

        Example:
            >>> Plate.create_index_coordinates(range(2), range(2))
            [(1, 0), (1, 1), (0, 0), (0, 1)]
        """
        # count from left to right, starting at well in top left
        return list(itertools.product(
                                    range(len(rows)-1, -1, -1),
                                    range(0, len(columns))
                                    )
                )
    
    @staticmethod
    def create_alphanumerical_coordinates(rows:list, columns: list) ->  list:
        """
        Static method to create alphanumerical coordinates for the wells.

        Args:
            rows (list): A list of row indices.
            columns (list): A list of column indices.

        Returns:
            list: A list of alphanumerical coordinates (e.g., "A1", "B2").

        Example:
            >>> Plate.create_alphanumerical_coordinates([0, 1], [0, 1, 2])
            ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']
            >>> Plate.create_alphanumerical_coordinates([0], [0, 1])
            ['A1', 'A2']
        """
        row_labels = list(string.ascii_uppercase)[:len(rows)]
        return [f"{row_labels[row]}{col+1}" for row, col in itertools.product(rows, columns)]
    

class SamplePlate(Plate):
    _default_sample_code : str = "S"
    _default_sample_name : str = "Specimen"

    def __init__(self, plate_dim: Tuple[int, int] | List[int] | Dict[str, int] | int = None, plate_id: int = 1):
        super().__init__(plate_dim, plate_id)

        for well in self.wells:
            well.metadata["sample_code"] = self._default_sample_code
            well.metadata["sample_name"] = self._default_sample_name

    def as_plotly_figure(
        self,
        annotation_metadata_key="sample_code",  # Changed default value
        color_metadata_key="sample_code",  # Changed default value
        fontsize=14,
        title_str=None,
        title_fontsize=14,
        alpha=0.7,
        well_size=45,  # Adjusted for Plotly marker size
        fig_width=1000,  # Adjusted for Plotly size in pixels
        fig_height=700,  # Adjusted for Plotly size in pixels
        colormap_continuous="Viridis",  # Default colormap in Plotly
        colormap_discrete="D3",  # Default colormap in Plotly
        text_rotation=0,
        show_grid=True,
        theme='plotly',
        dark_mode=False,
        marker_shape='circle'
    ):
        """
        Generates a Plotly figure that visualizes the plate and optional metadata 

        This method overrides the as_plotly_figure() from the Plate class to provide other defaults for annotaion and color based on QC and sample codes
        """
        # Call the superclass method with possibly modified default values
        # Here, if annotation_metadata_key or color_metadata_key are not provided in the call,
        # it uses the new defaults specified above
        return super().as_plotly_figure(
            annotation_metadata_key=annotation_metadata_key,
            color_metadata_key=color_metadata_key,
            fontsize=fontsize,
            title_str=title_str,
            title_fontsize=title_fontsize,
            alpha=alpha,
            well_size=well_size,
            fig_width=fig_width,
            fig_height=fig_height,
            colormap_continuous=colormap_continuous,
            colormap_discrete=colormap_discrete,
            text_rotation=text_rotation,
            show_grid=show_grid,
            theme=theme,
            dark_mode=dark_mode,
            marker_shape=marker_shape
        )

    
# A plate with QC samples is a subclass of a Plate class
class QCPlate(Plate):
    """_summary_
    Class that represents a multiwell plate where some wells can 
    contain quality control samples according to the scheme defined 
    in QC_config; either a <config_file.toml> file or a dict following the same structure

    Args:
        Plate (_type_): _description_
    """
    
    _non_qc_sample_code : str = "S"
    _non_qc_sample_name : str = "Specimen"
    
    def __init__(self, QC_config = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if QC_config is not None:
            
            if isinstance(QC_config, dict):
                self.config = QC_config
            else:                                        
                self.config = self.load_config_file(QC_config)
            
            if self.config is not None: 
                self.create_QC_plate_layout()
                
            else:
                logger.error(f"No scheme for QC samples provided.")
            
    
    def __repr__(self):
        return f"{self.__class__.__name__}(({len(self._rows)},{len(self._columns)}), plate_id={self.plate_id})"
    
    def __str__(self):
        plate_summary = f"Plate ID: {self.plate_id}\n"
        plate_summary += f"Dimensions: {self._n_rows} rows x {self._n_columns} columns\n"
        plate_summary += "Plate Layout (Sample Codes):\n"
        plate_array_str = np.array_str(self.get_metadata_as_numpy_array("sample_code"))
        plate_summary += plate_array_str
        return plate_summary
    
    def define_unique_QC_sequences(self):
        """ Sets up the unique QC sequences for each round based on the new config structure. """
        logger.debug("Setting up dynamic QC scheme from config file")

        # Initialize variables
        total_wells = self.size
        qc_round_frequency = self.config['QC']['run_QC_after_n_specimens']
        max_rounds = total_wells // qc_round_frequency

        # Step 1: Initialize sequence map
        self.qc_sequence_map = {round_num: [] for round_num in range(1, max_rounds + 1)}

        # Step 2: Apply specific round patterns
        for key, value in self.config['QC']['patterns'].items():
            if key.startswith('round_'):
                round_number = int(key.split('_')[1])
                self.qc_sequence_map[round_number] = value

        # Step 3: Apply repeat pattern
        if 'repeat' in self.config['QC']['patterns']:
            repeat_config = self.config['QC']['patterns']['repeat_pattern']
            pattern, times = repeat_config['pattern'], repeat_config['times']
            for i in range(1, times + 1):
                if not self.qc_sequence_map[i]:
                    self.qc_sequence_map[i] = pattern

        # Step 4: Apply alternating patterns
        if 'alternating' in self.config['QC']['patterns']:
            alternating_patterns = self.config['QC']['patterns']['alternating']
            alt_index = 0
            for round_num in range(1, max_rounds + 1):
                if not self.qc_sequence_map[round_num]: 
                    self.qc_sequence_map[round_num] = alternating_patterns[alt_index % len(alternating_patterns)]
                    alt_index += 1


        # Log the defined sequences
        for round_number, sequence in self.qc_sequence_map.items():
            logger.debug(f"Round {round_number}: {sequence}")

    def create_QC_plate_layout(self):
        """
        Creates the plate layout with QC and specimen samples based on the configuration provided.

        This method initializes the QC sample placement according to the unique QC sequences defined for each round.
        It iterates over all the wells in the plate, placing QC samples at the specified intervals and filling the
        rest with specimen samples. The method handles the transition between different rounds of QC samples and ensures
        that each well is assigned the correct sample type metadata.

        The process accounts for special configurations such as starting the plate with a QC round and adjusts the
        placement of QC and specimen samples accordingly. If the iterator of QC samples for a given round is exhausted,
        the method transitions to the next round's sequence of QC samples.

        Attributes:
            None directly used, but utilizes class attributes such as self.config and self.size which are set during initialization.

        Raises:
            StopIteration: An exception is caught to indicate the end of a QC sample sequence for a round,
                        triggering the transition to the next round or switching back to specimen sample assignment.
        """
        logger.info("Creating dynamic plate layout with QC samples.")

        self.define_unique_QC_sequences()

        # Initialize counters for QC sample types and control variables for round and specimen handling
        counts = {qc_type: 0 for qc_type in self.config["QC"]["names"].keys()}
        round_counter = 1
        specimen_counter = 0
        qc_round_frequency = self.config['QC']['run_QC_after_n_specimens']
        start_with_qc = self.config['QC']['start_with_QC_round']
        current_round_qc_samples = iter(self.qc_sequence_map.get(round_counter, []))


        # Start and End patterns have highest priority
        # +++++++++++++++++++++++++++++++++++++++++++++++++++
        # Pre-allocate wells for 'start' pattern
        start_pattern = self.config['QC']['patterns'].get('start', [])
        for i, qc_sample in enumerate(start_pattern):
            self.assign_qc_sample_metadata(i, qc_sample, counts)
        
        start_well_offset = len(start_pattern)
        end_pattern = self.config['QC']['patterns'].get('end', [])
        end_well_offset = len(end_pattern)

        # Pre-allocate wells for 'end' pattern at the end of the plate
        for i, qc_sample in enumerate(end_pattern):
            self.assign_qc_sample_metadata(self.size - end_well_offset + i, qc_sample, counts)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++
        
        current_round_qc_samples = iter(self.qc_sequence_map.get(round_counter, []))

        # Adjust the loop to start and end accounting for 'start' and 'end' patterns
        # for well_index in range(0, self.size - end_well_offset):
        for well_index in range(start_well_offset, self.size - end_well_offset):
        # for well_index in range(self.size):
            # Handle the initial placement of QC samples if the configuration specifies starting with a QC round
            if start_with_qc and round_counter == 1:
                try:
                    # Attempt to place a QC sample for the first round
                    qc_sample = next(current_round_qc_samples)
                    self.assign_qc_sample_metadata(well_index, qc_sample, counts)
                    continue # Skip to the next iteration to continue placing QC samples
                except StopIteration:
                    # If no more QC samples are available for the current round, transition to the next round
                    round_counter += 1
                    current_round_qc_samples = iter(self.qc_sequence_map.get(round_counter, []))

            # Check if it's time to place a QC sample based on the specified frequency
            if specimen_counter >= qc_round_frequency:
                try:
                    # Place a QC sample and reset the specimen counter for the next sequence
                    qc_sample = next(current_round_qc_samples)
                    self.assign_qc_sample_metadata(well_index, qc_sample, counts)
                except StopIteration:
                    # Transition to the next round of QC samples if available, or continue with specimen placement
                    round_counter += 1
                    specimen_counter = 0 # Reset specimen counter as we're starting a new QC round or specimen sequence
                    current_round_qc_samples = iter(self.qc_sequence_map.get(round_counter, []))
                    # Place a specimen sample immediately if QC samples for the new round are exhausted or not defined
                    self.assign_specimen_sample_metadata(well_index, specimen_counter)
                    specimen_counter += 1
            else:
                # Place a specimen sample and increment the counter
                self.assign_specimen_sample_metadata(well_index, specimen_counter)
                specimen_counter += 1

        # Log the final layout for debugging
        for well in self.wells:
            logger.debug(f"Well {well.name}: {well.metadata}")

    def assign_qc_sample_metadata(self, well_index, qc_sample, counts: dict):
        self.wells[well_index].metadata["QC"] = True
        sample_code = qc_sample
        counts[sample_code] += 1
        self.wells[well_index].metadata["sample_code"] = sample_code
        self.wells[well_index].metadata["sample_type"] = self.config["QC"]["names"][sample_code]
        self.wells[well_index].metadata["sample_name"] = f"{sample_code}{counts[sample_code]}"

    def assign_specimen_sample_metadata(self, well_index, count):
        self.wells[well_index].metadata["QC"] = False
        self.wells[well_index].metadata["sample_code"] = self._non_qc_sample_code
        self.wells[well_index].metadata["sample_type"] = self._non_qc_sample_name
        self.wells[well_index].metadata["sample_name"] = f"{self._non_qc_sample_code}{count + 1}"

    def as_plotly_figure(
        self,
        annotation_metadata_key="sample_code",  # Changed default value
        color_metadata_key="sample_code",  # Changed default value
        fontsize=14,
        title_str=None,
        title_fontsize=14,
        alpha=0.7,
        well_size=45,  # Adjusted for Plotly marker size
        fig_width=1000,  # Adjusted for Plotly size in pixels
        fig_height=700,  # Adjusted for Plotly size in pixels
        colormap_continuous="Viridis",  # Default colormap in Plotly
        colormap_discrete="D3",  # Default colormap in Plotly
        text_rotation=0,
        show_grid=True,
        theme='plotly',
        dark_mode=False,
        marker_shape='circle'
    ):
        """
        Generates a Plotly figure that visualizes the plate and optional metadata 

        This method overrides the as_plotly_figure() from the Plate class to provide other defaults for annotaion and color based on QC and sample codes
        """
        # Call the superclass method with possibly modified default values
        # Here, if annotation_metadata_key or color_metadata_key are not provided in the call,
        # it uses the new defaults specified above
        return super().as_plotly_figure(
            annotation_metadata_key=annotation_metadata_key,
            color_metadata_key=color_metadata_key,
            fontsize=fontsize,
            title_str=title_str,
            title_fontsize=title_fontsize,
            alpha=alpha,
            well_size=well_size,
            fig_width=fig_width,
            fig_height=fig_height,
            colormap_continuous=colormap_continuous,
            colormap_discrete=colormap_discrete,
            text_rotation=text_rotation,
            show_grid=show_grid,
            theme=theme,
            dark_mode=dark_mode,
            marker_shape=marker_shape
        )

    @property
    def capacity(self):
        # number of non-QC samples that can be added to the plate - TODO change name?
        return sum([not well.metadata["QC"] for well in self.wells])

    @staticmethod
    def load_config_file(config_file: str = None) -> dict:
        
        # READ CONFIG FILE
        if config_file is None: 
            
            logger.warning("No config file specified. Trying to find a toml file in current folder.")
            
            config_file_search = glob.glob("*.toml")      
            
            if config_file_search:
                config_file = config_file_search[0]
                logger.info(f"Using toml file '{config_file}'")
        
        try:
            with open(config_file, mode="rb") as fp:
                config = tomli.load(fp)
            
            logger.info(f"Successfully loaded config file {config_file}")
            logger.debug(f"{config}")
            
            return config
        
        except FileNotFoundError:
            logger.error(f"Could not find/open config file {config_file}")
            
            raise FileExistsError(config_file)      


class PlateFactory:

    @staticmethod
    def validate_qc_scheme(scheme: Union[str, Dict]) -> Dict:
        """
        Validates the QC scheme configuration. If a file path is provided, the method
        reads and validates the TOML configuration file. If a dictionary is provided,
        it directly validates the configuration.

        Validation checks include:
        - Presence of essential sections and fields.
        - Consistency of QC sample names across sections.
        - Format and validity of specified patterns.

        Args:
            scheme (Union[str, Dict]): Path to the QC scheme TOML file or the scheme as a dictionary.

        Returns:
            Dict: The validated and parsed QC scheme configuration.

        Raises:
            FileNotFoundError: If the TOML file does not exist.
            ValueError: If the configuration is invalid.
        """
        # Load configuration from file or use the provided dict
        config = scheme
        if isinstance(scheme, str):
            scheme_path = Path(scheme)
            if not scheme_path.exists():
                raise FileNotFoundError(f"The configuration file '{scheme}' does not exist.")
            with scheme_path.open('rb') as f:
                config = tomli.load(f)

        # Basic structure validation
        if "QC" not in config or "patterns" not in config["QC"] or "names" not in config["QC"]:
            raise ValueError("Invalid QC scheme configuration: Missing required sections.")

        # Validate QC names
        qc_names = config["QC"]["names"]
        if not isinstance(qc_names, dict) or not qc_names:
            raise ValueError("Invalid QC names configuration.")

       # Validate patterns using QC names
        patterns = config["QC"].get("patterns", {})
        for pattern_name, pattern_value in patterns.items():
            if isinstance(pattern_value, list):
                # Check if the list contains lists (for patterns like 'alternating')
                if pattern_value and isinstance(pattern_value[0], list):
                    for sample_list in pattern_value:
                        for sample_name in sample_list:
                            if sample_name not in qc_names:
                                raise ValueError(f"Undefined QC sample name '{sample_name}' in pattern '{pattern_name}'.")
                else:
                    # Validate each sample name in the list (for patterns like 'round_1')
                    for sample_name in pattern_value:
                        if sample_name not in qc_names:
                            raise ValueError(f"Undefined QC sample name '{sample_name}' in pattern '{pattern_name}'.")
            elif isinstance(pattern_value, dict) and 'pattern' in pattern_value and 'times' in pattern_value:
                # Validate repeat pattern format
                if not isinstance(pattern_value['pattern'], list) or not isinstance(pattern_value['times'], int):
                    raise ValueError(f"Invalid format for repeat pattern '{pattern_name}'.")
                for sample_name in pattern_value['pattern']:
                    if sample_name not in qc_names:
                        raise ValueError(f"Undefined QC sample name '{sample_name}' in repeat pattern '{pattern_name}'.")
            else:
                raise ValueError(f"Invalid pattern format for '{pattern_name}'.")

        return config

    @staticmethod
    def create_plate(*args, **kwargs) -> Plate:
        """
        Creates a plate object, deciding on the specific type of plate (SamplePlate or QCPlate)
        based on the presence of a 'QC_config' argument.

        If 'QC_config' is provided and not None, a QCPlate is created with the given QC configuration.
        Otherwise, a SamplePlate is created. The method dynamically selects the appropriate constructor
        based on the provided arguments.

        Args:
            *args: Positional arguments passed directly to the plate's constructor.
            **kwargs: Keyword arguments passed directly to the plate's constructor. If 'QC_config' is
                    among these keyword arguments and is not None, a QCPlate is instantiated. Otherwise,
                    a SamplePlate is instantiated.

        Returns:
            Plate: An instance of either SamplePlate or QCPlate, depending on the provided arguments.

        Raises:
            Exception: If the QC scheme validation fails.

        Examples:
            >>> sample_plate = PlateFactory.create_plate(plate_dim=(8, 12))
            >>> isinstance(sample_plate, SamplePlate)
            True

            >>> # Example QC configuration for testing purposes
            >>> qc_config = {
                        'QC': {
                            'start_with_QC_round': False,
                            'run_QC_after_n_specimens': 11,
                            'names': {
                                'EC': 'EC: External_Control_(matrix)',
                                'PB': 'PB: Paper_Blank',
                                'PO': 'PO: Pooled_specimens'
                            },
                            'patterns': {
                                'alternating': [['EC', 'PB'], ['EC', 'PO']],
                            }
                        }
                    }
            >>> qc_plate = PlateFactory.create_plate(plate_dim=(8, 12), QC_config=qc_config)
            >>> isinstance(qc_plate, QCPlate)
            True

            >>> # Creating a plate without specifying 'plate_dim', default dimensions should be used
            >>> default_plate = PlateFactory.create_plate()
            >>> isinstance(default_plate, SamplePlate)
            True

        """
        if 'QC_config' in kwargs:
            try:
                qc_config = PlateFactory.validate_qc_scheme(kwargs['QC_config'])
                # Replace the original QC_config with the validated version
                kwargs['QC_config'] = qc_config
                return QCPlate(*args, **kwargs)
            except (FileNotFoundError, ValueError) as e:
                # Issue a warning to the user
                warnings.warn(f"Failed to validate QC scheme: {e}")
                # Remove the 'QC_config' key from kwargs if validation fails
                kwargs.pop('QC_config', None)  # Safely remove 'QC_config' without causing KeyError
                return SamplePlate(*args, **kwargs)
        else:
            return SamplePlate(*args, **kwargs)
        
    @staticmethod
    def dict_to_plate(plate_data: dict) -> Plate:
        """
        Deserializes a dictionary back into a Plate object, deciding on the specific type
        of plate (Plate, SamplePlate, or QCPlate) based on the presence of QC metadata.

        The method inspects the dictionary for keys or structures indicative of QC metadata.
        If found, it returns an instance of QCPlate; otherwise, it defaults to returning a
        SamplePlate or a generic Plate, depending on the nature of the data provided.

        Args:
            plate_data (dict): The dictionary representation of a plate.

        Returns:
            Plate: The deserialized Plate object.

        Examples:
            >>> plate_data_sample = {
                    "plate_id": 1,
                    "n_rows": 8,
                    "n_columns": 12,
                    "wells": [{"name": "A1", "metadata": {"sample_code": "S1"}}]
                }
            >>> plate_sample = PlateFactory.dict_to_plate(plate_data_sample)
            >>> isinstance(plate_sample, SamplePlate)
            True

            >>> plate_data_qc = {
                    "plate_id": 2,
                    "n_rows": 8,
                    "n_columns": 12,
                    "wells": [{"name": "A1", "metadata": {"QC": True, "sample_code": "QC1"}}],
                    "QC_config": {"QC": {"patterns": {"start": ["QC1"], "end": ["QC2"]}, "names": {"QC1": "Quality Control 1", "QC2": "Quality Control 2"}}}
                }
            >>> plate_qc = PlateFactory.dict_to_plate(plate_data_qc)
            >>> isinstance(plate_qc, QCPlate)
            True
        """
        # Check if the plate data contains QC metadata
        contains_qc_metadata = any("QC" in well.get("metadata", {}) for well in plate_data.get("wells", []))

        # Determine the plate dimensions
        n_rows = plate_data.get("n_rows", Plate._default_n_rows)
        n_columns = plate_data.get("n_columns", Plate._default_n_columns)
        plate_id = plate_data.get("plate_id", 1)

        if contains_qc_metadata:
            # Instantiate a QCPlate with dimensions and ID but without initializing wells in __init__ method.
            plate = QCPlate(plate_dim=(n_rows, n_columns), plate_id=plate_id)

            # Assuming the presence of a 'QC_config' in plate_data to configure QCPlate
            # If 'QC_config' is not directly available, the method or an attribute could be adapted to derive it from the well metadata.
            if "QC_config" in plate_data:
                qc_config = plate_data["QC_config"]
                plate.config = PlateFactory.validate_qc_scheme(qc_config)
                plate.create_QC_plate_layout()
        else:
            # Default to SamplePlate if no specific QC configuration is detected
            plate = SamplePlate(plate_dim=(n_rows, n_columns), plate_id=plate_id)

        # Clear existing wells and repopulate from the dictionary data
        plate.wells.clear()

        # Deserialize each well using the appropriate method
        for well_data in plate_data.get("wells", []):
            well = Well.dict_to_well(well_data)
            plate.wells.append(well)

        return plate
    

# @dataclass
# class SampleWell(Well):
#     """
#     A class to represent a well in a multiwell plate that contains a biological sample.

#     sample_well = SampleWell(
#             name="B2", plate_id=2, coordinate=(1, 1), index=5, rgb_color=(0.5, 0.5, 0.5),
#             sample_type="Blood", qc_type="Standard", sample_volume=50.0
#         )
#     """
#     sample_type: str = ""
#     qc_type: str = "" 
#     sample_volume: float = 0.0 


# class CompoundWell(Well):
#     def __init__(self, compounds: Dict[str, float] = None, volume: float = 0.0):

#         self.volume = volume
#         self.compounds = compounds if compounds else {}  # Compound names with their amounts in weight

#     def add_compound(self, compound_name: str, amount: float):
#         self.compounds[compound_name] = self.compounds.get(compound_name, 0) + amount

#     def take(self, volume: float):
#         if volume > self.volume:
#             raise ValueError("Not enough volume to take")
        
#         taken_compounds = {name: (amount * volume / self.volume) for name, amount in self.compounds.items()}
#         self.volume -= volume
#         for compound in self.compounds:
#             self.compounds[compound] -= taken_compounds[compound]

#         return CompoundWell(compounds=taken_compounds, volume=volume)

#     def add(self, other_well):
#         if not isinstance(other_well, CompoundWell):
#             raise ValueError("Can only add from another CompoundWell")
        
#         for compound, amount in other_well.compounds.items():
#             self.add_compound(compound, amount)
        
#         self.volume += other_well.volume

#     def get_concentration(self, compound_name: str) -> float:
#         """
#         Calculate and return the concentration of a specific compound in the well.
#         Concentration is returned in mg/mL assuming the volume is in µL and the amount is in mg.
#         """
#         if compound_name not in self.compounds:
#             raise ValueError(f"Compound {compound_name} not found in the well.")
        
#         # Convert the volume from µL to mL for concentration calculation
#         volume_ml = self.volume / 1000
#         if volume_ml == 0:
#             raise ValueError("Volume is zero, cannot calculate concentration.")

#         return self.compounds[compound_name] / volume_ml
        

# class ChemicalPlate(BasicPlate):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.wells = [CompoundWell(name=f"{row}{col+1}", 
#                                    coordinate=(row, col), 
#                                    index=row * self._n_columns + col, 
#                                    plate_id=self.plate_id, 
#                                    rgb_color=self._NaN_color)
#                       for row, col in itertools.product(range(self._n_rows), range(self._n_columns))]