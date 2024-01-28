import itertools
import tomli
import glob
import string
from dataclasses import dataclass, field, asdict
from pathlib import Path

from typing import Tuple, Union, Optional, Dict, Any, List

import numpy as np
import pandas as pd

import plotly.express as px

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch

from plate_planner.logger import logger

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
        >>> well = Well(name="B2", plate_id=2, coordinate=(1, 1), index=5, rgb_color=(0.5, 0.5, 0.5),)
    """
    name: str = "A1"
    plate_id: int = 1
    coordinate: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    index: int = None
    empty: bool = True
    rgb_color: Tuple[float, float, float] = field(default_factory=lambda: (1, 1, 1))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self):
        """
        Converts the well object to a dictionary.

        The method returns a dictionary representation of the well object with the 
        direct attributes of the well and the keys in the metadata attribute.

        Returns:
            dict: A dictionary representation of the well object.

        Example:
            Convert a Well instance to a dictionary:
            >>> well = Well(name="B2", plate_id=2, coordinate=(1, 1), index=5, rgb_color=(0.5, 0.5, 0.5))
            >>> well_dict = well.as_dict()
            >>> print(well_dict)
            {'name': 'B2', 'plate_id': 2, 'coordinate': (1, 1), 'index': 5, 'rgb_color': (0.5, 0.5, 0.5)}

        """
        attrib_dict = asdict(self)
        del attrib_dict["metadata"]
        attrib_dict.update(self.metadata)

        return attrib_dict
    
    def get_attribute_or_metadata(self, key: str):
        """
        Get the value of a direct attribute or a key in the metadata dictionary.

        This method first checks if the provided key corresponds to a direct 
        attribute of the well object. If not, it then checks if the key exists 
        in the metadata dictionary.

        Parameters:
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
    
    def set_attribute_or_metadata(self, key: str, value: Any):
        """
        Set the value of a direct attribute or a key in the metadata dictionary.

        This method first checks if the provided key corresponds to a direct 
        attribute of the well object. If so, it sets the value of that attribute. 
        If not, it then updates or adds the key-value pair in the metadata dictionary.

        Parameters:
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

        Parameters:
            plate_dim (Tuple[int, int], optional): The dimensions of the plate as (rows, columns).
                If None, default dimensions are used.
            plate_id (int, optional): A unique identifier for the plate.

        The constructor initializes the plate with the specified dimensions, generating wells 
        with default properties and assigning them unique coordinates and identifiers.

        Examples:
            Creating a Plate instance with default dimensions and a specific plate ID:

            >>> plate = Plate(plate_id=1)
            >>> plate.size  # This will depend on the default number of rows and columns
            96  # Example output, assuming 8 rows x 12 columns

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

    def __iter__(self):
        return iter(self.wells)

    def __len__(self):
        """
        Returns the number of wells in the plate.

        Returns:
            int: The number of wells.
        """
        return len(self.wells)
    
    def __str__(self):
        plate_summary = f"Plate ID: {self.plate_id}\n"
        plate_summary += f"Dimensions: {self._n_rows} rows x {self._n_columns} columns\n"
        plate_summary += "Plate Layout (Well Names):\n"
        plate_array_str = np.array_str(self.get_metadata_as_numpy_array("name"))
        plate_summary += plate_array_str
        return plate_summary
    
    def __getitem__(self, key: Union[int, Tuple[int,int], str]):
        """
        Retrieve a well from the plate based on its index, coordinate, or name.

        Parameters:
            key (int, tuple, or str): The identifier for the well. Can be an integer index, 
            a tuple indicating row and column coordinates, or a string specifying the well's name.

        Returns:
            Well: The well object corresponding to the given key.

        Raises:
            TypeError: If the key is not an integer, tuple, or string.
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
        
    def __setitem__(self, key, well_object: Well):
        """
        Set or replace a well in the plate based on its index, coordinate, or name.

        Parameters:
            key (int, tuple, or str): The identifier for the well to be set or replaced. 
                Can be an integer index, a tuple indicating row and column coordinates, or a string specifying the well's name.
        well_object (Well): The well object to set at the specified key.

        Raises:
            ValueError: If the well_object is not an instance of Well.
            IndexError: If the well index is out of range.
            TypeError: If the key is not a string, integer, or tuple.
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

    def _parse_plate_dimensions(self, plate_dim: Union[Tuple[int, int], List[int], Dict[str, int], int]):
        """
        Parse the dimensions of the plate and return the number of rows and columns. This method can handle various 
        formats for specifying the dimensions: as a tuple or list (rows, columns), as a dictionary with 'rows' and 
        'columns' keys, or as an integer representing the total number of wells in a plate. For integer inputs, 
        the method attempts to design a plate with a 2:3 aspect ratio (height to width).

        Parameters:
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

    def _coordinates_to_index(self, coordinate: tuple):
        """
        Convert a well coordinate to its corresponding index in the plate's well list.

        Parameters:
            coordinate (tuple): The row and column coordinate of the well (row, col).

        Returns:
            int: The index of the well corresponding to the given coordinate.

        Raises:
            IndexError: If the coordinate is out of range of the plate's dimensions.
        """
        row, col = coordinate
        if row < 0 or row >= self._n_rows or col < 0 or col >= self._n_columns:
            raise IndexError("Coordinate out of range")
        return row * self._n_columns + col
    
    def _to_numpy_array(self, data: list) -> np.ndarray:
        """
        Convert a list of data corresponding to each well into a numpy array matching the plate's layout.

        Parameters:
            data (list): A list of data values corresponding to each well in the plate.

        Returns:
            numpy.ndarray: A numpy array representing the plate's layout with the provided data.

        Raises:
            Warning: If the number of data elements does not match the plate's size.
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
    
    def get_metadata(self, metadata_key: Optional[str]) -> list:
        """
        Retrieve metadata values for all wells in the plate based on the specified key.

        Parameters:
            metadata_key (str, optional): The metadata key for which values are to be retrieved. 
                If None, a default value of 'NaN' is returned for each well.

        Returns:
            list: A list of metadata values for each well in the plate.
        """
        if metadata_key is None:
            return ["NaN" for _ in self.wells]

        metadata_values = []
        for well in self.wells:
            value = well.get_attribute_or_metadata(metadata_key)
            metadata_values.append(value)

        return metadata_values

    def get_metadata_as_numpy_array(self, metadata_key : str) -> object:
        """
        Retrieve metadata values for all wells in a numpy array format based on the specified key.

        Parameters:
            metadata_key (str): The metadata key for which values are to be retrieved.

        Returns:
            numpy.ndarray: A numpy array representing the metadata values for the plate's layout.
        """
        metadata = self.get_metadata(metadata_key)
        return self._to_numpy_array(metadata)
    
    def _is_qualitative_colormap(self, colormap_name):
        """Check if a given colormap is qualitative."""
        # This list can be expanded with more qualitative colormaps
        qualitative_colormaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 
                                'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']
        return colormap_name in qualitative_colormaps
    
    def assign_well_color(self, metadata_key: Optional[str], colormap: str) -> None:
        """
        Assign colors to each well in the plate based on the specified metadata key and colormap.
        
        Parameters:
            metadata_key (str, optional): The metadata key to use for coloring the wells. 
                If None, a default color is assigned to each well.
        colormap (str): The name of the colormap to use for coloring the wells.
        
        Raises:
            ValueError: If the metadata_key is invalid or not found.
        """
        if colormap is None:
            colormap = self._default_colormap

        self._metadata_color_map = {}

        if metadata_key is not None:
            
            metadata_values = self.get_metadata(metadata_key)
            unique_values = list(set(metadata_values))
            
            cmap = plt.get_cmap(colormap)

            if self._is_qualitative_colormap(colormap):
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

    def as_records(self):
        """
        Convert the plate's well data into a list of dictionaries.

        Each well's attributes are converted into a dictionary, and all these dictionaries
        are compiled into a list, with one dictionary per well.

        Returns:
            list of dict: A list where each element is a dictionary representing a well's attributes.
        """
        return [well.as_dict() for well in self]

    def as_dataframe(self):
        """
        Converts the plate data into a Pandas DataFrame.

        Each well and its attributes are represented as a row in the DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame representing the plate's wells and their attributes.
        """
        return pd.DataFrame(self.as_records())
    
    def is_valid_metadata_key(self, key:str) -> bool:
        """
        Check if the provided key is a valid metadata key for the Well instances in the plate.

        This method verifies whether the specified key is either a direct attribute of the Well instances
        or a key within their metadata dictionary.

        Parameters:
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
                  ):
        """
        Create a visual representation of the plate using matplotlib.

        This method generates a figure representing the plate, with options for annotations,
        coloring based on metadata, and various styling adjustments.

        Parameters:
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
        self.assign_well_color(color_metadata_key, colormap)

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
        theme='plotly'
    ):
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
            plotly.graph_objs._scatter.Figure: A Plotly scatter plot figure.

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

        # Generate grid data for plotting, assuming equal spacing between wells
        step = 1 
        x = np.arange(0, len(self._columns)*step, step)
        y = np.arange(0, len(self._rows)*step, step)
        Xgrid, Ygrid = np.meshgrid(x, y)

        # Transform the plate data into a DataFrame for easier manipulation
        df = self.as_dataframe()

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
                # font=dict(size=fontsize, color="black"),
                bgcolor='rgba(255, 255, 255, 0.75)'
            )

        fig.update_traces(marker=dict(size=well_size, line=dict(width=2), opacity=alpha), selector=dict(mode='markers'))
        
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
                # range=[x[0] -x[1]*0.5, x[-1]+x[1]*0.5]
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
                # range=[y[0] -y[1]*0.5, y[-1]+y[1]*0.5]
            ),
            template=theme,
        )

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

        Parameters:
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
        
    def add_metadata(self, key, values):
        """
        Add or update metadata for all wells in the plate. If a list of values is provided,
        assign each value to the corresponding well. If a single value is provided, assign it to all wells.

        Parameters:
            key (str): The metadata key to add or update.
            values: A single value or a list of values to set for the given metadata key. 
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
    def row_labels(self):
        """
        Get the row labels for the plate.

        This property generates a list of alphabetical characters representing the row labels
        of the plate, based on the number of rows in the plate.

        Returns:
            list: A list of strings, each representing a row label.
        """
        return list(string.ascii_uppercase)[:len(self._rows)]
    
    @property
    def column_labels(self):
        """
        Get the column labels for the plate.

        This property generates a list of numerical strings representing the column labels
        of the plate, based on the number of columns in the plate.

        Returns:
            list: A list of strings, each representing a column label.
        """
        return [str(row_id+1) for row_id in self._columns]
    
    @property
    def capacity(self):
        """
        Get the number of samples that can be added to the plate, which is the same as the number of wells in this class
        """
        return self.size
    
    @property
    def plate_id(self):
        """
        Get the plate ID.

        This property returns the unique identifier of the plate.

        Returns:
            int: The plate ID.
        """
        return self._plate_id

    @plate_id.setter
    def plate_id(self, new_id):
        """
        Set a new plate ID.

        This method updates the plate ID and propagates the change to all the wells 
        within the plate.

        Parameters:
            new_id (int): The new plate ID to be set.
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

        Parameters:
            rows (iterable): An iterable representing the rows of the plate.
            columns (iterable): An iterable representing the columns of the plate.

        Returns:
            list: A list of tuples, each representing the (row, column) index of a well.
        """
        # count from left to right, starting at well in top left
        return list(itertools.product(
                                    range(len(rows)-1, -1, -1),
                                    range(0, len(columns))
                                    )
                )
    
    @staticmethod
    def create_alphanumerical_coordinates(rows, columns):
        """
        Static method to create alphanumerical coordinates for the wells.

        Parameters:
            rows (list): A list of row indices.
            columns (list): A list of column indices.

        Returns:
            list: A list of alphanumerical coordinates (e.g., "A1", "B2").
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
        self.qc_sequence_map = {round_num: None for round_num in range(1, max_rounds + 1)}

        # Step 2: Apply specific round patterns
        for key, value in self.config['QC']['patterns'].items():
            if key.startswith('round_'):
                round_number = int(key.split('_')[1])
                self.qc_sequence_map[round_number] = value

        # Step 3: Apply start/end patterns
        if 'start' in self.config['QC']['patterns']:
            self.qc_sequence_map[1] = self.config['QC']['patterns']['start']
        if 'end' in self.config['QC']['patterns']:
            self.qc_sequence_map[max_rounds] = self.config['QC']['patterns']['end']

        # Step 4: Apply repeat pattern
        if 'repeat_pattern' in self.config['QC']['patterns']:
            repeat_config = self.config['QC']['patterns']['repeat_pattern']
            pattern, times = repeat_config['pattern'], repeat_config['times']
            for i in range(1, times + 1):
                if self.qc_sequence_map[i] is None:
                    self.qc_sequence_map[i] = pattern

        # Step 5: Apply alternating patterns
        if 'then_alternating' in self.config['QC']['patterns']:
            alternating_patterns = self.config['QC']['patterns']['then_alternating']
            alt_index = 0
            for round_num in range(1, max_rounds + 1):
                if self.qc_sequence_map[round_num] is None:
                    self.qc_sequence_map[round_num] = alternating_patterns[alt_index % len(alternating_patterns)]
                    alt_index += 1

        # Step 6: Apply every N rounds patterns
        every_n_patterns = {key: value for key, value in self.config['QC']['patterns'].items() if key.startswith('every_')}

        for key, pattern in every_n_patterns.items():
            # Correctly extract the frequency (e.g., 4 in 'every_4_rounds')
            try:
                frequency = int(''.join(filter(str.isdigit, key)))
            except ValueError:
                logger.error(f"Invalid frequency format in pattern key: {key}")
                continue

            for round_num in range(1, max_rounds + 1):
                if round_num % frequency == 0 and self.qc_sequence_map[round_num] is None:
                    self.qc_sequence_map[round_num] = pattern

        # Log the defined sequences
        for round_number, sequence in self.qc_sequence_map.items():
            logger.debug(f"Round {round_number}: {sequence}")

    def create_QC_plate_layout(self):
        """ Creates the plate layout with QC and specimen samples. """
        logger.info("Creating dynamic plate layout with QC samples.")

        self.define_unique_QC_sequences()

        # Initialize counters for QC sample types
        counts = {qc_type: 0 for qc_type in self.config["QC"]["names"].keys()}
        
        round_counter = 1
        specimen_counter = 0
        qc_round_frequency = self.config['QC']['run_QC_after_n_specimens']
        start_with_qc = self.config['QC']['start_with_QC_round']
        current_round_qc_samples = iter(self.qc_sequence_map.get(round_counter, []))

        for well_index in range(self.size):
            if start_with_qc and round_counter == 1:
                try:
                    # Place QC samples for the first round
                    qc_sample = next(current_round_qc_samples)
                    self.assign_qc_sample_metadata(well_index, qc_sample, counts)
                    continue
                except StopIteration:
                    round_counter += 1
                    current_round_qc_samples = iter(self.qc_sequence_map.get(round_counter, []))

            if specimen_counter >= qc_round_frequency:
                try:
                    qc_sample = next(current_round_qc_samples)
                    self.assign_qc_sample_metadata(well_index, qc_sample, counts)
                except StopIteration:
                    round_counter += 1
                    specimen_counter = 0
                    current_round_qc_samples = iter(self.qc_sequence_map.get(round_counter, []))
                    self.assign_specimen_sample_metadata(well_index, specimen_counter)
                    specimen_counter += 1
            else:
                self.assign_specimen_sample_metadata(well_index, specimen_counter)
                specimen_counter += 1

        # Log the final layout
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