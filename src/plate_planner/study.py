from pathlib import Path
import copy
import datetime
from typing import Union, Iterator, Any, Tuple, Literal

import pandas as pd
import numpy as np

import pulp as lp
from scipy.stats import chi2_contingency
from scipy.stats import entropy

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from .plate import Plate, QCPlate, PlateFactory
from .logger import logger

class Study:
    """
    A class to manage and manipulate study-related data, including specimen records, plate layouts, 
    and sample distributions.

    This class provides functionalities for loading specimen records, sorting and randomizing 
    specimen order, distributing specimens across plates, and exporting data to various formats.

    Attributes:
        name (str): Name of the study.
        plates (list): List of Plate objects used in the study.
        total_plates (int): Total number of plates in the study.
        specimen_records_df (DataFrame): Pandas DataFrame holding specimen records.
        records_file_path (str): Path to the file containing specimen records.
        _column_with_group_index (str): Column name in specimen_records_df that holds group indices.

    Examples:
        >>> study = Study(study_name="Cancer")
        >>> study.load_specimen_records("specimens.csv", sample_group_id_column="GroupID")
        >>> study.randomize_order(case_control=True)

        >>> qc_plate = QCPlate(QC_config="./data/plate_config_dynamic.toml")
        >>> study.randomize_order()
        >>> study.distribute_samples_to_plates()
        >>> study.to_layout_lists()
    """


    _default_distribute_samples_equally : bool = False
    
    _batch_count : int = 0
    _iter_count : int = 0
    _default_seed = 1234 # seed number for the random number generator in case randomization of specimens should be reproducible
    _N_permutations : int = 0
    _column_with_group_index : str = ""
    
    def __init__(self, name=None, samples=None) -> None:
        """
        Initializes a new instance of the Study class.

        This constructor sets up a study with a specified name. If no name is provided, a default name 
        is generated using the current date.

        Args:
            name (Optional[str]): The name for the study. If None, a default name in the format 
                "Study_YYYY-MM-DD" is assigned, where YYYY-MM-DD represents the current date.
            samples (Optional[Union[str, Path, pd.DataFrame]]): The specimen records for the study. This can be a path to a file or a pandas DataFrame.

        Examples:
            >>> study1 = Study(study_name="Alzheimer's Research")
            >>> study1.name
            "Alzheimer's Research"
            
            >>> study2 = Study()
            >>> study2.name
            "Study_2024-01-21"  # Example output; actual output will vary based on the current date.
        """
        
        if name is None:
            name = f"Study_{datetime.date}"

        if samples is not None:
            self.load_sample_list(samples)

            
        self.name = name
        self.plates = []
        
    def __iter__(self) -> Iterator[Union[Plate, QCPlate]]:
        """
        Initialize the iterator for the Study class.

        This method sets up the class to iterate over its plates, resetting the internal counter 
        to zero. It allows the Study instance to be used in a loop (e.g., a for loop), 
        facilitating iteration over its plates.

        Returns:
            Iterator[Union[Plate, QCPlate]]: An iterator that yields either `Plate` or `QCPlate` 
            instances, allowing the Study instance to be used in a loop.
        """
        self._iter_count = 0
        return self
        
    def __next__(self) -> Union[Plate, QCPlate]:
        """
        Proceed to the next plate in the Study class during iteration.

        This method returns the next plate in the study, which can be either a `Plate` or a `QCPlate` instance.
        It is automatically called in each iteration of a loop. When all plates have been iterated over,
        it raises the StopIteration exception.

        Returns:
            Union[Plate, QCPlate]: The next plate in the study, either a `Plate` or `QCPlate` instance.

        Raises:
            StopIteration: If all plates have been iterated over.
        """
        if self._iter_count < self.total_plates:
            plate_to_return = self.plates[self._iter_count]
            self._iter_count += 1
            return plate_to_return
        else:
            raise StopIteration
    
    def __len__(self) -> int:
        """
        Return the total number of plates in the Study.

        This method enables the use of the len() function on the Study instance.

        Returns:
            int: The number of plates in the study.
        """
        return len(self.plates)
        
    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of the Study instance.

        This method is useful for debugging and logging purposes, as it represents the Study
        object in a clear and concise way.

        Returns:
            str: A string representation of the Study instance.
        """
        return f"Study({self.name})"

    def __str__(self) -> str:
        """
        Return a readable string representation of the Study instance.

        This method provides a user-friendly string representation of the Study, which includes
        its name, the number of study specimens, and the total number of plates.

        Returns:
            str: A string describing the Study instance.
        """
        return f"{self.name}\n {self.study_specimens} on {self.total_plates}"
    
    def __getitem__(self, index) -> Union[Plate, QCPlate]:
        """
        Retrieve a specific plate from the study by its index.

        This method allows for direct access to a plate in the study using the indexing syntax.

        Args:
            index (int): The index of the plate to retrieve.

        Returns:
            Union[Plate, QCPlate]: The plate at the specified index.

        Raises:
            IndexError: If the index is out of range of the plates list.
        """
        return self.plates[index]
        
    def load_sample_list(self, sample_list: Union[str, Path, pd.DataFrame], sample_group_id_column=None, sample_id_column=None) -> None:
        """
        Loads specimen records from a specified file or DataFrame into the study.

        This method reads specimen data from a file (Excel or CSV) or directly from a provided DataFrame and stores it in a DataFrame.
        It also identifies or sets the column used for grouping specimens.

        Args:
            records (Union[str, Path, DataFrame]): The path to the file containing specimen records or a DataFrame with the records.
            sample_group_id_column (Optional[str]): The column name in the records that represents the group ID of samples. 
                If None, the method attempts to find a suitable column automatically.

        Raises:
            FileExistsError: If the specified records_file does not exist and records is a file path.

        Examples:
            >>> study = Study(study_name="Oncology Study")
            >>> study.load_sample_data("specimens.xlsx", sample_group_id_column="PatientGroup")
            >>> study.load_sample_data(df, sample_group_id_column="PatientGroup")
            >>> study.specimen_records_df.shape
            (200, 5)  # Example output, indicating 200 rows and 5 columns in the DataFrame.
        """

        if isinstance(sample_list, str):
            records_path = Path(sample_list)
            logger.debug(f"Loading records from file: {sample_list}")
            
            if not records_path.exists():
                logger.error(f"Could not find file {sample_list}")
                raise FileExistsError(sample_list)
            
            extension = records_path.suffix
            print(extension)
            if extension in [".xlsx", ".xls"]:
                logger.debug("Importing Excel file.")
                records_df = pd.read_excel(sample_list)
            elif extension == ".csv":
                logger.debug("Importing CSV file.")
                records_df = pd.read_csv(sample_list)
            else:
                logger.error("File extension not recognized")
                records_df = pd.DataFrame()
        elif isinstance(sample_list, pd.DataFrame):
            logger.debug("Loading records from DataFrame.")
            records_df = sample_list
        else:
            logger.error("Unsupported records format")
            records_df = pd.DataFrame()

        if sample_group_id_column is None and not records_df.empty:
            self._column_with_group_index = self._find_column_with_group_index(records_df)
        else:
            self._column_with_group_index = sample_group_id_column
        
        logger.debug(f"{records_df.shape[0]} specimens loaded")
        logger.info("Metadata:")
        for col in records_df.columns:
            logger.info(f"\t{col}")

        if sample_id_column:
            logger.debug(f"Sorting records in ascending order based on column '{sample_id_column}'")
            records_df = records_df.sort_values(by=[sample_id_column])
            
        self.sample_records_df = records_df

    def sort_records_within_groups(self, sortby_column: str) -> None:
        """
        Sorts specimen records within each group based on a specified column.

        This method groups the specimen records by a predefined group index column and then sorts each group's records
        based on the specified 'sortby_column'. The sorted groups are then concatenated back into the main DataFrame.
        This is useful for organizing records in a manner that respects the grouping while ordering the records within each group.

        Args:
            sortby_column (str): The column name in the specimen records DataFrame to sort by within each group.

        Raises:
            ValueError: If no group column is defined in the Study instance.

        Examples:
            >>> study = Study(...)
            >>> study.load_specimen_records("specimens.csv", sample_group_id_column="GroupID")
            >>> study.sort_records_within_groups("Age")
            # This will sort the specimen records within each group based on the "Age" column.
        """
        if self._column_with_group_index:
            logger.info(f"Sorting samples within {self._column_with_group_index} by {sortby_column}")
            # Step 1: Group the DataFrame by 'group_ID'
            grouped = self.sample_records_df.groupby(self._column_with_group_index)

            # Step 2: Sort each group by the 'sortby_column' column
            sorted_groups = [group.sort_values(sortby_column) for _, group in grouped]

            # Step 3: Concatenate the sorted groups back into a single DataFrame
            self.sample_records_df = pd.concat(sorted_groups)

        else:
            raise ValueError(f"No group column defined: self._column_with_group_index: {self._column_with_group_index}")
    
    def position_sample_within_groups(self, sortby_column: str, sample_value: Any, position_index: int) -> None:
        """
        Repositions a specific sample within each group based on a specified value and index.

        This method allows altering the position of a sample within each group in the specimen records DataFrame.
        It locates a sample based on the 'sortby_column' and 'sample_value', then repositions this sample within its group
        to the specified 'position_index'. The method is useful for customizing the order of samples within groups
        based on specific criteria or requirements.

        Args:
            sortby_column (str): The column name in the specimen records DataFrame to identify the sample.
            sample_value (Any): The value in the 'sortby_column' that identifies the sample to reposition.
            position_index (int): The new index within the group where the sample should be positioned.

        Raises:
            ValueError: If no group column is defined in the Study instance.

        Examples:
            >>> study = Study(...)
            >>> study.load_specimen_records("specimens.csv", sample_group_id_column="GroupID")
            >>> study.position_sample_within_groups("PatientID", 12345, 2)
            # This will move the sample with PatientID 12345 to the index 2 position within its respective group.
        """
        if self._column_with_group_index:
            logger.info(f"Positioning sample within {self._column_with_group_index} based on {sortby_column} value {sample_value} at index {position_index}")
            # Step 1: Group the DataFrame by 'group_ID'
            grouped = self.sample_records_df.groupby(self._column_with_group_index)

            # Step 2: Modify each group
            modified_groups = []
            for _, group in grouped:
                # Find the row to reposition
                sample_row = group[group[sortby_column] == sample_value]

                # Remove this row from the group
                group = group[group[sortby_column] != sample_value]

                # Split the group at the specified position index
                first_part = group.iloc[:position_index]
                second_part = group.iloc[position_index:]

                # Concatenate first part, sample row, and second part
                modified_group = pd.concat([first_part, sample_row, second_part])

                modified_groups.append(modified_group)

            # Step 3: Concatenate the modified groups back into a single DataFrame
            self.sample_records_df = pd.concat(modified_groups).reset_index(drop=True)

        else:
            raise ValueError(f"No group column defined: self._column_with_group_index: {self._column_with_group_index}")

    def _add_specimens_to_plate(self, study_plate: object, specimen_samples_df: object) -> None:
        """
        Adds specimens from a DataFrame to a specified plate based on well metadata.

        This method iterates through each well of the given plate and, if the well's metadata indicates 
        it should contain a specimen (as marked by "sample_code" == "S"), adds specimen data from the 
        DataFrame to the well's metadata. The method stops adding specimens once all have been placed 
        or the plate is filled.

        Args:
            study_plate (Plate): The plate object to which specimens will be added.
            specimen_samples_df (pd.DataFrame): DataFrame containing specimen data to be added to the plate.

        Notes:
            - This is a private method intended for internal use within the class.
            - The method assumes that 'study_plate' is an instance of 'Plate' or a compatible type, and
            'specimen_samples_df' is a Pandas DataFrame with columns corresponding to metadata fields.

        Examples:
            >>> study = Study(...)
            >>> study_plate = Plate(...)  # Assume Plate is properly initialized
            >>> specimens_df = pd.DataFrame(...)  # DataFrame with specimen data
            >>> study._add_specimens_to_plate(study_plate, specimens_df)
            # Specimens from specimens_df are now added to the study_plate based on its well metadata
        """
        
        logger.debug(f"Adding {len(specimen_samples_df)} samples to plate {study_plate.plate_id}")
        specimen_samples_df = specimen_samples_df.reset_index(drop=True)
        columns = specimen_samples_df.columns
        
        # keep track on how many wells we should use per batch
        N_specimens_left = len(specimen_samples_df)
        plate_specimen_count = 0
        
        for i, well in enumerate(study_plate):

            if well.metadata["sample_code"] == "S": 
                # add metadata key (and values) for each column in dataframe
                for col in columns:
                    well.metadata[col] = specimen_samples_df[col][plate_specimen_count]
                    
                plate_specimen_count += 1
                well.empty = False
                
            else:
                # add metadata key and nan value for each column in dataframe
                for col in columns:
                    well.metadata[col] = "NaN"

                well.empty = False
                    
            study_plate[i] = well
            
            if plate_specimen_count >= N_specimens_left:
                    logger.debug(f"\t -> Done. Last specimen placed in {well.name}")
                    break
                
        return study_plate
                
        # --- END OF FOOR LOOP ---
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the data from all plates in the study into a single Pandas DataFrame.

        This method iterates over each plate in the study and converts its data to a DataFrame. 
        These DataFrames are then concatenated into a single DataFrame representing the entire study.

        Returns:
            pd.DataFrame: A DataFrame containing data from all plates in the study.

        Examples:
            >>> study = Study(...)
            >>> study_df = study.to_dataframe()
            >>> study_df.head()
            # Displays the first few rows of the combined DataFrame for the study.
        """
        dfs = []
        for plate in self:
            dfs.append(plate.as_dataframe())
        return pd.concat(dfs).reset_index(drop=True)
    
    def to_layout_lists(self, metadata_keys: list = [], 
                    file_format: str = "csv",
                    folder_path: str = None,
                    plate_name: str = "plate") -> None:
        """
        Exports the layout of each plate in the study to files in the specified format.

        This method iterates over each plate in the study and exports its layout to a file. 
        The files are saved in a specified format (CSV by default) and stored in a designated folder.

        Args:
            metadata_keys (list, optional): A list of metadata keys to include in the exported files.
            file_format (str, optional): The file format for the exported layouts (e.g., 'csv').
            folder_path (str, optional): The path to the folder where the layout files will be saved. 
                If None, the current working directory is used.
            plate_name (str, optional): A base name for the layout files.

        Examples:
            >>> study = Study(...)
            >>> study.to_layout_lists(metadata_keys=["sample_type", "concentration"], 
                                    file_format="csv", 
                                    folder_path="/path/to/layouts", 
                                    plate_name="experiment_plate")
            # This will save layout files for each plate in the '/path/to/layouts' directory.
        """

        if folder_path is None:
            folder_path = Path.cwd()
        else:
            folder_path = Path(folder_path)

        # New folder path for "layout_lists"
        new_folder_path = folder_path / "layout_lists"

        # Create the "layout_lists" folder if it does not exist
        new_folder_path.mkdir(parents=True, exist_ok=True)
        
        for plate in self:
            file_name = f"{self.name}_{plate_name}_{plate.plate_id}"
            # Update file path to include the new folder
            file_path = new_folder_path / file_name
            
            plate.to_file(file_path=str(file_path),
                        file_format=file_format,
                        metadata_keys=metadata_keys)
    
    def to_layout_figures(self,
                          annotation_metadata_key : str,
                          color_metadata_key : str,
                        file_format : str = None,
                        folder_path : str = None,
                        plate_name : str = "Plate", **kwargs) -> None:
        """
        Creates and visual representations of each plate in the study as figures.

        This method iterates over each plate in the study, generating a figure based on specified metadata keys
        for annotation and coloring. If the file format is specified, the figures are saved in the specified file format in a designated folder.

        Args:
            annotation_metadata_key (str): The metadata key used for annotating elements in the figure.
            color_metadata_key (str): The metadata key used for coloring elements in the figure.
            file_format (str, optional): The format in which to save the figures (default is 'pdf').
            folder_path (Optional[str]): The path to the folder where the figures will be saved. 
                If None, the current working directory is used.
            plate_name (str, optional): A base name for the figure files.
            **kwargs: Additional keyword arguments passed to the `as_figure` method of each plate.

        Examples:
            >>> study = Study(...)
            >>> study.to_layout_figures(annotation_metadata_key="sample_id",
                                    color_metadata_key="status",
                                    file_format="png",
                                    folder_path="/path/to/figures",
                                    plate_name="study_plate")
            # This will create and save figures for each plate in the '/path/to/figures' directory,
            # with annotations and colorings based on 'sample_id' and 'status'.
        """
        
        if file_format is not None:
            if folder_path is None:
                folder_path = Path.cwd()
            else:
                folder_path = Path(folder_path)

            # New folder path for "layout_lists"
            new_folder_path = folder_path / "layout_figures"

            # Create the "layout_lists" folder if it does not exist
            new_folder_path.mkdir(parents=True, exist_ok=True)
            
        for plate in self:
            file_name = f"{self.name}_{plate_name}_{plate.plate_id}_{annotation_metadata_key}_{color_metadata_key}.{file_format}"
            file_path = new_folder_path / file_name
            
            # Define title        
            title_str = f"{self.name}: Plate {plate.plate_id}, showing {annotation_metadata_key} colored by {color_metadata_key}"
           
            fig = plate.as_figure(annotation_metadata_key, color_metadata_key, title_str=title_str, **kwargs)
    
            logger.info(f"Saving plate figure to {file_path}")

            if file_format:
                fig.savefig(file_path)

    def distribute_samples_to_plates_ilp(self,
            plate_layout: Union[Plate, QCPlate],
            balance_columns: list,
            blocking_column: str = None,
            group_column: str = None,
            max_samples_per_plate: int = None,
            block_shuffle=True,
            full_plate=False,
        ):
        
        if max_samples_per_plate is None:
            max_samples_per_plate = plate_layout.n_non_qc_wells

        assigned_plates_df = self.assign_plates_sort(
                self.sample_records_df.copy(),
                balance_columns=balance_columns,
                max_samples_per_plate=max_samples_per_plate,
                group_id_col=group_column,
                full_plate=full_plate,
        )
        
        # assigned_plates_df, solution_found = self.assign_plates_ilp(
        #         self.sample_records_df.copy(),
        #         balance_columns=balance_columns,
        #         max_samples_per_plate=max_samples_per_plate,
        #         group_column=group_column,
        #         full_plate=full_plate,
        #         tolerance=tolerance
        # )
     

        # if solution_found == 0:
        #     raise ValueError("Samples could not be distributed optimally using ILP with set constraints")

        plates = []

        for plate_id in assigned_plates_df["assigned_plate"].unique():

            logger.info(f"Processing {plate_id}")
            current_plate = copy.deepcopy(plate_layout)
            current_plate.plate_id = plate_id

            plate_sel = assigned_plates_df["assigned_plate"] == plate_id
            plate_samples_df = assigned_plates_df[plate_sel]

            # Block samples
            if blocking_column is not None:
                if blocking_column in assigned_plates_df.columns:
                    try:
                        plate_samples_df, solution_found = self.create_blocks_within_plate_ilp(
                            samples_df=plate_samples_df.copy(),
                            group_column=group_column,
                            blocking_column=blocking_column,
                            block_shuffle=block_shuffle,
                            
                            )
                    except Exception as e:
                        print(e)


                    if not solution_found:
                        raise ValueError("Samples could not be grouped optimally using ILP with set constraints")
                else:
                    raise ValueError(f"Blocking samples failed: column '{blocking_column}' is not inte sample list dataframe columns ({assigned_plates_df.columns})")

            # Add specimens to the current plate
            current_plate = self._add_specimens_to_plate(current_plate, plate_samples_df)
            plates.append(current_plate)


        self.plates = plates
        self.total_plates = len(plates)

        logger.info(f"Distributed samples across {self.total_plates} plates.")


    def distribute_samples_to_plates(self, plate_layout: Union[Plate, QCPlate], allow_group_split=False, N_samples_desired_plate=None) -> None:
        """
        Distributes specimens across multiple plates based on a specified layout, with an option to keep group integrity.

        This method iterates through the study's specimen records and distributes them across multiple plates 
        according to the provided plate layout. It supports options to either keep specimen groups together or 
        allow splitting them across different plates. The method can also handle a specified number of samples 
        per plate if desired.

        Args:
            plate_layout (Union[Plate, QCPlate]): The layout template for the plates. This can be an instance of 
                'Plate' or 'QCPlate'.
            allow_group_split (bool, optional): If False (default), keeps specimens within the same group on the 
                same plate. If True, allows splitting groups across plates.
            N_samples_desired_plate (Optional[int], optional): The desired number of samples per plate. If not 
                specified, fills each plate to its capacity.

        Raises:
            ValueError: If the group column is not defined in the specimen records.

        Examples:
            >>> study = Study(...)
            >>> study.load_specimen_records("specimens.csv", sample_group_id_column="GroupID")
            >>> plate_layout = Plate(...)  # Assume Plate is properly initialized
            >>> study.distribute_samples_to_plates(plate_layout, allow_group_split=False)
            # This will distribute the specimens across plates, keeping groups together.
        """

        plate_number = 1
        plates = []

        # Copy the specimen data to work on
        remaining_specimens = self.sample_records_df.copy()

        N_specimens = self.sample_records_df.shape[0]

        if N_samples_desired_plate is None:
            N_samples_desired_plate = plate_layout.capacity

        # N_QCsample_in_plate = plate_layout.get_metadata_as_numpy_array("QC").sum()
        # N_plates_estimate = N_specimens / (plate_layout._specimen_capacity)

        while not remaining_specimens.empty:
            current_plate = copy.deepcopy(plate_layout)
            current_plate.plate_id = plate_number

            # Select specimens for the current plate
            N_remaining = remaining_specimens.shape[0]

            # if the remaining samples can fit on a whole place we put them there,
            # otherwise we place the desired number of samples on the plate.
            # if not we 
            if N_remaining < current_plate.capacity:
                selected_specimens = remaining_specimens.head(current_plate.capacity)
            else:
                selected_specimens = remaining_specimens.head(N_samples_desired_plate)

            if not allow_group_split:
                # Extract unique group IDs from the selected specimens. This step identifies the distinct groups
                # that are represented within the specimens currently being considered for this plate.
                group_ids = selected_specimens[self._column_with_group_index].unique()

                # Find all specimens in the remaining pool that belong to the same groups as the selected specimens.
                specimens_in_groups = remaining_specimens[remaining_specimens[self._column_with_group_index].isin(group_ids)]

                if len(specimens_in_groups) > len(selected_specimens):
                    # If there are more specimens in the remaining pool belonging to the same groups,
                    # it indicates that the last group in 'selected_specimens' is split between this plate and the remaining pool.
                    # To avoid splitting the group, we modify 'selected_specimens' to exclude this last group.

                    # Determine the groups to keep on the current plate. This is done by excluding the last group ID
                    # from the list of unique group IDs in the selected specimens. This way, we ensure that an entire group 
                    # is not split across plates.
                    groups_to_keep = group_ids[:-1]

                    # Update 'selected_specimens' to only include specimens from the groups that are not split.
                    selected_specimens = selected_specimens[selected_specimens[self._column_with_group_index].isin(groups_to_keep)]

            # Remove selected specimens from the pool
            remaining_specimens.drop(index=selected_specimens.index, inplace=True)
            selected_specimens.reset_index(drop=True, inplace=True)
            remaining_specimens.reset_index(drop=True, inplace=True)

            # Add specimens to the current plate
            current_plate = self._add_specimens_to_plate(current_plate, selected_specimens)
            plates.append(current_plate)

            plate_number += 1

        self.plates = plates
        self.total_plates = plate_number - 1

        logger.info(f"Distributed samples across {self.total_plates} plates.")

    
       
    @staticmethod
    def _find_column_with_group_index(specimen_records_df: pd.DataFrame) -> str:
        """
        Identifies the column in a DataFrame that likely represents group indices based on integer pairs.

        This static method analyzes a provided DataFrame to determine which column could represent group indices.
        It looks for columns with integer values where pairs of consecutive numbers are common, indicating a
        potential grouping pattern. The method is specifically designed to identify groups based on pair numbers.

        Args:
            specimen_records_df (pd.DataFrame): The DataFrame containing specimen records.

        Returns:
            str: The name of the column that likely represents group indices. Returns an empty string if no suitable
                column is found.

        Examples:
            >>> specimen_records_df = pd.DataFrame({'GroupID': [1, 1, 2, 2, 3, 3], 'Data': [100, 101, 102, 103, 104, 105]})
            >>> Study._find_column_with_group_index(specimen_records_df)
            'GroupID'
        """
        int_cols = specimen_records_df.select_dtypes("int")
                    
        logger.debug(f"Looking for group index of study pairs in the following table columns:")
        
        for col_name in int_cols.columns:
            
            logger.debug(f"\t\t{col_name}")
            
            # sort in ascending order
            int_col = int_cols[col_name].sort_values()
            # compute difference: n_1 - n_2, n_2 - n_3, ...
            int_diffs = np.diff(int_col)
            # count instances were numbers were the same, i.e. diff == 0
            n_zeros = np.sum(list(map(lambda x: x==0, int_diffs)))
            # we assume column contains pairs if #pairs == #samples / 2
            column_have_pairs = n_zeros == (int_col.shape[0]//2)

            if column_have_pairs:# we found a column so let's assume it is the correct one
                logger.info(f"Found group index in column {col_name}")
                return col_name
            
        return "" 
    
    @staticmethod
    def assign_plates_monte_carlo(
        samples_df: pd.DataFrame,
        sample_id_column: str,
        balance_column: str,
        max_samples_per_plate,
        group_id_column: str = None,
        iterations=1000) -> pd.DataFrame:
        
        def calculate_abs_deviation(organ_ids, plate_assignments, total_plates, global_proportions):
            summed_abs_residuals = 0
            expected_counts_per_plate = global_proportions * (len(organ_ids) / total_plates)
            
            for plate in range(total_plates):
                plate_mask = plate_assignments == plate
                actual_counts = np.bincount(organ_ids[plate_mask], minlength=len(global_proportions))
                
                # Calculate variance as the sum of squared differences from expected counts
                summed_abs_residuals += np.sum(np.abs(actual_counts - expected_counts_per_plate) )
            
            # Normalize variance by the number of plates to avoid scaling effects
            # normalized_variance = deviation_sum / total_plates
            return summed_abs_residuals
         
        # use index if samples are not grouped in blocks 
        if group_id_column is None:
            samples_df = samples_df.reset_index()
            group_id_column = "index"

         # Calculate the required number of plates based on max_samples_per_plate
        total_samples = len(samples_df)
        total_plates = -(-total_samples // max_samples_per_plate)  # Ceiling division to ensure enough plates

        samples_df = samples_df.copy()

        # code categories to be balances
        balance_column_id = f"{balance_column}_id"
        samples_df[balance_column_id] = samples_df[balance_column].astype('category').cat.codes

        # Create a NumPy matrix from the DataFrame
        data_matrix = samples_df[[sample_id_column, group_id_column, balance_column_id]].to_numpy()

        category_counts_global = np.bincount(data_matrix[:, 2])  # Assuming organ_id is in the 3rd column
        total_samples = len(data_matrix)
        global_proportions = category_counts_global / total_samples

        best_deviation = np.inf
        best_assignment = None
        
        unique_groups = np.unique(data_matrix[:, 1])  # group_id is in the 2nd column
        group_sizes = {group: (data_matrix[:, 1] == group).sum() for group in unique_groups}

        deviation_search = np.zeros([iterations, 1])

        for i in range(iterations):
            np.random.shuffle(unique_groups)  # Shuffle groups to randomize distribution
            group_plate_assignments = np.zeros(unique_groups.shape, dtype=int) - 1  # Initialize with -1
            plate_samples_count = np.zeros(total_plates, dtype=int)

            for group in unique_groups:
                possible_plates = [plate for plate in range(total_plates) if plate_samples_count[plate] + group_sizes[group] <= max_samples_per_plate]
                if possible_plates:
                    chosen_plate = np.random.choice(possible_plates)
                    group_plate_assignments[group] = chosen_plate
                    plate_samples_count[chosen_plate] += group_sizes[group]

            # Create a mapping from group IDs to plate assignments
            group_to_plate_map = np.zeros(unique_groups.max() + 1, dtype=int) - 1
            group_to_plate_map[unique_groups] = group_plate_assignments

            # Vectorized assignment of samples to plates based on group ID
            sample_plate_assignments = group_to_plate_map[data_matrix[:, 1].astype(int)]

            # Handle any groups that might not have been assigned due to filtering or preprocessing
            unassigned_mask = sample_plate_assignments == -1
            if np.any(unassigned_mask):
                # Handle unassigned samples; options might include assigning to a default plate or redistributing
                sample_plate_assignments[unassigned_mask] = np.random.randint(0, total_plates, size=np.sum(unassigned_mask))

            # Calculate variance of organ distribution across plates
            deviation = calculate_abs_deviation(data_matrix[:, 2], sample_plate_assignments, total_plates, global_proportions)    
            deviation_search[i] = deviation 
            if deviation < best_deviation:
                best_deviation = deviation
                best_assignment = sample_plate_assignments

        samples_df["assigned_plate"] = best_assignment
        
        return samples_df, deviation_search
    
    @staticmethod
    def assign_plates_monte_carlo_cramer(
        samples_df: pd.DataFrame,
        sample_id_column: str,
        balance_column: str,
        max_samples_per_plate,
        group_id_column: str = None,
        iterations=1000,
        output_search_traj=False) -> pd.DataFrame:

        def calculate_cramers_v(observed_counts):
            chi2, p, dof, _ = chi2_contingency(observed_counts)
            n = np.sum(observed_counts)  # Total number of samples
            cramers_v = np.sqrt(chi2 / (n * (min(observed_counts.shape) - 1)))
            
            return cramers_v, chi2, p

        def calculate_observed_expected_counts(category_ids, plate_assignments, total_plates, global_proportions):
            observed_counts = np.zeros((len(global_proportions), total_plates))
            total_category_samples = len(category_ids)
            expected_counts = global_proportions[:, None] * (total_category_samples / total_plates)

            for plate in range(total_plates):
                plate_mask = plate_assignments == plate
                for category_id in range(len(global_proportions)):
                    observed_counts[category_id, plate] = np.sum(category_ids[plate_mask] == category_id)

            return observed_counts, expected_counts

        # Use index if samples are not grouped in blocks
        if group_id_column is None:
            samples_df = samples_df.reset_index()
            group_id_column = "index"

        # Calculate the required number of plates based on max_samples_per_plate
        total_samples = len(samples_df)
        total_plates = np.ceil(total_samples / max_samples_per_plate).astype(int)  # Ceiling division to ensure enough plates

        samples_df = samples_df.copy()

        # Code categories to be balanced
        balance_column_id = f"{balance_column}_id"
        samples_df[balance_column_id] = samples_df[balance_column].astype('category').cat.codes

        # Create a NumPy matrix from the DataFrame
        data_matrix = samples_df[[sample_id_column, group_id_column, balance_column_id]].to_numpy()

        category_counts_global = np.bincount(data_matrix[:, 2], minlength=len(samples_df[balance_column_id].unique()))  # Assuming organ_id is in the 3rd column
        total_samples = len(data_matrix)
        global_proportions = category_counts_global / total_samples

        best_cramers_v = 1 
        best_chi2 = None
        best_p = 0

        if output_search_traj:
            cramers_phi_trials = []
            chi2_trials = []
            p_trials = []

        best_assignment = None

        unique_groups = np.unique(data_matrix[:, 1])  # group_id is in the 2nd column
        group_sizes = {group: (data_matrix[:, 1] == group).sum() for group in unique_groups}

        for i in range(iterations):
            np.random.shuffle(unique_groups)  # Shuffle groups to randomize distribution
            group_plate_assignments = np.zeros(unique_groups.shape, dtype=int) - 1  # Initialize with -1
            plate_samples_count = np.zeros(total_plates, dtype=int)

            for group in unique_groups:
                possible_plates = [plate for plate in range(total_plates) if plate_samples_count[plate] + group_sizes[group] <= max_samples_per_plate]
                if possible_plates:
                    chosen_plate = np.random.choice(possible_plates)
                    group_plate_assignments[group] = chosen_plate
                    plate_samples_count[chosen_plate] += group_sizes[group]

            # Create a mapping from group IDs to plate assignments
            group_to_plate_map = np.zeros(unique_groups.max() + 1, dtype=int) - 1
            group_to_plate_map[unique_groups] = group_plate_assignments

            # Vectorized assignment of samples to plates based on group ID
            sample_plate_assignments = group_to_plate_map[data_matrix[:, 1].astype(int)]

            observed_counts, expected_counts = calculate_observed_expected_counts(data_matrix[:, 2], sample_plate_assignments, total_plates, global_proportions)
            cramers_v, chi2, p = calculate_cramers_v(observed_counts)

            if output_search_traj:
                cramers_phi_trials.append(cramers_v)
                chi2_trials.append(chi2)
                p_trials.append(p)


            if cramers_v < best_cramers_v:
                best_cramers_v = cramers_v
                best_chi2 = chi2
                best_p = p

                best_assignment = sample_plate_assignments

        samples_df["assigned_plate"] = best_assignment

        best_metrics = {
            "cramers_phi": best_cramers_v,
            "chi2": best_chi2,
            "p": best_p,
        }

        if output_search_traj:
            best_metrics["cramers_phi_trials"] = cramers_phi_trials
            best_metrics["chi2_trials"] = chi2_trials
            best_metrics["p_trials"] = p_trials

        return samples_df, best_metrics
    

    @staticmethod
    def assign_plates_sort(
            samples_df: pd.DataFrame,
            balance_columns: list,
            max_samples_per_plate: int,
            group_id_col: str = None,
            full_plate: bool = False
        ):

        if group_id_col is None:
            samples_df = samples_df.reset_index()
            group_id_col = "index"

        # columns that will be added
        group_size_col = "group_size"
        assigned_plate_col = "assigned_plate"

        # group size 
        samples_df[group_size_col] = samples_df.groupby(group_id_col)[group_id_col].transform("count")

        # group ids
        groups = samples_df[group_id_col].unique()

        # Number of plates needed
        n_plates = -(-samples_df.shape[0] // max_samples_per_plate)

        # STEP 1: Sort dataframe based on groups and variables to be balanced
        samples_df = samples_df.set_index(group_id_col)

        samples_df = samples_df.sort_values(by=[group_size_col, *balance_columns])
        samples_df = samples_df.reset_index(drop=False)
        
        # STEP 2: assign samples to plates
        sublists = [[] for _ in range(n_plates)] 
        for i, group in enumerate(groups):
            df_group = samples_df[samples_df[group_id_col] == group]
            sublists[i % n_plates].append(df_group)

        # concatenate to one df per plate
        batches_df = []
        for df_i in sublists:
            batches_df.append(pd.concat(df_i) )

        # permute the plate order
        np.random.shuffle(batches_df)

        # add plate id column
        for i, df in enumerate(batches_df):
            df[assigned_plate_col] = i
            batches_df[i] = df


        # Shuffle groups within each assigned plate
        shuffled_df = pd.DataFrame()
        for i, df in enumerate(batches_df):

            unique_groups_in_plate = df[group_id_col].unique()
            np.random.shuffle(unique_groups_in_plate)
            
            # Concatenating shuffled groups
            shuffled_plate_df = pd.concat([df[df[group_id_col] == group] for group in unique_groups_in_plate])

            batches_df[i] = shuffled_plate_df

            shuffled_df = pd.concat([shuffled_df, shuffled_plate_df])

        samples_df = shuffled_df.reset_index(drop=True)


        return samples_df


    
    
    
    @staticmethod
    def assign_plates_ilp(
            samples_df: pd.DataFrame,
            balance_columns: list,
            max_samples_per_plate: int,
            group_column: str = None,
            tolerance: Union[int, str] = None,
            full_plate: bool = False
        ) -> Tuple[pd.DataFrame, int]:

        samples_df = samples_df.copy()

        if group_column is None:
            samples_df = samples_df.reset_index()
            group_column = "index"

        if tolerance is None:
            tolerance = "group_min"

        if isinstance(tolerance, str):
            if not tolerance in ["group_avg", "group_min"]:
                raise ValueError(f"Tolerance mode {tolerance} not available. Current modes: 'group'")


        # Calculate the required number of plates
        total_samples = len(samples_df)
        total_plates = -(-total_samples // max_samples_per_plate)

        # Calculate the total capacity of all full plates
        full_plate_count = total_samples // max_samples_per_plate if full_plate else 0
        remaining_plates = total_plates - full_plate_count
        full_plate_count = total_samples // max_samples_per_plate

        logger.info(f"Number of samples: {total_samples}")
        logger.info(f"Max samples per plate: {max_samples_per_plate}")

        logger.info(f"Total plates needed: {total_plates}")
        logger.info(f"Fill plates: {full_plate}")
        logger.info(f"Full plates count: {full_plate_count}")


        # Initialize the ILP problem
        linprob = lp.LpProblem("GroupedBalanceDistribution", lp.LpMinimize)

        # Permute sample groups for randomness
        unique_groups = samples_df[group_column].unique()
        np.random.shuffle(unique_groups)

        # Define binary decision variables for group-to-plate assignments
        assign_group = lp.LpVariable.dicts("assignGroup",
                                           (unique_groups, range(total_plates)),
                                           cat='Binary')

        # Ensure each group is assigned to exactly one plate
        for g in unique_groups:
            linprob += lp.lpSum(assign_group[g][p] for p in range(total_plates)) == 1, f"Group_{g}_SingleAssignment"

        # Handling full plates if enabled
        if full_plate:
            
            # For <full_plate_count> plates,  apply full plate constraint 
            for p in range(full_plate_count):
                linprob += lp.lpSum(assign_group[g][p] * len(samples_df[samples_df[group_column] == g]) for g in unique_groups) == max_samples_per_plate, f"FullPlate_{p}"

        else:
            # For all plates, apply less than or equal max samples per plate constraint
            for p in range(total_plates):
                linprob += lp.lpSum(assign_group[g][p] * len(samples_df[samples_df[group_column] == g]) for g in unique_groups) <= max_samples_per_plate, f"MaxPlate_{p}"


        # Balance constraints
        for balance_column in balance_columns:

            balance_categories = samples_df[balance_column].unique()

            # Calculate category counts and proportions
            category_counts = samples_df[balance_column].value_counts()
            category_proportions = samples_df[balance_column].value_counts(normalize=True)

            logger.debug(f"Category counts: {category_counts}")

            expected_full_plate_counts = category_proportions * max_samples_per_plate

            remaining_sample_counts = category_counts - expected_full_plate_counts * full_plate_count
            expected_remaining_plate_counts = remaining_sample_counts / remaining_plates if remaining_plates > 0 else 0

            expected_equal_plate_counts = category_counts / total_plates
    

            # Combine expected allocations for full and remaining plates
            expected_per_plate = {
                'full': expected_full_plate_counts,
                'remaining': expected_remaining_plate_counts,
                'equal': expected_equal_plate_counts
            }

            logger.debug(f"Category counts: {expected_per_plate}")

            all_category_tolerances = []

            for category in balance_categories:

                category_groups = samples_df[samples_df[balance_column] == category][group_column].unique()
                
                for p in range(total_plates):
                    if full_plate:
                        plate_type = 'full' if p < full_plate_count else 'remaining'
                    else:
                        plate_type = 'equal'

                    expected_allocation = int(np.round(expected_per_plate[plate_type][category]))

                    match tolerance:
                        case "group_avg":
                            category_tolerance = samples_df[samples_df[balance_column] == category].groupby(group_column).size().mean()

                        case "group_min":
                            category_tolerance = samples_df[samples_df[balance_column] == category].groupby(group_column).size().min()

                        case _:
                            category_tolerance = tolerance
            
                    category_tolerance = int(category_tolerance)

                    all_category_tolerances.append(category_tolerance)

                    minbound = expected_allocation - category_tolerance
                    maxbound = expected_allocation + category_tolerance

                    linprob += lp.lpSum(assign_group[g][p] * len(samples_df[samples_df[group_column] == g]) for g in category_groups) >= int(minbound), f"Min_{balance_column}_{category}_{p}_{plate_type}"
                    linprob += lp.lpSum(assign_group[g][p] * len(samples_df[samples_df[group_column] == g]) for g in category_groups) <= int(maxbound), f"Max_{balance_column}_{category}_{p}_{plate_type}"

            logger.debug(f"Category tolerances for {balance_column}")
            logger.debug(f"Min tolerances used: {min(all_category_tolerances)}")
            logger.debug(f"Max tolerances used: {max(all_category_tolerances)}")

        # Solve the problem
        linprob.solve()

        wall_time = linprob.solutionTime

        if linprob.sol_status != lp.LpStatusOptimal:
            logger.error(f"Solver could not distribute samples to plates within defined constraints.")
            sol_status = False
        else:
            logger.info("Optimal solution found")
            logger.info(f"PuLP solver wall time: {wall_time}")
            sol_status = True
            
        # Extract group assignments and update the DataFrame
        group_assignments = {g: p for g in unique_groups for p in range(total_plates) if lp.value(assign_group[g][p]) == 1}
        samples_df['assigned_plate'] = samples_df[group_column].map(group_assignments)

        # Shuffle groups within each assigned plate
        shuffled_df = pd.DataFrame()
        for plate in range(total_plates):
            plate_df = samples_df[samples_df['assigned_plate'] == plate]
            unique_groups_in_plate = plate_df[group_column].unique()
            np.random.shuffle(unique_groups_in_plate)
            
            # Concatenating shuffled groups
            shuffled_plate_df = pd.concat([plate_df[plate_df[group_column] == group] for group in unique_groups_in_plate])
            shuffled_df = pd.concat([shuffled_df, shuffled_plate_df])

        samples_df = shuffled_df.reset_index(drop=True)

        return samples_df, sol_status
    
    @staticmethod
    def create_blocks_within_plate_ilp(
        samples_df,
        blocking_column,
        group_column,
        initial_tolerance=0,
        max_attempts=20,
        block_shuffle=True
    ) -> Tuple[pd.DataFrame, int]:

        def find_whole_number_ratios(distribution):
            decimal_places = 1
            while True:
                rounded_distribution = {k: round(v, decimal_places) for k, v in distribution.items()}
                base = 10 ** decimal_places
                ratios = {k: int(v * base) for k, v in rounded_distribution.items()}
                if all(ratio == round(ratio) for ratio in ratios.values()):
                    gcd = np.gcd.reduce(list(ratios.values()))
                    normalized_ratios = {k: v // gcd for k, v in ratios.items()}
                    return normalized_ratios
                decimal_places += 1

        # Ensure group_column is in DataFrame
        if group_column not in samples_df.columns:
            raise ValueError(f"{group_column} is not a column in the DataFrame")
        
        logger.debug(f"Blocking variable: {blocking_column} ")

        # Get proportions of each category and calculate ratios
        category_counts = samples_df[blocking_column].value_counts(normalize=False)
        category_proportions = samples_df[blocking_column].value_counts(normalize=True)

        block_ratios = find_whole_number_ratios(category_proportions)

        logger.debug(f"Category counts in plate: {category_counts}")
        logger.debug(f"Total number of samples: {category_counts.sum()}")

        logger.debug(f"Category proportions in plate: {category_proportions}")
        logger.debug(f"Ideal ratios in each block: {block_ratios}")

        basic_block_size = sum(block_ratios.values())

        # Get number of samples in each group 
        group_sizes = samples_df.groupby(group_column).size()
        group_size = int(group_sizes.max())

        logger.debug(f"Max sample group size: {group_size}")

        block_size = basic_block_size * group_size

        block_ratios_list = []
        for key, val in category_counts.items():
            ratio = block_ratios[key]
            block_ratios_list.append(val / (ratio * group_size))

        full_blocks = int(np.floor(min(block_ratios_list)))

        logger.debug(f"Number of blocks with ideal ratios: {full_blocks}")

        # Group the samples
        grouped_samples = samples_df.groupby(group_column)

        num_blocks = int(np.ceil(len(samples_df) / block_size))

        attempt = 0

        tolerance = initial_tolerance
        solution_found = False

        while not solution_found and attempt < max_attempts:
            # Reinitialize the ILP problem for each attempt
            prob = lp.LpProblem("GroupedBlockCreation", lp.LpMinimize)
            
            # Define binary decision variables
            group_ids = grouped_samples.groups.keys()

            assign_group = lp.LpVariable.dicts("AssignGroup", [(group_id, j) for group_id in group_ids for j in range(num_blocks)], cat='Binary')

            try:
                # Constraint: Each group should be assigned to exactly one block
                for group_id in group_ids:
                    prob += lp.lpSum(assign_group[(group_id, j)] for j in range(num_blocks)) == 1

                # Constraint: Each block should have at most the specified number of samples
                for j in range(num_blocks):
                    prob += lp.lpSum(assign_group[(group_id, j)] * len(grouped_samples.get_group(group_id)) for group_id in group_ids) <= block_size

                # Adjusted constraints for the blocks with expected ratios
                for j in range(full_blocks):
                    for category, count in category_counts.items():
                        cat_groups = samples_df[samples_df[blocking_column] == category].groupby(group_column)
                        cat_group_ids = cat_groups.groups.keys()

                        # Adjusted constraint considering tolerance
                        required_groups = block_ratios[category]
                        prob += lp.lpSum(assign_group[(group_id, j)] for group_id in cat_group_ids) >= required_groups 
                        prob += lp.lpSum(assign_group[(group_id, j)] for group_id in cat_group_ids) <= required_groups

                # # Adjusted constraints for the non-ideal blocks
                # for j in range(full_blocks, num_blocks):
                #     for category, count in category_counts.items():
                #         cat_groups = samples_df[samples_df[blocking_column] == category].groupby(group_column)
                #         cat_group_ids = cat_groups.groups.keys()
                #         required_groups = block_ratios[category]

                #         # Apply tolerance
                #         min_bound = max(0, required_groups - tolerance) # Ensures lower bound is not negative
                #         max_bound = required_groups + tolerance

                #         prob += lp.lpSum(assign_group[(group_id, j)] for group_id in cat_group_ids) >= min_bound
                #         prob += lp.lpSum(assign_group[(group_id, j)] for group_id in cat_group_ids) <= max_bound

                prob.solve()
            except Exception as e:
                print(e)

            if prob.sol_status == 1:  # Check if the solution is optimal
                solution_found = True
            else:
                tolerance += 1  # Increase tolerance for the next attempt
                attempt += 1

            if not solution_found:
                logger.error("Failed to find an optimal solution within the maximum number of attempts.")
                samples_df['assigned_block'] = 0
                return samples_df, prob.sol_status
            
        logger.info(f"Solution found using tolerance = {tolerance}")
        
        # Extract group assignments and update the DataFrame
        group_block_assignments = {group_id: j for group_id in group_ids for j in range(num_blocks) if lp.value(assign_group[(group_id, j)]) == 1}
        samples_df['assigned_block'] = samples_df[group_column].map(group_block_assignments)
        
        samples_df=samples_df.sort_values(by=['assigned_block', group_column])

        if block_shuffle:
            # Shuffle samples within each block
            block_shuffled_df = pd.DataFrame()

            logger.info(f"Randomizing sample order within each block")

            for block in samples_df['assigned_block'].unique():
                block_df = samples_df[samples_df['assigned_block'] == block].copy()
                block_df = block_df.sample(frac=1).reset_index(drop=True)  # Shuffle the block
                block_shuffled_df = pd.concat([block_shuffled_df, block_df], ignore_index=True)
            samples_df =  block_shuffled_df

        samples_df = samples_df.reset_index(drop=True)
        
        return samples_df, prob.sol_status

   
    def randomize_order(self, case_control : bool = None, reproducible=True) -> None:
        """
        Randomizes the order of specimen records in the study, optionally maintaining group integrity.

        This method either randomizes the entire order of specimens or maintains the order within groups,
        depending on the 'case_control' flag. It also allows for reproducible randomization using a fixed seed.

        Args:
            case_control (Optional[bool]): If True, maintains group order (samples within a group are not shuffled).
                If False, shuffles all samples regardless of group. If None, the behavior is determined based on 
                the presence of a group index column.
            reproducible (bool): If True, uses a fixed seed for randomization to ensure reproducibility.

        Examples:
            >>> study = Study(study_name="Diabetes Study")
            >>> study.load_specimen_records("patients.csv", sample_group_id_column="GroupID")
            >>> study.randomize_order(case_control=True)
            >>> study.specimen_records_df.head(3)  # Example output showing randomized order within groups.
        """
        
        if not len(self.sample_records_df) > 0:
            logger.error("There are no study records loaded. Use 'load_specimen_records' method to import study records.")
            return
        
        if case_control is None:
            if self._column_with_group_index:
                case_control = True
            else:
                case_control = False
        
        specimen_records_df_copy = self.sample_records_df.copy()
        
        if case_control:
            column_with_group_index = self._column_with_group_index
                        
            logger.debug(f"Randomly permuting group order (samples within group unchanged) using variable '{column_with_group_index}'")
            logger.debug("Creating multiindex dataframe")
            specimen_records_df_copy = specimen_records_df_copy.set_index([column_with_group_index, specimen_records_df_copy.index])
            drop = False
        else:
            logger.debug(f"Randomly permuting sample order.")
            specimen_records_df_copy = specimen_records_df_copy.set_index([specimen_records_df_copy.index, specimen_records_df_copy.index])
            column_with_group_index = 0
            drop = True
            
            
        group_IDs = np.unique(specimen_records_df_copy.index.get_level_values(0))

        # Permute order in table
        if reproducible:
            logger.info(f"Using a fixed seed to random number generator for reproducibility; \
                running this method will always give the same result.")
            logger.debug(f"Using class-determined seed {self._default_seed} for random number generator")
            np.random.seed(self._default_seed)

        permutation_order = np.random.permutation(group_IDs)
        
        prev_index_str = "index_before_permutation"
        
        # if multiple randomization rounds, remove old column = prev_index_str 
        if prev_index_str in specimen_records_df_copy.columns:
            specimen_records_df_copy = specimen_records_df_copy.drop(columns=prev_index_str)
        
        specimen_records_df_copy = specimen_records_df_copy \
                                    .loc[permutation_order]\
                                    .reset_index(level=column_with_group_index, drop=drop)\
                                    .reset_index(drop=False)
        
        specimen_records_df_copy = specimen_records_df_copy.rename(columns = {"index": "index_before_permutation"})

        self._N_permutations += 1
        self.sample_records_df = specimen_records_df_copy.copy()

    @staticmethod
    def _get_attribute_distribution(df: pd.DataFrame, attribute, ignore_nans=True, normalize=True):
        if ignore_nans:
            df = df.replace("NaN", pd.NA)
            df = df.dropna()

        distribution = df[attribute].value_counts(normalize=normalize)
        return distribution

    
    def get_attribute_plate_distributions(self, attribute, ignore_nans=True, normalize=True, long_format=True) -> dict:
        plate_distributions = {}

        for plate in self.plates:
            df = plate.as_dataframe()
            distribution = self._get_attribute_distribution(df, attribute, ignore_nans, False)
            plate_distributions[plate.plate_id] = distribution

        df = pd.DataFrame(plate_distributions).fillna(0)

        if normalize:
            # Calculate the sum of counts for each category across all plates
            category_totals = df.sum(axis=1)
            # Normalize the data frame by dividing by the category totals
            df = df.div(category_totals, axis=0) * 100

        if long_format:
            value_name = "Percentage" if normalize else "Counts"
            df = df.reset_index().melt(id_vars=attribute, var_name='Plate', value_name=value_name)
    

        return df
    
    def plot_attribute_plate_distributions(self, attribute, normalize=False, colormap='tab20b', plt_style="ggplot"):
        """
        Plots a stacked bar chart for a specified attribute across different plates.

        This method retrieves distribution data for the given attribute and plots it 
        as a stacked bar chart. Each bar in the chart represents a different category 
        of the attribute, with segments in the bar showing the count or proportion 
        from each plate. The method supports normalization of the data and allows for 
        customization of the plot's colormap.

        Args:
            attribute (str): The attribute for which the distributions are plotted.
            normalize (bool, optional): If True, normalizes the counts within each 
                category to proportions that sum to 100%. Defaults to False.
            colormap (str, optional): The name of the matplotlib colormap to use for 
                the plot. Defaults to 'tab20b'.

        Returns:
            matplotlib.figure.Figure: The figure object containing the bar chart.
        """

        df = self.get_attribute_plate_distributions(attribute=attribute, normalize=normalize, long_format=False)

        # Plotting the stacked bar chart
        plt.style.use(plt_style)
        fig, ax = plt.subplots()

        # Apply the chosen colormap
        df.plot(kind='bar', stacked=True, ax=ax, colormap=colormap)

        # Set titles and labels
        ax.set_title(f"Counts of {attribute} across plates" + (" (Normalized)" if normalize else ""))
        ax.set_xlabel(f"{attribute}")
        ax.set_ylabel('Proportion (%)' if normalize else 'Counts')

        # Place the legend outside of the plotting area
        ax.legend(title="Plates", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Rotate and align x-axis labels
        plt.xticks(rotation=45, ha='right')

        return fig
    
    def plot_attribute_distributions_plotly(
            self,
            attribute,
            normalize=True,
            barmode = "stack",
            title=None):

        df = self.get_attribute_plate_distributions(attribute, ignore_nans=True, normalize=True)

        if normalize:
            df['PercentageText'] = df['Percentage'].apply(lambda x: f"{x:.1f}%")  # Adjust decimal places as needed

        if title is None:
            title = f"{attribute} distributions across plates"

        fig = px.bar(df, x='organ', y='Percentage', color='Plate', text='PercentageText',
                    title=title, labels={'index': attribute, 'Percentage': 'Percentage (%)'},
                    barmode=barmode, height=400)

        # Update layout for clarity
        fig.update_layout(xaxis_title=attribute,
                        yaxis_title='Percentage (%)' if normalize else 'Count',
                        legend_title="Plate ID",
                        template="plotly_white"
                        )

        # color_discrete_sequence=px.colors.sequential.Agsunset

        fig.update_traces(textposition='outside')

        return fig

    def to_dict(self) -> dict:
        study_dict = {
            "name": self.name,
            "plates": [plate.as_dict() for plate in self.plates],  # Assuming Plate objects have a as_dict method
            "total_plates": self.total_plates,
            "sample_records_df": self.sample_records_df.to_dict("records") 
        }
        return study_dict
    
    def dict_to_study(study_dict: dict) -> 'Study':
        study = Study(name=study_dict.get("name"))
        study.plates = [PlateFactory.dict_to_plate(plate_data) for plate_data in study_dict.get("plates", [])]
        study.total_plates = study_dict.get("total_plates", 0)
        study.sample_records_df = pd.DataFrame(study_dict.get("sample_records_df", {}))
        return study
    

    @staticmethod
    def report_plate_imbalance(df, plate_assign_col, category_col):
        df = df.dropna()
        categories = df[category_col].unique()
        plates = df[plate_assign_col].unique()
        plate_cat_residuals_dict = {}

        for category in categories:
            total_samples_in_category = len(df[df[category_col] == category])
            expected_frequency = np.round(total_samples_in_category / len(plates))
           
            # Initialize a dict to hold observed frequencies for each plate with default of 0
            observed_frequency_dict = dict.fromkeys(plates, 0)
            # Update with actual observed frequencies
            observed_frequencies = df[df[category_col] == category][plate_assign_col].value_counts()
            for plate, count in observed_frequencies.items():
                observed_frequency_dict[plate] = count

            # Convert to list ensuring all plates are represented
            observed_frequency_list = [observed_frequency_dict[plate] for plate in plates]

            # Calculate residuals
            plate_category_residuals = [abs(of - expected_frequency) for of in observed_frequency_list]
            plate_cat_residuals_dict[category] = plate_category_residuals

        imbalance_df = pd.DataFrame(plate_cat_residuals_dict)
        mar = imbalance_df.mean(axis=1)
        sar = imbalance_df.sum(axis=1)
        max_ar = imbalance_df.max(axis=1)

        imbalance_df.insert(0, "plate", value=imbalance_df.index.values)
        imbalance_df['mean'] = mar
        imbalance_df['sum'] = sar
        imbalance_df['max'] = max_ar

        return imbalance_df
    
    
    def plate_balance_chi_square_test(self, plate_assign_col, category_columns, exclude_plates = None):
        """
        The commonly accepted rule of thumb is that a chi2 test may not be reliable if more than 20% of the expected frequencies are less than 5, or any expected frequencies are less than 1.
        """

        df = self.to_dataframe().dropna()

        if exclude_plates is not None:
            df = df[~df[plate_assign_col].isin(exclude_plates)]

        all_observed_frequencies = []
        all_expected_frequencies = []

        total_samples = len(df)
        total_plates = len(df[plate_assign_col].unique())

        for category_col in category_columns:
            levels = df[category_col].unique()
            
            for level in levels:
                level_df = df[df[category_col] == level]
                total_samples_in_level = len(level_df)

                expected_frequency = total_samples_in_level / total_plates
                expected_frequencies = [expected_frequency] * total_plates
                all_expected_frequencies.extend(expected_frequencies)

                observed_frequencies = level_df[plate_assign_col].value_counts().sort_index().values
                observed_frequencies = np.array(observed_frequencies if len(observed_frequencies) == total_plates else np.append(observed_frequencies, [0] * (total_plates - len(observed_frequencies))))
                all_observed_frequencies.extend(observed_frequencies)

        all_observed_frequencies = np.reshape(all_observed_frequencies, (len(all_expected_frequencies) // total_plates, total_plates))
        
        all_expected_frequencies = np.reshape(all_expected_frequencies, all_observed_frequencies.shape)

        chi2, p, dof, expected = chi2_contingency(all_observed_frequencies)

        n = np.sum(all_observed_frequencies)  # Total number of samples
        r = all_observed_frequencies.shape[0]  # Number of rows (categories and levels combined)
        c = total_plates  # Number of columns (plates)
        
        cramer_v = np.sqrt(chi2 / (n * (min(r - 1, c - 1))))

        low_expected_count = sum([freq < 5 for freq in all_expected_frequencies.flatten()])
        warning_message = ""
        if low_expected_count > 0.2 * len(all_expected_frequencies.flatten()) or min(all_expected_frequencies.flatten()) < 1:
            warning_message = "Warning: Chi-squared test may not be reliable."

        return {
            'chi2': chi2,
            'p_value': p,
            'cramer_v': cramer_v,
            'warning': warning_message
        }
    
    


                    