from pathlib import Path
import copy
import datetime
from typing import Union, Iterator, Any

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from plate_planner.plate import Plate, QCPlate
from plate_planner.logger import logger

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
    
    def __init__(self, study_name=None,) -> None:
        """
        Initializes a new instance of the Study class.

        This constructor sets up a study with a specified name. If no name is provided, a default name 
        is generated using the current date.

        Args:
            study_name (Optional[str]): The name for the study. If None, a default name in the format 
                "Study_YYYY-MM-DD" is assigned, where YYYY-MM-DD represents the current date.

        Examples:
            >>> study1 = Study(study_name="Alzheimer's Research")
            >>> study1.name
            "Alzheimer's Research"
            
            >>> study2 = Study()
            >>> study2.name
            "Study_2024-01-21"  # Example output; actual output will vary based on the current date.
        """
        
        if study_name is None:
            study_name = f"Study_{datetime.date}"
            
        self.name = study_name
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
        
    def load_sample_file(self, records_file: str, sample_group_id_column=None, sample_id_column=None) -> None:
        """
        Loads specimen records from a specified file into the study.

        This method reads specimen data from a file (Excel or CSV) and stores it in a DataFrame.
        It also identifies or sets the column used for grouping specimens.

        Args:
            records_file (str): The path to the file containing specimen records.
            sample_group_id_column (Optional[str]): The column name in the file that represents the group ID of samples. 
                If None, the method attempts to find a suitable column automatically.

        Raises:
            FileExistsError: If the specified records_file does not exist.

        Examples:
            >>> study = Study(study_name="Oncology Study")
            >>> study.load_specimen_records("specimens.xlsx", sample_group_id_column="PatientGroup")
            >>> study.specimen_records_df.shape
            (200, 5)  # Example output, indicating 200 rows and 5 columns in the DataFrame.
        """
        self.records_file_path = records_file
        records_path = Path(records_file)

        logger.debug(f"Loading records file: {records_file}")
        extension = records_path.suffix
        
        if not records_path.exists():
            logger.error(f"Could not find file {records_file}")
            raise FileExistsError(records_file)
        
        if extension in [".xlsx", ".xls"]:
            logger.debug("Importing Excel file.")
            records = pd.read_excel(records_file)
        elif extension == ".csv":
            logger.debug("Importing csv file.")
            records = pd.read_csv(records_file)
        else:
            logger.error("File extension not recognized")
            records = pd.DataFrame()
            
        if sample_group_id_column is None:
            self._column_with_group_index = Study._find_column_with_group_index(records)
        else:
            self._column_with_group_index = sample_group_id_column
        
        logger.debug(f"{records.shape[0]} specimens in file")
        logger.info("Metadata in file:")
        for col in records.columns:
            logger.info(f"\t{col}")

        if sample_id_column:
            logger.debug(f"Sorting records in ascending order based on column '{sample_id_column}'")
            records = records.sort_values(by=[sample_id_column])
            
        # if self._column_with_group_index:
        #     logger.debug(f"Sorting records in ascending order based on column '{self._column_with_group_index}'")
            # records = records.sort_values(by=[self._column_with_group_index])
        
        self.specimen_records_df = records

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
            grouped = self.specimen_records_df.groupby(self._column_with_group_index)

            # Step 2: Sort each group by the 'sortby_column' column
            sorted_groups = [group.sort_values(sortby_column) for _, group in grouped]

            # Step 3: Concatenate the sorted groups back into a single DataFrame
            self.specimen_records_df = pd.concat(sorted_groups)

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
            grouped = self.specimen_records_df.groupby(self._column_with_group_index)

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
            self.specimen_records_df = pd.concat(modified_groups).reset_index(drop=True)

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
                        file_format : str = "pdf",
                        folder_path : str = None,
                        plate_name : str = "Plate", **kwargs) -> None:
        """
        Creates and saves visual representations of each plate in the study as figures.

        This method iterates over each plate in the study, generating a figure based on specified metadata keys
        for annotation and coloring. The figures are saved in the specified file format in a designated folder.

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
            
            fig.savefig(file_path)
    
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
        remaining_specimens = self.specimen_records_df.copy()

        N_specimens = self.specimen_records_df.shape[0]

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
        
        if not len(self.specimen_records_df) > 0:
            logger.error("There are no study records loaded. Use 'load_specimen_records' method to import study records.")
            return
        
        if case_control is None:
            if self._column_with_group_index:
                case_control = True
            else:
                case_control = False
        
        specimen_records_df_copy = self.specimen_records_df.copy()
        
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
        self.specimen_records_df = specimen_records_df_copy.copy()

    @staticmethod
    def _get_attribute_distribution(df: pd.DataFrame, attribute, ignore_nans=True, normalize=True):
        if ignore_nans:
            df = df.replace("NaN", pd.NA)
            df = df.dropna()

        distribution = df[attribute].value_counts(normalize=normalize)
        return distribution

    def randomize_with_uniformity_check(self, case_control, attribute, samples_per_plate, uniformity_criterion, max_attempts = 10, reproducible=False):
        attempt = 0
        while attempt < max_attempts:
            self.randomize_order(case_control, reproducible)
            if self._uniformity_within_tolerance(attribute, samples_per_plate, uniformity_criterion):
                return
            attempt += 1
        raise Exception(f"Unable to achieve uniform distribution after {max_attempts} attempts")
    
    def _uniformity_within_tolerance(self, attribute, block_size, tolerance):
        overall_distribution = self._get_attribute_distribution(self.specimen_records_df, attribute)
        for start_idx in range(0, len(self.specimen_records_df), block_size):
            end_idx = start_idx + block_size
            block = self.specimen_records_df.iloc[start_idx:end_idx]
            block_distribution = block[attribute].value_counts(normalize=True)

            if not self._meets_criterion(block_distribution, overall_distribution, tolerance):
                return False
        return True

    def _meets_criterion(self, block_distribution, overall_distribution, tolerance):
        for category in overall_distribution.index:
            overall_percent = overall_distribution[category] * 100
            block_percent = block_distribution.get(category, 0) * 100  # Default to 0 if category not in block
            if abs(block_percent - overall_percent) > tolerance:
                return False
        return True
    
    def get_attribute_plate_distributions(self, attribute, ignore_nans=True, normalize=True) -> dict:
        plate_distributions = {}

        for plate in self.plates:
            df = plate.as_dataframe()
            distribution = self._get_attribute_distribution(df, attribute, ignore_nans, normalize)
            plate_distributions[plate.plate_id] = distribution

        return plate_distributions
    
    def plot_attribute_plate_distributions(self, attribute, normalize=False, colormap='tab20b'):
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

        distributions = self.get_attribute_plate_distributions(attribute=attribute, normalize=False)

        # Convert the dictionary to a DataFrame and rename columns
        df = pd.DataFrame(distributions)
        df.columns = [f"plate_{key}" for key in distributions.keys()]

        if normalize:
            # Normalize each column to sum to 100%
            df = df.div(df.sum(axis=1), axis=0) * 100

        # Plotting the stacked bar chart
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

                