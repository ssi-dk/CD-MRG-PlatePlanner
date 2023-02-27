import numpy as np
import pandas as pd
import itertools, os, tomli, glob
import matplotlib.pyplot as plt
import matplotlib as mpl

import string
from .logger import logger

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
             
       

class Well:
    """
    A class to represent a well in a multiwell plate. 
    """
    name : str
    coordinate : tuple
    metadata : dict
    
    def __init__(self,
                 name="", 
                 coordinate=(int, int), 
                 index = None,
                 plate_id = None,
                 metadata = None) -> None:
        
        self.name = name
        self.coordinate = coordinate
        
        if metadata is None:
            metadata = {"index": index,
                    "plate_id": plate_id}
    
        self.metadata = metadata
     
    def __str__(self) -> str:
        return f"name: {self.name}\ncoordinate: {self.coordinate}\nmetadata: {self.metadata}"
    
    def __repr__(self) -> str:
        return f"Well(name={self.name}, coordinate={self.coordinate})"


class Plate:
    
    """_summary_
    A class to represent a multiwell plate. 
    
    Attributes
    
    rows
    columns
    well_coordinates
    well_names
    
    plate_well_names
    plate_well_coordinates
    
    Methods
    
    
    """
    
    specimen_code : str = "S"
    specimen_base_name : str = "Specimen"
    
    # "public"
    plate_id : int
    rows: list 
    columns: list 
    wells: list # list of well objects
    
    # "private"
    _coordinates: list # list of tuples with a pair of ints (row, column)
    _alphanumerical_coordinates: list # list of strings for canonical naming of plate coordnates, i.e A1, A2, A3, ..., B1, B2, etc
    _coordinates2index : dict # map well index to well coordinate
    _name2index : dict # map well index to well name
    
    
    def __init__(self,  *args, plate_id=1, **kwargs) -> None:    
        """_summary_
        
        Constructs all the necessary attributes for the Plate object, given the specifications in the plate.toml file
        
        """
        
        if args:
            rows = args[0][0]
            columns = args[0][1]
        
        if kwargs:
            rows = kwargs["rows"]
            columns = kwargs["columns"]
            
        self.plate_id = plate_id
        
        self.rows = list(range(0,rows))
        self.columns = list(range(0,columns))
        self.capacity = len(self.rows) * len(self.columns)
      
        
        self._coordinates = Plate.create_index_coordinates(self.rows, self.columns)
        self._alphanumerical_coordinates = Plate.create_alphanumerical_coordinates(self.rows, self.columns)
        
        self.define_empty_wells()
        
        logger.info(f"Created a plate with {len(self)} wells:")
        logger.debug(f"Canonical well coordinate:\n{self}")
        logger.debug(f"Well index coordinates:\n{self.plate_to_numpy_array(self._coordinates)}")

        
    @staticmethod    
    def create_index_coordinates(rows, columns) -> list:
        # count from left to right, starting at well in top left
        return list(itertools.product(
                                    range(len(rows)-1, -1, -1),
                                    range(0, len(columns))
                                    )
                                    )
    @staticmethod
    def create_alphanumerical_coordinates(rows, columns) -> list:
        alphabet = list(string.ascii_uppercase)
        
        row_crds = alphabet
        number_of_repeats = 1
        
        while len(rows) > len(row_crds):
            row_crds = list(itertools.product(alphabet, repeat=number_of_repeats))
            number_of_repeats += 1
            
        alphanumerical_coordinates = []
        for r_i, row in enumerate(rows):
            for c_i, column in enumerate(columns):
                #an_crd = row_crds[r_i], column
                r_str = str()
                for r in row_crds[r_i]:
                    r_str = r_str.join(r)
                
                alphanumerical_coordinates.append(f"{r_str}_{c_i+1}")
        
        return alphanumerical_coordinates
        
    
    def define_empty_wells(self):
        
        wells = []
        well_crd_to_index_map = {}
        well_name_to_index_map = {}
        
        for i, crd in enumerate(zip(self._coordinates, self._alphanumerical_coordinates)):
            index_crd = crd[0]
            name_crd = crd[1]
            
            wells.append(
                Well(coordinate=index_crd,
                     name=name_crd,
                     index=i, 
                     plate_id=self.plate_id)
            )
            
            well_crd_to_index_map[index_crd] = i
            well_name_to_index_map[name_crd] = i
            
        self.wells = wells
        self._coordinates2index = well_crd_to_index_map
        self._name2index = well_name_to_index_map
        
          
    def __len__(self) -> int:
        return len(self.wells)
    
    def __contains__(self):
        # TODO
        pass
    
    def __getitem__(self, key) -> object:
        
        key_type = type(key)
        
        if key_type is str:
            index = self._name2index[key]
        elif key_type is int:
            index = key
        elif key_type is tuple:
            index = self._coordinates2index[key]
        else:
            raise KeyError(key)
        
            
        return self.wells[index]
        
    def __setitem__(self, key, well_object) -> None:
        
        key_type = type(key)
        
        if key_type is str:
            index = self._name2index[key]
        elif key_type is int:
            index = key
        elif key_type is tuple:
            index = self._coordinates2index[key]
        else:
            raise KeyError(key)
        
        self.wells[index] = well_object
        self.update_well_position(index)
        
        
    def __delitem__(self):
        # TODO
        pass
    
    def __str__(self):
        return f"{self.plate_to_numpy_array(self._alphanumerical_coordinates)}"
    
    def update_well_position(self, index):
        self.wells[index].name = self._alphanumerical_coordinates[index]
        self.wells[index].coordinate = self._coordinates[index]
        self.wells[index].metadata["index"] = index
        
    
    def plate_to_numpy_array(self, data: list) -> object:
         
        plate_array = np.empty((len(self.rows), len(self.columns)), 
                               dtype=object)

        for i, well in enumerate(self._coordinates):
            r, c = well[0], well[1]
            plate_array[r][c] = data[i]

        return np.flipud(plate_array)
    
    
    
# A plate with QC samples is a subclass of a Plate
class QCplate(Plate):
    """_summary_
    Class that represents a multiwell plate where some wells should 
    contain quality control samples according to the scheme defined 
    in <config_file.toml>

    Args:
        Plate (_type_): _description_
    """
    
    def __init__(self, config_file = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.configfile = config_file
        
        self.load_config_file(config_file)
        self.create_plate_layout()
        
        
    def __str__(self):
        return f"{self.plate_to_numpy_array(self._alphanumerical_coordinates)}"
    
    
    def __repr__(self):
        pass
       
       
    def load_config_file(self, config_file: str = None):
        
        # READ CONFIG FILE
        if config_file is None: 
            
            logger.warning("No config file specified. Trying to find a toml file in current folder.")
            
            config_file_search = glob.glob("*.toml")      
            
            if config_file_search:
                config_file = config_file_search[0]
                logger.info(f"Using toml file '{config_file}'")
  
        
        try:
            with open(config_file, mode="rb") as fp:
                
                self.config = tomli.load(fp)
            
            logger.info(f"Successfully loaded config file {config_file}")
            logger.debug(f"{self.config}")
            
        except FileNotFoundError:
            logger.error(f"Could not find/open config file {config_file}")


    
    def define_unique_QC_sequences(self):
        """_summary_

        """
        
        # Check if QC scheme defined in config file
        logger.debug("Setting up QC scheme from config file")
        
        self.N_specimens_between_QC = self.config['QC']['run_QC_after_n_specimens']
        QCscheme = self.config['QC']['scheme']
       
        self.N_QC_samples_per_round = np.max(
            [QCscheme[key]['position_in_round'] for key in QCscheme.keys()]
            )
        
        self.N_unique_QCrounds = np.max(
            [QCscheme[key]['every_n_rounds'] for key in QCscheme.keys()]
        )
        # <- end if 

        # Generate QC sequence list in each unique QC round
        QC_unique_seq = []
        for m in range(0, self.N_unique_QCrounds):

            m_seq = []

            for key in QCscheme:
                order = QCscheme[key]['position_in_round']

                if QCscheme[key]["introduce_in_round"] == m + 1:
                    m_seq.append((key, order))

                if QCscheme[key]["introduce_in_round"] == 0:
                    m_seq.append((key, order))
            # <- end for loop

            m_seq.sort(key=lambda x: x[1])
            QC_unique_seq.append(m_seq)
            
        # <- end for loop
        
        logger.debug(f"{len(QC_unique_seq)} unique QC sequences defined: ")
        for i, qc_seq in enumerate(QC_unique_seq):
            logger.debug(f"\t{i+1}) {qc_seq}")
            
        self._QC_unique_seq = QC_unique_seq
        

    def define_QC_rounds(self):
        
        #Number of QC rounds per plate
        N_QC_rounds = self.capacity // (self.N_QC_samples_per_round + self.N_specimens_between_QC - 1)

        logger.debug(f"Assigning {N_QC_rounds} QC rounds per plate")

        # Create dict with key = QC round; 
        # value = a dict with well indices and associated sequence of QC samples
        
        QC_rounds = {}
        well_count_start = 0
        round_count = 0
        QC_well_indices = []
        
        while round_count < N_QC_rounds:
            
            if not self.config["QC"]["start_with_QC_round"]: 
                well_count_start += self.N_specimens_between_QC
                
            for sequence in self._QC_unique_seq:
                well_indices = list(range(well_count_start,
                                          well_count_start + len(sequence)))
                round_count += 1
 
                QC_rounds[round_count] = {"sequence": sequence,
                                       "well_index": well_indices}
                QC_well_indices += well_indices
                
                well_count_start += len(sequence) 
                well_count_start += self.N_specimens_between_QC 
                
                logger.debug(f"\t Round {round_count}: {sequence}; wells {well_indices}")
           
        
        self.QC_rounds = QC_rounds
        self._well_inds_QC = QC_well_indices
        self._well_inds_specimens = [w for w in range(0,self.capacity) if w not in QC_well_indices]



    def create_plate_layout(self):
        self.define_unique_QC_sequences()
        self.define_QC_rounds()
        
        counts = {}
        for qc_type in self.config["QC"]["names"].keys():
            counts.setdefault(qc_type, 0)
            
        # set metadata for wells that are for QC samples
        for val in self.QC_rounds.values(): 
            for sample, index in zip(val["sequence"], val["well_index"]):
                self.wells[index].metadata["QC"] = True
                
                sample_code = sample[0]
                counts[sample_code] += 1
                self.wells[index].metadata["sample_code"] = sample_code
                
                sample_type = self.config["QC"]["names"][sample[0]]
                self.wells[index].metadata["sample_type"] = sample_type
                self.wells[index].metadata["sample_name"] = f"{sample[0]}{counts[sample_code]}"
        
        # set metadata for wells that are for specimen samples
        for i, index in enumerate(self._well_inds_specimens):
            self.wells[index].metadata["QC"] = False
            self.wells[index].metadata["sample_code"] = self.specimen_code
            self.wells[index].metadata["sample_type"] = self.specimen_base_name
            self.wells[index].metadata["sample_name"] = f"{self.specimen_code}{i+1}"

        # # Collect all in one dict
        # self.QC = {
        #     "samples": QCnames,
        #     "freq": QCscheme,
        #     "unique_sequence": QC_unique_seq,
        #     "sequence_per_round": QC_seq_rounds,
        #     "frequency": N_specimens_between_QC,
        #     "N_per_round": N_QC_samples_per_round,   
        #     "N_unique_rounds": N_unique_QCrounds,
        # }
        
        # self.N_cons_specimens = N_specimens_between_QC
        # self.N_cons_QC =N_QC_samples_per_round

    # def create_layout_template(self):
    #     """_summary_

    #     """
        
    #     # check if config file is in right format for QC???
        
    #     # Check if QC scheme defined in config file
    #     QC = self.config.get('QC', False) 
        
    #     if QC:
    #         logger.debug("Setting up QC scheme from config file")
            
    #         N_specimens_between_QC = self.config['QC']['run_QC_after_n_specimens']
    #         QCscheme = self.config['QC']['scheme']
    #         QCnames = self.config['QC']['names']
            
    #         N_QC_samples_per_round = np.max(
    #             [QCscheme[key]['position_in_round'] for key in QCscheme.keys()]
    #             )
            
    #         N_unique_QCrounds = np.max(
    #             [QCscheme[key]['every_n_rounds'] for key in QCscheme.keys()]
    #         )
    #     # <- end if 

    #     # Generate QC sequence list in each unique QC round
    #     QC_unique_seq = []
    #     for m in range(0, N_unique_QCrounds):

    #         m_seq = []

    #         for key in QCscheme:
    #             order = QCscheme[key]['position_in_round']

    #             if QCscheme[key]["introduce_in_round"] == m + 1:
    #                 m_seq.append((key, order))

    #             if QCscheme[key]["introduce_in_round"] == 0:
    #                 m_seq.append((key, order))
    #         # <- end for loop

    #         m_seq.sort(key=lambda x: x[1])
    #         QC_unique_seq.append(m_seq)
            
    #     # <- end for loop
        
    #     logger.debug(f"{len(QC_unique_seq)} unique QC sequences defined: ")
    #     for i, qc_seq in enumerate(QC_unique_seq):
    #         logger.debug(f"\t{i+1}) {qc_seq}")
            
    #     self._QC_unique_seq = QC_unique_seq

    #     #Number of QC rounds per plate
    #     N_QC_rounds = self.capacity // (N_QC_samples_per_round + N_specimens_between_QC - 1)

    #     logger.debug(f"Assigning {N_QC_rounds} QC rounds per plate")

    #     # Generate QC sample sequence for each QC round
    #     qc_rounds = 0
    #     QC_seq_rounds = []
    #     while qc_rounds < N_QC_rounds:
    #         for seq in QC_unique_seq:
    #             qc_rounds += 1
    #             QC_seq_rounds.append(seq)
    #             logger.debug(f"\t Round {qc_rounds}: {seq}")

    #         qc_rounds = len(QC_seq_rounds)

    #     # Collect all in one dict
    #     self.QC = {
    #         "samples": QCnames,
    #         "freq": QCscheme,
    #         "unique_sequence": QC_unique_seq,
    #         "sequence_per_round": QC_seq_rounds,
    #         "frequency": N_specimens_between_QC,
    #         "N_per_round": N_QC_samples_per_round,   
    #         "N_unique_rounds": N_unique_QCrounds,
    #     }
        
    #     self.N_cons_specimens = N_specimens_between_QC
    #     self.N_cons_QC =N_QC_samples_per_round
        



class Nisse:

    def plot_layout(self, **kwargs):
        
        well_labels = self.sample_order   
        well_color_data = [val.split('_')[0] for val in well_labels ]
        
        self.plateplot(well_labels, well_color_data, **kwargs)
        
        
    def plot_batch(self, batch_index, 
                   well_label_column, well_color_column, 
                   label_dtype=None,
                   **kwargs):
        
        title_str = f"Batch {batch_index + 1}: {well_label_column} colored by {well_color_column}" 

        
    
        self.plateplot(self.batches_df[batch_index][well_label_column].astype(label_dtype),
                       self.batches_df[batch_index][well_color_column],
                       title_str = title_str,
                       **kwargs)
        
        
    def plateplot(self, well_label_data: list, well_color_data: list,
                fontsize: int = 8,
                rotation: int = 0,
                colormap: str = "tab20",
                NaN_color: tuple = (1,1,1),
                step = 10,
                title_str = '',
                alpha_val = 0.7,
                well_size = 1200
                ):

        # DEFINE COLORS TO USE
        levels = pd.unique(well_color_data)    
        print(f"Number of colors to use: {len(levels)}")
        
        RGB_colors = {}
        for i,s in enumerate(levels): 
            col = mpl.colormaps[colormap](i)[0:3] #get RGB values for i-th color in colormap
            if not pd.isnull(s):
                RGB_colors.setdefault(s, col) 
            else: 
                RGB_colors.setdefault("NaN", NaN_color)
            
        # Define list with RGB color for each well
        RGB_per_well = []    
        for dp in well_color_data:
            if not pd.isnull(dp):
                RGB_per_well.append(RGB_colors[dp])
            else:
                RGB_per_well.append(RGB_colors["NaN"])
            
        # DEFINE GRID FOR WELLS
        # 1 - define the lower and upper limits 
        minX, maxX, minY, maxY = 0, self._n_columns*step, 0, self._n_rows*step
        # 2 - create one-dimensional arrays for x and y
        x = np.arange(minX, maxX, step)
        y = np.arange(minY, maxY, step)
        # 3 - create a mesh based on these arrays
        X, Y = np.meshgrid(x, y)

        # PLOT
        # variables for scatter plot
        bubble_size = well_size
        grid_col = (1,1,1)
        edge_color = (0.5, 0.5, 0.5)

        # Style and size
        fig = plt.figure(dpi=100)
        plt.style.use('bmh')
        fig.set_size_inches(11.69, 8.27)
        ax = fig.add_subplot(111)

        # PLOT WELLS 
        well_count = 1
        N_wells_to_use = len(well_label_data)
        
        # for i in range(n_rows-1,-1,-1):
        #     for j in range(0,n_cols): 
        # use itertools to create compound iterator instead of above
        for i,j in itertools.product(range(self._n_rows-1, -1, -1), range(0, self._n_columns)):
            
            x_i = X[i,j]
            y_i = Y[i,j]
            
            if well_count > N_wells_to_use: 
                info_str = ""
                col = NaN_color
            else:     
                info_str = f"{well_label_data[well_count-1]}"                
                col = RGB_per_well[well_count-1]
            
            ax.annotate(info_str, (x_i, y_i), 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        rotation=rotation,
                        fontsize=fontsize)
        
            ax.scatter(x_i, y_i, 
                    bubble_size,color=col, 
                    alpha=alpha_val, 
                    edgecolors=edge_color)
            
            well_count += 1
        
        # end loop ------------------------
        
        
        # LEGENDS 
        # Create dummy plot to map legends, save plot handles to list
        lh = []
        for key,color in RGB_colors.items():
            lh.append(ax.scatter([],[],bubble_size*0.8, 
                                color=color, label=key, 
                                alpha=alpha_val, 
                                edgecolors=edge_color))
                    
        # Add a legend
        # Adjust position depending on number of legend keys to show
        pos = ax.get_position()
        if len(levels) < 6:
            ax.set_position([pos.x0, pos.y0*1.8, pos.width, pos.height*0.9])
            ax.legend(
                handles = lh,
                bbox_to_anchor=(0.15, -0.15, 0.7, 1.3),
                loc='lower center', 
                frameon = False,
                labelspacing=4,
                ncol=4
                )
        else:
            ax.set_position([pos.x0, pos.y0*2, pos.width, pos.height*0.8])
            ax.legend(
                handles = lh,
                bbox_to_anchor=(0.15, -0.25, 0.7, 1.3),
                loc='lower center', 
                frameon = False,
                labelspacing=1,
                ncol=8
                )

        # FIG PROPERTIES
        # X axis
        ax.set_xticks(x)
        ax.set_xticklabels(self.columns)
        ax.xaxis.grid(color=grid_col, linestyle='dashed', linewidth=1)
        ax.set_xlim(-1*x.max()*0.05,x.max()*1.05)

        # Y axis
        ax.set_yticks(y)
        ax.set_yticklabels(self.rows[::-1])
        ax.yaxis.grid(color=grid_col, linestyle='dashed', linewidth=1)
        ax.set_ylim(-1*y.max()*0.07,y.max()*1.07)

        # 
        ax.set_title(title_str)
        
        # Hide grid behind graph elements
        ax.set_axisbelow(True)
                
        return fig

    

class Study:
    
    name = str
    sample_layout = object
    QC_config_file = str
    batches = list
    
    
    def __init__(self, 
                 study_name=None, 
                 study_samples = None):
        
        self.name = study_name
    
    def __repr__(self):
        pass
    def __str__():
        pass
    
    def __contains__():
        pass
    
    def load_specimen_records():
        pass
    
    def distribute_to_plates():
        pass
    
    def add_samples_to_plate():
        pass
    
    def create_batch_layout_lists():
        pass
    
    def create_batch_layout_figures():
        pass
    
    
    def create_batches(self, specimen_df: object) -> None: 
            """Distributes specimen samples in <specimen_df> together with QC samples on plates according to the QC sample scheme.
            
            Created attributes. 
            Each batch (plate) is a pandas dataframe stored in a list in the object attribute 'batches_df', and ''

            Args:
                specimen_df (object): pandas dataframe where each row is a specimen sample
            """
            
            batch_count = 1
            batches_df_list = []
            
            # get specimen data from study list
            
            specimen_df_copy = specimen_df.copy()
            spec_count = 0
            
            while specimen_df_copy.shape[0] > 0:

                # extract max specimen samples that will fit on plate; select from top and remove them from original DF
                sel = specimen_df_copy.head(self.N_specimens)
                specimen_df_copy.drop(index=sel.index, inplace=True) 
                
                # reset index to so that rows always start with index 0
                sel.reset_index(inplace=True, drop=True)
                specimen_df_copy.reset_index(inplace=True, drop=True)

                # keep track on how many wells we should use per batch
                N_specimens_left = len(sel)

                # populate batch dict 
                batch_temp = {
                    "well_name": [],
                    "well_coords": [],
                    "sample_name": [],
                    "batch_number": [],
                }
                
                # add keys from dataframe columns
                columns = specimen_df.columns
                for col in columns:
                    batch_temp.setdefault(col, [])
                
                
                batch = batch_temp.copy()
                
                # For some reason we have to reset the batch dict like this
                for key in batch.keys():
                    batch[key]=[]

                for i, sample in enumerate(self.sample_order):
                    
                    #print(f"BATCH {batch_count}, SAMPLE {sample}")
                    sample_type, sample_number = sample.split("_")
                    
                    if sample_type == "S" and (int(sample_number) > N_specimens_left):
                        logger.debug(f"Finished distributing specimen samples to plate wells. Last specimen is {self.sample_order[i-1]}")
                        break
                                    
                    batch['well_name'].append( str().join(self.well_names[i]) )
                    batch['well_coords'].append(self.well_coordinates[i])
                    batch['sample_name'].append(self.sample_order[i])
                    batch['batch_number'].append(batch_count)
                    
                    index = int(sample_number)-1
                    
                    if sample_type == "S":
                        for col in columns:
                            batch[col].append( sel[col][index])
                            
                        spec_count += 1
                    else:
                        for col in columns: 
                            batch[col].append(np.nan)
                        
                # --- END OF FOOR LOOP ---

                batch_df = pd.DataFrame(batch)
                
                batches_df_list.append(batch_df)
                
                batch_count += 1

            # --- END OF WHILE LOOP ---
            
            self.batches_df = batches_df_list
            self.N_batches = len(batches_df_list)
            
            # concatenate all DFs to one long DF
            self.all_batches_df = pd.concat(batches_df_list).reset_index()
            
            logger.info(f"Finished distributing samples onto plates; {self.N_batches} batches created.")

    def to_file(self, 
                fileformat: str = "csv",
                batch_index: list = None,
                folder_path: str = None,
                write_columns: list = None):
        
        if batch_index is None:
            batch_index = range(0, len(self.batches_df))
            
        if folder_path is None:
            folder_path = os.getcwd()
        
        for id in batch_index:
            filename = f"batch_{id+1}."
            filepath = os.path.join(folder_path, filename)
            
            batch = self.batches_df[id]
            
            if write_columns is not None:
                dropcolumns = [col for col in batch.columns if col not in write_columns]
                batch = batch.drop(columns=dropcolumns)
                logger.debug("Dropping columns from dataframe:")
                for col in dropcolumns:
                    logger.debug(f"\t{col}")
            
            if fileformat == "tsv" or ("tab" in fileformat): 
                fext = "tsv"
                batch.to_csv(filepath+fext, sep="\t")
                
            elif fileformat == "csv" or ("comma" in fileformat):
                fext = "csv"
                batch.to_csv(filepath+fext)
                
            else:
                fext = "xlxs"
                batch.to_excel(filepath+fext)
                
            logger.info(f"Saving batch {id} to {filepath+fext} ")    



