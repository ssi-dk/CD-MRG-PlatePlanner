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
    plate_id : int
    metadata : dict
    
    def __init__(self,
                 name, 
                 coordinate, 
                 index = None,
                 plate_id = None,
                 metadata = None) -> None:
        
        self.name = name
        self.coordinate = coordinate
        self.index = index
        self.plate_id = plate_id
        self.metadata = metadata
        
    def __str__(self) -> str:
        pass
    
    def __repr__(self) -> str:
        pass

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
    
    rows: list = []
    columns: list = []
    wells: list = [] # list of well objects
    index_coordinates: list = [] # list of tuples with a pair of ints (row, column)
    alphanumerical_coordinates: list = [] # list of strings for canonical naming of plate coordnates, i.e A1, A2, A3, ..., B1, B2, etc
    
    well_coordinates: list = []
    well_names: list = []
    
    plate_well_names: object = np.empty(0) # -> numpy array 
    plate_well_coordinates: object = np.empty(0) # -> numpy array 
    
    
    ##def __init__(self, annotation_data: list,color_data: list,rows=list('ABCDEFGH'),columns=list(range(0, 12))):
    def __init__(self,  *args, plate_id = 1, **kwargs) -> None:    
        """_summary_
        
        Constructs all the necessary attributes for the Plate object, given the specifications in the plate.toml file
        
        """
        self.plate_id = plate_id
        
        if args:
            rows = args[0][0]
            columns = args[0][1]
        
        if kwargs:
            rows = kwargs['rows']
            columns = kwargs['columns']
            
        self.rows = list(range(0,rows))
        self.columns = list(range(0,columns))
        self.capacity = len(self.rows) * len(self.columns)
        
        self.create_index_coordinates()
        self.create_alphanumerical_coordinates()
        self.define_wells()
        
        # # create well names from plate row and column labels
        # self.well_names = list(itertools.product(
        #                         [str(val) for val in self.rows],
        #                         [str(val) for val in self.columns]
        #                         )
        #                        )
        
    def create_index_coordinates(self):
        # count from left to right, starting at well in top left
        self.index_coordinates = list(itertools.product(
                                        range(len(self.rows)-1, -1, -1),
                                        range(0, len(self.columns))
                                        )
                                      )
        
    def create_alphanumerical_coordinates(self):
        alphabet = list(string.ascii_uppercase)
        
        row_crds = alphabet
        number_of_repeats = 1
        
        while len(self.rows) > len(row_crds):
            row_crds = list(itertools.product(alphabet, repeat=number_of_repeats))
            number_of_repeats += 1
            
        alphanumerical_coordinates = []
        for r_i, row in enumerate(self.rows):
            for c_i, column in enumerate(self.columns):
                #an_crd = row_crds[r_i], column
                r_str = str()
                for r in row_crds[r_i]:
                    r_str = r_str.join(r)
                
                alphanumerical_coordinates.append(f"{r_str}_{c_i+1}")
        
        self.alphanumerical_coordinates = alphanumerical_coordinates
        
    def define_wells(self):
        
        for i, index_crd, crd in enumerate(zip(self.index_coordinates, self.alphanumerical_coordinates)):
            self.wells.append(
                Well(coordinate=index_crd,
                     name=crd,
                     index=i, 
                     plate_id=self.plate_id)
            )
            
          
    def __len__(self):
        # TODO
        pass
    
    def __contains__(self):
        # TODO
        pass
    
    def __getitem__(self, coordinate):
        # TODO
        pass
        
    def __setitem__(self, coordinate, well):
        # TODO
        pass
    
    def __delitem__(self):
        # TODO
        pass
        
    def create_layout(self):
        self.build_coordinate_system()
        self.create_layout_template()
    
    def load_config_file(self, config_file: str = None):
        
        # READ CONFIG FILE
        if config_file is None: 
            
            logger.warning("No config file specified. Trying to find a toml file in current folder.")
            
            config_file_search = glob.glob("*.toml")      
            
            if config_file_search:
                config_file = config_file_search[0]
                logger.info(f"Using toml file '{config_file}'")
            # <- end if
        # <- end if     
        
        try:
            with open(config_file, mode="rb") as fp:
                
                self.config = tomli.load(fp)
            
            logger.info(f"Successfully loaded config file {config_file}")
            logger.debug(f"{self.config}")
            
        except FileNotFoundError:
            logger.error(f"Could not find/open config file {config_file}")

        # <- end try
    
    @staticmethod
    def create_labels(rowcol_specification, flip_order:bool=False):
         
        flipper = -1 if flip_order else  1
        
        if isinstance(rowcol_specification, str):
            label =  list(rowcol_specification)[::flipper]
        elif isinstance(rowcol_specification, int):
            label = list(range(0, rowcol_specification))[::flipper]
        elif isinstance(rowcol_specification, list): 
            label =  rowcol_specification
        else:
            label = None
            
        return label
    

    def build_coordinate_system(self):
        
        # SET WELL LABELS AND DIMENSIONS
        
        self.rows = Plate.create_labels(self.config['plate']['rows'], flip_order=False)
        
        if self.rows is None: 
            logger.error("Unknown format for plate row labels in config file")
            raise RuntimeError
        
        self.columns = Plate.create_labels(self.config['plate']['columns'])
        
        if self.columns is None: 
            logger.error("Unknown format for plate column labels in config file")
            raise RuntimeError
        
        
        # Create coordinate system - use zero-indexing to conform with Python indexing 
        
        self._n_rows = len(self.rows)
        self._n_columns = len(self.columns)
        self._n_wells = self._n_rows * self._n_columns
        
        # count from left to right, starting at well in top left
        self.well_coordinates = list(itertools.product(range(self._n_rows-1, -1, -1),
                                                       range(0, self._n_columns)))
        # create well names from plate row and column labels
        self.well_names = list(itertools.product(
            [str(val) for val in self.rows],
            [str(val) for val in self.columns]
        ))
        
        # store well names for plates in a matrix representation 
        self.plate_well_names = self.populate_plate(
            [lval+rval for lval,rval in self.well_names]
            )
        
        # store well coordinates for plates in a matrix representation 
        self.plate_well_coordinates = self.populate_plate(self.well_coordinates)
        
        logger.info(f"Created a plate with {self._n_wells} wells: \n {self.plate_well_names}")
        

    def create_layout_template(self):
        """_summary_

        """
        
        # check if config file is in right format for QC???
        
        # Check if QC scheme defined in config file
        QC = self.config.get('QC', False) 
        
        if QC:
            logger.info("Setting up QC scheme from config file")
            
            N_specimens_between_QC = self.config['QC']['run_QC_after_n_specimens']
            QCscheme = self.config['QC']['scheme']
            QCnames = self.config['QC']['names']
            
            N_QC_samples_per_round = np.max(
                [QCscheme[key]['position_in_round'] for key in QCscheme.keys()]
                )
            
            N_unique_QCrounds = np.max(
                [QCscheme[key]['every_n_rounds'] for key in QCscheme.keys()]
            )
        # <- end if 

        # Generate QC sequence list in each unique QC round
        QC_unique_seq = []
        for m in range(0, N_unique_QCrounds):

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

        #Number of QC rounds per plate
        N_QC_rounds = self._n_wells // (N_QC_samples_per_round + N_specimens_between_QC - 1)

        logger.debug(f"Assigning {N_QC_rounds} QC rounds per plate")

        # Generate QC sample sequence for each QC round
        qc_rounds = 0
        QC_seq_rounds = []
        while qc_rounds < N_QC_rounds:
            for seq in QC_unique_seq:
                qc_rounds += 1
                QC_seq_rounds.append(seq)
                logger.debug(f"\t Round {qc_rounds}: {seq}")

            qc_rounds = len(QC_seq_rounds)

        # Collect all in one dict
        self.QC = {
            "samples": QCnames,
            "freq": QCscheme,
            "unique_sequence": QC_unique_seq,
            "sequence_per_round": QC_seq_rounds,
            "frequency": N_specimens_between_QC,
            "N_per_round": N_QC_samples_per_round,   
            "N_unique_rounds": N_unique_QCrounds,
        }
        
        self.N_cons_specimens = N_specimens_between_QC
        self.N_cons_QC =N_QC_samples_per_round
        
        self.create_sample_order()
        
        logger.info(f"\n\t{self.plate_sample_order}")


    def create_sample_order(self):

        # GENERATE TEMPLATE FOR PLATE LAYOUT IN TERMS OF QC AND SPECIMEN SAMPLE ORDER

        # Check if QC scheme defined in config file
        QC = self.config.get('QC', False)
        if QC: 
            start_with_QC = self.config['QC']['start_with_QC_round']
        else:
            start_with_QC = False
            
        sample_order = []

        # Keep track of "run state" for entering or leaving QC/specimen run count
        state = {
            "run_QC":  start_with_QC,  # start with QC round?
            "QC_round": 0,
            "QC_count": 0,
            "total_QC_count": 0,
            "specimen_count": 0,
            "total_specimen_count": 0,
            "N_wells_assigned": 0,
            "specimen_rounds": 0,
        }

        # Set up counter for each QC sample
        QC_count = {key: 0 for key in self.QC['samples']}

        logger.info(f"Distributing specimen and QC samples on plate ")
        # Loop row-wise over plate wells and assign sample order
        for row, col in self.well_coordinates: 
        
            if state['run_QC']:  # RUN QC ROUND
  
                state['total_QC_count'] += 1
                
                logger.debug(f"\t Assigning QC sample {state['total_QC_count'] }")
                
                # Get QC sample in the sequence for current QC round
                QC_sample = self.QC['sequence_per_round'][state['QC_round']
                                                    ][state['QC_count']][0]
                QC_count[QC_sample] += 1

                sample_order.append(f"{QC_sample}_{QC_count[QC_sample]}")

                # increment QC count while in QC round
                state['QC_count'] += 1

                if state['QC_count'] >= self.QC['N_per_round']:  # end QC round
                    logger.debug(f"Finished with QC round {state['QC_round']}")
                    state['run_QC'] = False
                    state['QC_count'] = 0
                    state['QC_round'] += 1

            else:  # RUN SPECIMEN ROUND
                
                # increment speciment count wile in specimen round
                state['specimen_count'] += 1
                state['total_specimen_count'] += 1

                logger.debug(f"Assigning specimen sample S{state['total_specimen_count']}")
                
                # end speciment round and start QC round
                if QC and (state['specimen_count'] >= self.QC['frequency']):
                    state['run_QC'] = True
                    state['specimen_count'] = 0
                    state['specimen_rounds']

                sample_order.append(f"S_{state['total_specimen_count']}")

            # Update assigment counter
            state['N_wells_assigned'] += 1
        # END LOOP

        self.N_specimens = state['total_specimen_count']
        self.N_QC_samples = state['total_QC_count']
        self.N_QC_rounds = state['QC_round']
        self.N_specimen_rounds = state['specimen_rounds']
        
        self.sample_order = sample_order
        self.plate_sample_order = self.populate_plate(sample_order)   
        
        
    def populate_plate(self, data: list) -> object:
         
        plate_array = np.empty((self._n_rows, self._n_columns), 
                               dtype=object)

        for i, well in enumerate(self.well_coordinates):
            r, c = well[0], well[1]
            plate_array[r][c] = data[i]

        return np.flipud(plate_array)

        
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
    
    def __str__(self) -> str:
        return f"{self.plate_sample_order}"
    
