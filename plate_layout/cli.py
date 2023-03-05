
# from .plate_layout import Study
# from .plate_layout import Plate
# from .plate_layout import QCPlate
# from .pl_logger import logger
from plate_layout import Study
from plate_layout import Plate
from plate_layout import QCPlate
from pl_logger import logger

import logging, argparse

def setup_option_parser(parser):
    
        parser.add_argument("study-file",
                        help="<path>: file path to csv/excel file with study specimen samples: <path/samples.[csv, xlsx, xls]>")
        
        parser.add_argument("-r", "--randomize",
                        default="no",
                        choices=['yes', 'no'],
                        help="randomize specimen order (if sample groups defined in study file, the order of _the groups_ will be randomized.")
        
        parser.add_argument("-p", "--plate-size",
                            default='96',
                        help="<plate size> (=#wells), e.g. '6', '12', '24', '48', 96,... . The number of wells will be changed if it does not fit in a 2:3 format")
        
        parser.add_argument("-q", "--qc-file",
                        help="<path> to a 'toml' file with QC scheme parameters:  <path/qc_config.toml>")
        
        parser.add_argument("-n", "--study-name",
                            default = "Study<date(today)>",
                        help="<name of study> to include in output files.")
        
        parser.add_argument("-o", "--output-folder",
                            default="plate_layouts", 
                        help="<folder path> for saving layout files and figures. ")
        
        parser.add_argument("-l", "--log-level", 
                            default='debug',
                        help="<'info'/'debug'> level of information printed to console")
        
        parser.add_argument("-e", "--export", help="<''>")
        
        parser.add_argument("-m", "--metadata", help="Column names (metadata) from the study input file that will be printed to lists. The last two will be used when rendering figures, where the second last annotates the plate by color and the last annotates by value.", nargs='+')
        


        return parser.parse_args()
    

def main():

    parser = argparse.ArgumentParser(description="Create sample layout lists/figures for multiwell plates from a csv/excel with study sample record",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    op = setup_option_parser(parser)
    if op.log_level == 'debug':
        logger.setLevel(logging.DEBUG)
    else:
       logger.setLevel(logging.INFO) 
    
    
    logger.debug(f"Running plate_layout in CLI mode") 
   
    print(f"args: {op.randomize}")
    
    
if __name__ == "__main__":
    main()