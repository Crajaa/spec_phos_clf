import logging
from os.path import dirname, join, realpath
from os import mkdir
import time
import pickle

from data_manager import DataManager
from model_manager import ModelManager
from utils.conf import Params
from utils.loggers import setup_loggers
from utils.report_run import write_report_from_run

PARAMS_PATH = join(dirname(realpath(__file__)), "..", "conf", "conf.yaml")
logger_name = Params(PARAMS_PATH, 'Logger').get_dict_params()['name']

augment_data = Params(PARAMS_PATH, 'Data_Settings').get_dict_params()['augment_data']

PATHS = Params(PARAMS_PATH, 'Paths').get_dict_params()
path_to_raw = PATHS['path_to_raw']
model_input_path = PATHS['path_to_model_input']
model_name = PATHS['model_name']
results_path = PATHS['path_to_results']

generate_results_report = Params(PARAMS_PATH, 'Choice').get_dict_params()['generate_results_report']


def init():
    setup_loggers(logging.INFO)
    logger = logging.getLogger(logger_name)
    try :
        logger.info("="*20 + " Begin main program " + "="*20)
        logger.info("Checking no path from config file is empty...")
        assert(path_to_raw and model_input_path and model_name and results_path)
        return logger
    except AssertionError as e:
        logger.error(
            "At least one path from config file is empty.\
            Program will be interrupted here.Details : " + str(e)
            )
        raise e

def run():
    """
    """
    logger = init()
    
    try:
        data = DataManager(path_to_raw)
        model = ModelManager(data, augment_data)

        logger.info(f"Creating new folder in {results_path} to save model object and results...")
        folder_name = time.strftime("%Y-%m-%d_%H-%M-%S_RUN")
        mkdir(results_path+folder_name)
        logger.info(f"{folder_name} created in {results_path}.")

        logger.info(f"Saving model object in the created folder...")
        with open(results_path + folder_name + "/" + model_name, "wb") as output:
            pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
        logger.info("Saving of model object completed.")

        if generate_results_report :
            report_file = open(results_path + folder_name + "/" + "Run_Report.txt", "w")
            report_file.write(write_report_from_run(model, augment_data))
            report_file.close()
        
        logger.info("Main program completed successfully.")
        return model

    except Exception as e:
        logger.error("An error occured, program will be interrupted. Details : " + str(e))
        raise e 

if __name__ == "__main__":
    run()

