##This program allows for phosphate classification according to the methodology explained in our article "Sedimentary phosphate classification based on spectral analysis and machine learning"

To run the main program :

1) first, go to conf/conf.yaml, and fill the required parameters and paths. No need to hardcode anything in source code files (.py) files. 
2) then, go to src/, and run the launcher.py program : in a terminal, type : python launcher.py
    Running launcher.py generates logs, that are visible on terminal, and also written in logs/run.log
    At the end of the run, a results folder with timestamp is created in results/. In it, two files : 
        1- model_object.model : serialized model object
        2- run_report.txt : text file containing summarized results
3) If you wish to check your model's attributes, you can use python's pickle module: pickle.load('your model')

/!\: The constructed data set is not available as the authors don't have the funder permission to share data.