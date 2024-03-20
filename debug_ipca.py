import factor_models as fm
import wrds_function as wrds
import numpy as np
import pandas as pd


listFactors = [5]
sizeCovarianceWindow = 252
sizeWindow = [60]
initialOOSYear = 2000
capProportion = [0.001]

# wrds.get_daily_crsp_data(start_date="1969-12-31")

# wrds.process_compustat(save=True)

fm.run_factor_models()
