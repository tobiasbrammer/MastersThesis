import factor_models as fm
import wrds_function

wrds_function.process_compustat(save=True)

fm.run_ipca()
