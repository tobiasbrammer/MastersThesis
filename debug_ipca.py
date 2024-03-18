import factor_models
import factor_models as fm
import wrds_function as wrds

# wrds.process_compustat(save=True)


factor_models.run_FF(sizeWindow=[60], intitialOOSYear=1998, capProportion=[0.001])
