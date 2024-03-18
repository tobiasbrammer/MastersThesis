import factor_models
import factor_models as fm
import wrds_function as wrds

listFactors = [5]
sizeCovarianceWindow = 252
sizeWindow = [60]
initialOOSYear = 2000
capProportion = [0.001]

# wrds.process_compustat(save=True)


# factor_models.run_factor_models()

fm.run_ipca(listFactors=listFactors, sizeWindow=sizeWindow, capProportion=capProportion)
