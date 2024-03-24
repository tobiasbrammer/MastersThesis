import factor_models as fm
import numpy as np
import pandas as pd

fm.run_factor_models()


res = "factor_data/residuals/pca/OOSResiduals_PCA_factor1_rollingwindow_60.npz"

data = np.load(res)
