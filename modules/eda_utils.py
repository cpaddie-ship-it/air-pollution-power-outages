from scipy import stats
import numpy as np

def partial_r_year(df, x_col, y_col):
    """
    Pearson r between x_col and y_col after partialing out 'year'
    from both variables via OLS residuals.
    
    Parameters
    df     : DataFrame containing 'year', x_col, y_col
    x_col  : predictor column name
    y_col  : outcome column name
    
    Returns
    r : float — partial Pearson correlation
    p : float — two-tailed p-value
    """
    sub = df[['year', x_col, y_col]].dropna()
    def resid(col):
        X = np.column_stack([np.ones(len(sub)), sub['year']])
        b = np.linalg.lstsq(X, sub[col].values, rcond=None)[0]
        return sub[col].values - X @ b
    return stats.pearsonr(resid(x_col), resid(y_col))
