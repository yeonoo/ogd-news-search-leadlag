"""
var_granger.py - Core analysis functions for the lead-lag analysis of news
coverage and open government data (OGD) portal search in South Korea.

Extracted from the research notebook; executable logic is unchanged. Docstrings
and comments were translated to English, and code-generated result-column names
were renamed to English. Raw-data column names (e.g. Korean BigKinds/search
columns) and Korean keyword values are preserved to match the source data.

Note: some functions write CSVs to a global OUTPUT_DIR; define it before calling
(e.g. OUTPUT_DIR = "output/") or set var_granger.OUTPUT_DIR.

Environment: Python 3.10, statsmodels 0.14.6, pandas, numpy, scipy.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR

OUTPUT_DIR = "output/"   # default; override as needed



# ======================================================================
# News intensity weighting  (source cell 17)
# ======================================================================
def calculate_news_intensity(df, keyword_col='특성추출(가중치순 상위 50개)',
                              date_col='일자'):
    """
    Compute news-coverage intensity from purely reverse linear rank weighting.
    
    Formula
    -------
    w_{i,k} = (N_i - k + 1) / sum_{j=1}^{N_i}(N_i - j + 1)
            = 2(N_i - k + 1) / (N_i * (N_i + 1))
    s.t. sum_{k=1}^{N_i} w_{i,k} = 1
    
    Parameters
    ----------
    df : pd.DataFrame (raw news)
    keyword_col : str, name of the comma-separated keyword-string column
    date_col : str, name of the date column
    
    Returns
    -------
    pd.DataFrame : monthly (YYYY-MM) x keyword pivot table
    
    Raises
    ------
    KeyError : missing column
    AssertionError : weight sum != 1 or formula self-test failure
    """
    if keyword_col not in df.columns:
        raise KeyError(f"'{keyword_col}' not in df.columns")
    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' not in df.columns")
    
    def calculate_normalized_weights(keyword_list):
        n_i = len(keyword_list)
        if n_i == 0:
            return []
        raw_scores = [n_i - k + 1 for k in range(1, n_i + 1)]
        total_score = sum(raw_scores)  # = n_i * (n_i + 1) / 2
        return [score / total_score for score in raw_scores]
    
    # Self-test: formula validation
    for n in [1, 5, 22, 50]:
        w = calculate_normalized_weights(['x'] * n)
        assert abs(sum(w) - 1.0) < 1e-10, f"weight sum ≠ 1 at N_i={n}"
        if n >= 2:
            assert w[0] > w[-1], f"reverse-linear violation at N_i={n}"
    
    df_valid = df.dropna(subset=[keyword_col]).copy()
    df_valid['keyword_list'] = df_valid[keyword_col].apply(
        lambda x: [kw.strip() for kw in str(x).split(',') if kw.strip()]
    )
    df_valid['normalized_weight'] = df_valid['keyword_list'].apply(calculate_normalized_weights)
    
    df_exploded = df_valid.explode(['keyword_list', 'normalized_weight'])
    df_exploded.rename(columns={'keyword_list': 'Keyword',
                                 'normalized_weight': 'Weight'}, inplace=True)
    df_exploded['Weight'] = df_exploded['Weight'].astype(float)
    
    # year-month conversion
    df_exploded['year_month'] = pd.to_datetime(df_exploded[date_col])\
                            .dt.to_period('M').astype(str)
    
    df_monthly = df_exploded.groupby(['year_month', 'Keyword'])['Weight']\
                             .sum().unstack(fill_value=0)
    
    return df_monthly



# ======================================================================
# Missing-data imputation (1/sqrt2 substitution)  (source cell 22)
# ======================================================================
def apply_lod_imputation(df_raw, ratio=1/np.sqrt(2)):
    """
    Apply LOD x ratio imputation to left-censored data.
    
    For each month, the smallest non-zero value is defined as that month's
    limit of detection (LOD), and zeros in that month are imputed as LOD x ratio.
    
    Parameters
    ----------
    df_raw : pd.DataFrame (rows=months, cols=keywords)
    ratio  : float, default 1/sqrt(2) ~ 0.707 (Helsel 2012)
    
    Returns
    -------
    pd.DataFrame : imputed pivot
    
    Raises
    ------
    ValueError : if df_raw is empty or all values are zero
    """
    if df_raw.empty:
        raise ValueError("df_raw is empty")
    if (df_raw.values == 0).all():
        raise ValueError("df_raw has no observed (nonzero) values")
    
    monthly_lod = df_raw.replace(0, np.nan).min(axis=1)
    df_imputed = df_raw.apply(
        lambda col: np.where(col == 0, monthly_lod * ratio, col)
    )
    return df_imputed



# ======================================================================
# ADF stationarity test  (source cell 27)
# ======================================================================
def adf_test(series, sig=0.05):
    """
    ADF test. If the raw series is non-stationary, re-test after first differencing.
    
    Parameters
    ----------
    series : pd.Series (log-transformed time series)
    sig    : float, significance level
    
    Returns
    -------
    tuple (pd.Series or None, bool or None, str)
        - stationary series (None on failure)
        - differenced flag (True/False, None on failure)
        - status flag: 'level', 'diff', 'nonstationary', 'zero_var', 'zero_var_diff'
    """
    if series.std() == 0:
        return None, None, 'zero_var'
    
    # raw-series test
    p_level = adfuller(series.dropna(), autolag='AIC')[1]
    if p_level < sig:
        return series.dropna(), False, 'level'
    
    # first-difference test
    diff = series.diff().dropna()
    if diff.std() == 0:
        return None, None, 'zero_var_diff'
    
    p_diff = adfuller(diff, autolag='AIC')[1]
    if p_diff < sig:
        return diff, True, 'diff'
    
    return None, None, 'nonstationary'



# ======================================================================
# Descriptive statistics  (source cell 36)
# ======================================================================
def describe_distribution(arr, name='var', log_transform=None):
    """
    Compute descriptive statistics for a 1-D array.
    
    Parameters
    ----------
    arr : np.ndarray (1D)
    name : str, variable name
    log_transform : {'ln', 'log1p', None}
        - 'ln': np.log(arr) (requires arr > 0)
        - 'log1p': np.log1p(arr) (arr >= 0)
        - None: raw values
    
    Returns
    -------
    dict with keys: N, mean, median, sd, min, max
    """
    arr = np.asarray(arr)
    
    if log_transform == 'ln':
        if (arr <= 0).any():
            raise ValueError(f"{name}: non-positive values for ln transform")
        vals = np.log(arr)
    elif log_transform == 'log1p':
        if (arr < 0).any():
            raise ValueError(f"{name}: negative values for log1p transform")
        vals = np.log1p(arr)
    else:
        vals = arr
    
    return {
        'variable': name,
        'N': len(vals),
        'mean': round(float(vals.mean()), 4),
        'median': round(float(np.median(vals)), 4),
        'std': round(float(vals.std(ddof=1)), 4),
        'min': round(float(vals.min()), 4),
        'max': round(float(vals.max()), 4),
    }



# ======================================================================
# VAR input preparation (per keyword)  (source cell 42)
# ======================================================================
def prepare_keyword_data(kw, df_search, df_news, adf_results):
    """
    Per keyword: apply log transform and ADF-based differencing, then
    reconstruct as a [News, Search] vector.
    
    Parameters
    ----------
    kw : str, keyword name
    df_search : pd.DataFrame (rows=months, cols=keywords, values=LOD-imputed search volume)
    df_news   : pd.DataFrame (rows=months, cols=keywords, values=news-coverage intensity)
    adf_results : pd.DataFrame (from Step 4; includes per-keyword differencing flag)
    
    Returns
    -------
    tuple (pd.DataFrame, dict)
        - df_var: [News, Search] 2 columns, stationary series
        - meta: differencing flag and number of observations
    
    Notes
    -----
    Column order fixed to the default Cholesky ordering [N, S].
    Because differencing may yield different lengths, the intersection index is used.
    """
    # log transform
    s_log = np.log(df_search[kw])        # search volume: ln (no zeros due to LOD imputation)
    n_log = np.log1p(df_news[kw])         # news: ln(x+1)
    
    # check differencing flag from ADF results
    row = adf_results[adf_results['keyword'] == kw].iloc[0]   # adf_results columns (built in the notebook)
    s_diffed = row['search_diff']
    n_diffed = row['news_diff']
    
    # apply differencing
    s_final = s_log.diff().dropna() if s_diffed else s_log
    n_final = n_log.diff().dropna() if n_diffed else n_log
    
    # index intersection
    common_idx = s_final.index.intersection(n_final.index)
    
    # arrange as [News, Search] (default Cholesky ordering)
    df_var = pd.DataFrame({
        'News': n_final.loc[common_idx].values,
        'Search': s_final.loc[common_idx].values
    }, index=common_idx)
    
    meta = {
        'keyword': kw,
        'search_diff': s_diffed,
        'news_diff': n_diffed,
        'n_obs': len(df_var)
    }
    
    return df_var, meta



# ======================================================================
# VAR estimation + AIC lag selection + Granger causality (per keyword)  (source cell 43)
# ======================================================================
def analyze_keyword(kw, df_search, df_news, adf_results, maxlag=4, siglevel=0.05):
    """
    Estimate a VAR and run bidirectional Granger tests for a single keyword.
    
    Parameters
    ----------
    kw : str, keyword name
    df_search, df_news : pd.DataFrame
    adf_results : pd.DataFrame
    maxlag : int, maximum lag (default 4)
    siglevel : float, significance level (default 0.05)
    
    Returns
    -------
    dict with keys:
        keyword, optimal_lag, stability, n_obs, search_diff, news_diff,
        N2S_F, N2S_p, N2S_sig, S2N_F, S2N_p, S2N_sig, type, error
    
    Notes
    -----
    grangercausalitytests(df[[X, Y]]) tests whether Y Granger-causes X.
    Therefore:
      - N->S test: df[['Search', 'News']] -> News causes Search
      - S->N test: df[['News', 'Search']] -> Search causes News
    """
    result = {
        'keyword': kw, 'optimal_lag': None, 'stability': None, 'n_obs': None,
        'search_diff': None, 'news_diff': None,
        'N2S_F': None, 'N2S_p': None, 'N2S_sig': None,
        'S2N_F': None, 'S2N_p': None, 'S2N_sig': None,
        'type': None, 'error': None
    }
    
    try:
        # prepare data
        df_var, meta = prepare_keyword_data(kw, df_search, df_news, adf_results)
        result['n_obs'] = meta['n_obs']
        result['search_diff'] = meta['search_diff']
        result['news_diff'] = meta['news_diff']
        
        # minimum observation check
        if len(df_var) < 10:
            result['error'] = f'insufficient observations ({len(df_var)})'
            return result
        
        # variance check
        if df_var['News'].std() == 0 or df_var['Search'].std() == 0:
            result['error'] = 'zero variance'
            return result
        
        # lag selection (adjust maxlag to n_obs)
        effective_maxlag = min(maxlag, len(df_var) // 5 - 1)
        if effective_maxlag < 1:
            effective_maxlag = 1
        
        model = VAR(df_var)
        
        try:
            lag_order = model.select_order(maxlags=effective_maxlag)
            optimal_lag = lag_order.aic
            if optimal_lag == 0:
                optimal_lag = 1  # force minimum lag of 1
        except Exception:
            optimal_lag = 1
        
        result['optimal_lag'] = optimal_lag
        
        # VAR estimation
        var_result = model.fit(optimal_lag)
        
        # stability check
        is_stable = var_result.is_stable()
        result['stability'] = is_stable
        if not is_stable:
            result['error'] = 'unstable model'
            return result
        
        # ----- bidirectional Granger tests -----
        # N->S: input [Search, News] tests News->Search
        gc_n2s = grangercausalitytests(
            df_var[['Search', 'News']].values,
            maxlag=optimal_lag, verbose=False
        )
        f_test = gc_n2s[optimal_lag][0]['ssr_ftest']
        result['N2S_F'] = round(f_test[0], 4)
        result['N2S_p'] = round(f_test[1], 4)
        result['N2S_sig'] = f_test[1] < siglevel
        
        # S->N: input [News, Search] tests Search->News
        gc_s2n = grangercausalitytests(
            df_var[['News', 'Search']].values,
            maxlag=optimal_lag, verbose=False
        )
        f_test = gc_s2n[optimal_lag][0]['ssr_ftest']
        result['S2N_F'] = round(f_test[0], 4)
        result['S2N_p'] = round(f_test[1], 4)
        result['S2N_sig'] = f_test[1] < siglevel
        
        # type classification
        n2s = result['N2S_sig']
        s2n = result['S2N_sig']
        if n2s and s2n:
            result['type'] = 'bidirectional'
        elif n2s and not s2n:
            result['type'] = 'N->S'
        elif not n2s and s2n:
            result['type'] = 'S->N'
        else:
            result['type'] = 'independent'
    
    except Exception as e:
        result['error'] = f'overall error: {str(e)[:80]}'
    
    return result



# ======================================================================
# Batch VAR/Granger over a keyword list  (source cell 44)
# ======================================================================
def run_var_granger_batch(keywords, df_search, df_news, adf_results, 
                           label='main', maxlag=4, siglevel=0.05):
    """
    Run VAR + Granger analysis over a list of keywords.
    
    Parameters
    ----------
    keywords : list, keywords to analyze
    df_search, df_news : pd.DataFrame
    adf_results : pd.DataFrame
    label : str, 'main' or 'sensitivity' (reflected in the output filename)
    maxlag : int, maximum lag
    siglevel : float, significance level
    
    Returns
    -------
    pd.DataFrame : full analysis results
    """
    print(f"\n{'='*60}")
    print(f"Starting VAR + Granger analysis [{label}] ({len(keywords)} keywords)")
    print(f"{'='*60}")
    
    all_results = []
    for i, kw in enumerate(keywords):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  progress: {i+1}/{len(keywords)} ({kw})")
        
        res = analyze_keyword(kw, df_search, df_news, adf_results,
                              maxlag=maxlag, siglevel=siglevel)
        all_results.append(res)
    
    df_all = pd.DataFrame(all_results)
    
    # save
    save_path = f'{OUTPUT_DIR}/var_granger_results_{label}.csv'
    df_all.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\nsaved: {save_path}")
    
    return df_all



# ======================================================================
# IRF and FEVD (per keyword, both Cholesky orderings)  (source cell 52)
# ======================================================================
def compute_irf_fevd(kw, df_search, df_news, adf_results,
                     optimal_lag, horizon=10):
    """
    Compute IRF and FEVD under both Cholesky orderings for a single keyword.
    
    Parameters
    ----------
    kw : str
    df_search, df_news : pd.DataFrame
    adf_results : pd.DataFrame
    optimal_lag : int, lag selected in Step 6
    horizon : int, IRF/FEVD horizon (default 10)
    
    Returns
    -------
    tuple (list, list)
        - irf_records: per-keyword IRF numeric records
        - fevd_records: per-keyword FEVD numeric records
    
    Notes
    -----
    Cholesky ordering notation:
      'N_first': df_var=[News, Search] -> identification where news leads
      'S_first': df_var=[Search, News] -> identification where search leads
    
    irfs[h][i, j]: response of variable i at step h to a shock in variable j
    decomp[i][h, j]: contribution of variable j's shock to variable i's forecast-error variance at step h+1
    """
    irf_records = []
    fevd_records = []
    
    # prepare_keyword_data returns in [News, Search] order
    df_var_nf, _ = prepare_keyword_data(kw, df_search, df_news, adf_results)
    
    # ===== N_first: [News, Search] =====
    model_nf = VAR(df_var_nf)
    res_nf = model_nf.fit(optimal_lag)
    irf_nf = res_nf.irf(horizon)
    fevd_nf = res_nf.fevd(horizon)
    
    # IRF: store 4 combinations for each h
    # indices: 0=News, 1=Search (N_first)
    for h in range(horizon + 1):
        # News shock -> Search response
        irf_records.append({
            'keyword': kw, 'ordering': 'N_first',
            'shock': 'News', 'response': 'Search',
            'horizon': h, 'IRF': round(float(irf_nf.irfs[h][1, 0]), 6)
        })
        # Search shock -> News response
        irf_records.append({
            'keyword': kw, 'ordering': 'N_first',
            'shock': 'Search', 'response': 'News',
            'horizon': h, 'IRF': round(float(irf_nf.irfs[h][0, 1]), 6)
        })
    
    # FEVD: store contribution ratios for each h (h=1..horizon)
    for h in range(horizon):
        # News-shock contribution to Search forecast-error variance
        fevd_records.append({
            'keyword': kw, 'ordering': 'N_first',
            'target': 'Search', 'note': 'News',
            'horizon': h + 1,
            'FEVD': round(float(fevd_nf.decomp[1][h, 0]), 6)
        })
        # Search-shock contribution to News forecast-error variance
        fevd_records.append({
            'keyword': kw, 'ordering': 'N_first',
            'target': 'News', 'note': 'Search',
            'horizon': h + 1,
            'FEVD': round(float(fevd_nf.decomp[0][h, 1]), 6)
        })
    
    # ===== S_first: [Search, News] reverse order =====
    df_var_sf = df_var_nf[['Search', 'News']]
    model_sf = VAR(df_var_sf)
    res_sf = model_sf.fit(optimal_lag)
    irf_sf = res_sf.irf(horizon)
    fevd_sf = res_sf.fevd(horizon)
    
    # indices: 0=Search, 1=News (S_first)
    for h in range(horizon + 1):
        # Search shock -> News response
        irf_records.append({
            'keyword': kw, 'ordering': 'S_first',
            'shock': 'Search', 'response': 'News',
            'horizon': h, 'IRF': round(float(irf_sf.irfs[h][1, 0]), 6)
        })
        # News shock -> Search response
        irf_records.append({
            'keyword': kw, 'ordering': 'S_first',
            'shock': 'News', 'response': 'Search',
            'horizon': h, 'IRF': round(float(irf_sf.irfs[h][0, 1]), 6)
        })
    
    for h in range(horizon):
        # Search-shock contribution to News forecast-error variance
        fevd_records.append({
            'keyword': kw, 'ordering': 'S_first',
            'target': 'News', 'note': 'Search',
            'horizon': h + 1,
            'FEVD': round(float(fevd_sf.decomp[1][h, 0]), 6)
        })
        # News-shock contribution to Search forecast-error variance
        fevd_records.append({
            'keyword': kw, 'ordering': 'S_first',
            'target': 'Search', 'note': 'News',
            'horizon': h + 1,
            'FEVD': round(float(fevd_sf.decomp[0][h, 1]), 6)
        })
    
    return irf_records, fevd_records



# ======================================================================
# Batch IRF/FEVD over significant keywords  (source cell 53)
# ======================================================================
def run_irf_fevd_batch(df_valid, df_search, df_news, adf_results,
                       label='main', horizon=10):
    """
    Compute IRF/FEVD for significant keywords (N2S_sig OR S2N_sig == True).
    
    Parameters
    ----------
    df_valid : pd.DataFrame, Step 6 results (includes the type column)
    ...
    label : str, 'main' / 'sensitivity' / 'sens_extra'
    
    Returns
    -------
    tuple (pd.DataFrame, pd.DataFrame)
    """
    sig = df_valid[
        (df_valid['N2S_sig'] == True) | (df_valid['S2N_sig'] == True)
    ].copy()
    
    print(f"\n{'='*60}")
    print(f"Computing IRF/FEVD [{label}] - {len(sig)} significant keywords")
    print(f"{'='*60}")
    
    all_irf, all_fevd = [], []
    
    for _, row in sig.iterrows():
        kw = row['keyword']
        lag = int(row['optimal_lag'])
        try:
            irf_r, fevd_r = compute_irf_fevd(
                kw, df_search, df_news, adf_results,
                optimal_lag=lag, horizon=horizon
            )
            all_irf.extend(irf_r)
            all_fevd.extend(fevd_r)
            print(f"  {kw} (lag={lag}, type={row['type']}): OK")
        except Exception as e:
            print(f"  {kw} error: {str(e)[:60]}")
    
    df_irf = pd.DataFrame(all_irf)
    df_fevd = pd.DataFrame(all_fevd)
    
    # save
    df_irf.to_csv(f'{OUTPUT_DIR}/irf_{label}.csv',
                  index=False, encoding='utf-8-sig')
    df_fevd.to_csv(f'{OUTPUT_DIR}/fevd_{label}.csv',
                   index=False, encoding='utf-8-sig')
    print(f"\nsaved: {OUTPUT_DIR}/irf_{label}.csv")
    print(f"saved: {OUTPUT_DIR}/fevd_{label}.csv")
    
    return df_irf, df_fevd



# ======================================================================
# Bootstrap IRF confidence intervals (Lutkepohl 2005, App. D.3)  (source cell 66)
# ======================================================================
def bootstrap_irf_ci(res, horizon=10, n_boot=500, signif=0.05, seed=42):
    """
    Bootstrap confidence intervals for the IRF of a VAR model.
    
    Parameters
    ----------
    res : VARResults
        Fitted VAR result object.
    horizon : int
        IRF horizon.
    n_boot : int
        Number of bootstrap replications.
    signif : float
        Significance level (default 0.05 -> 95% CI).
    seed : int
        Reproducibility seed.
    
    Returns
    -------
    tuple (ndarray, ndarray, ndarray)
        point_est: (horizon+1, K, K) point estimates
        ci_lower: (horizon+1, K, K) lower bound
        ci_upper: (horizon+1, K, K) upper bound
    """
    np.random.seed(seed)
    
    # original point estimates
    irf_point = res.irf(horizon).orth_irfs  # (horizon+1, K, K) orthogonalized IRF
    
    # residuals and fitted values
    resid = res.resid.values  # (T_eff, K)
    fitted = res.fittedvalues.values  # (T_eff, K)
    T_eff, K = resid.shape
    p = res.k_ar
    endog = res.model.endog  # (T, K) original data
    
    # store bootstrap IRFs
    boot_irfs = np.zeros((n_boot, horizon + 1, K, K))
    
    for b in range(n_boot):
        # residual resampling (simple resampling, not block)
        idx = np.random.choice(T_eff, size=T_eff, replace=True)
        resid_boot = resid[idx]
        
        # reconstruct bootstrap time series
        y_boot = np.zeros_like(endog)
        y_boot[:p] = endog[:p]  # fix initial values
        for t in range(p, len(endog)):
            y_lag = np.concatenate([y_boot[t-j] for j in range(1, p+1)])
            # predict using coefficient matrices + resampled residuals
            coefs = res.coefs  # (p, K, K)
            pred = res.intercept.copy()
            for j in range(p):
                pred += coefs[j] @ y_boot[t-j-1]
            y_boot[t] = pred + resid_boot[t - p]
        
        # re-fit VAR on the bootstrap sample
        try:
            model_boot = VAR(pd.DataFrame(y_boot, columns=res.names))
            res_boot = model_boot.fit(p)
            boot_irfs[b] = res_boot.irf(horizon).orth_irfs
        except Exception:
            boot_irfs[b] = irf_point  # fall back to point estimate on failure
    
    # quantile-based CI
    alpha = signif / 2
    ci_lower = np.percentile(boot_irfs, alpha * 100, axis=0)
    ci_upper = np.percentile(boot_irfs, (1 - alpha) * 100, axis=0)
    
    return irf_point, ci_lower, ci_upper

