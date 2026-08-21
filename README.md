# Lead–lag analysis of news coverage and OGD portal search (South Korea)

Analysis code for the paper examining the lead–lag relationship between news
coverage and open government data (OGD) portal search volume in South Korea,
using bivariate VAR models with Granger causality tests, IRF, and FEVD.

## Contents

- `var_granger.py` — core analysis functions (see below).
- `analysis.ipynb` — end-to-end pipeline that imports `var_granger.py` and
  reproduces the main and sensitivity analyses, the VAR stability check,
  Table A2, the Section 2.2 correlation, and Figure A1.
- `data/keyword_top50_2201_2505.xlsx` — search-volume data (see Data).
- `requirements.txt` — Python dependencies.

## Core functions (`var_granger.py`)

| Function | Purpose | Paper section |
|---|---|---|
| `calculate_news_intensity` | Reverse-linear rank weighting of news coverage | News intensity weighting |
| `apply_lod_imputation` | 1/√2 substitution for left-censored search volume | Missing-data imputation |
| `adf_test` | ADF stationarity test (level → first difference) | Keyword selection & stationarity |
| `describe_distribution` | Descriptive statistics | Descriptive statistics |
| `prepare_keyword_data` | Build [News, Search] input per keyword | Bivariate VAR model |
| `analyze_keyword` | VAR + AIC lag selection + bidirectional Granger | VAR / Granger causality |
| `run_var_granger_batch` | Batch over keyword list | Results (main & sensitivity) |
| `compute_irf_fevd` | Orthogonalized IRF/FEVD, both Cholesky orders | IRF and FEVD |
| `run_irf_fevd_batch` | Batch IRF/FEVD over significant keywords | IRF and FEVD results |
| `bootstrap_irf_ci` | Bootstrap IRF CIs (Lütkepohl 2005, App. D.3) | Figure A1 |

## Data

### Search-volume data (included)

Monthly top-50 search keywords from South Korea's Open Government Data portal
(data.go.kr), provided by the Open Data Center of the National Information
Society Agency (NIA) through the open-data provision procedure (approved for
research use). Covers January 2022 – May 2025 (41 months); for each month, the
keywords ranked in the top 50 by search volume and their search counts. The
top-50 cap is the provider's release standard, which produces the left-censored
structure handled by the 1/√2 imputation. The portal operator (Open Data
Utilization Support Center) has confirmed this data may be redistributed, so it
is included as `data/keyword_top50_2201_2505.xlsx`.

### News data (not included — collect from BigKinds)

> ⚠️ **Copyright notice.** BigKinds article data are **not redistributed** in
> this repository, in accordance with BigKinds' terms of use. The same dataset
> can be collected directly with the settings below.

News data were collected from [BigKinds](https://www.bigkinds.or.kr) (Korea
Press Foundation), which is publicly accessible free of charge. Collection
settings used in the paper:

| Setting | Value |
|---|---|
| Period | 2022-01-01 – 2025-05-31 |
| Outlets (9) | Chosun Ilbo, JoongAng Ilbo, Dong-A Ilbo, Hankyoreh, Kyunghyang Shinmun (five national dailies); KBS, MBC, SBS, YTN (four national broadcasters) |
| Categories | all categories of the integrated classification (politics, economy, society, culture, international, regional, sports, IT/science) |
| Filter | BigKinds "analysis articles" filter ON (excludes near-duplicates, personnel notices, obituaries, briefs, photo items) |
| Download | BigKinds data-download service (Excel), collected daily |
| Scale | ≈ 2,056,261 articles |

The analysis uses two columns of the BigKinds download: `일자` (article date)
and `특성추출(가중치순 상위 50개)` (top-50 feature-extracted keywords per
article, identified by BigKinds' TextRank algorithm and listed in order of
within-article importance). Merge the downloaded files into a single CSV named
`news_merged.csv`, preserving these columns.

### Directory layout

```
data/
  keyword_top50_2201_2505.xlsx   # included (search volume)
  news_merged.csv                # collect from BigKinds as described above
```
`DATA_DIR` and `OUTPUT_DIR` are set at the top of `analysis.ipynb`.

## Note on preserved Korean

Executable logic is unchanged from the original research code; only comments,
docstrings, and result-column names were translated to English. **Korean is
intentionally retained** where it refers to source-data columns and keyword
values, which must match the raw files:

| Korean (kept) | Meaning |
|---|---|
| `특성추출(가중치순 상위 50개)` | BigKinds extracted-keyword column |
| `일자` | article date | 
| `연도`, `월`, `순위`, `검색어`, `검색 건수` | search-data columns |
| `반려동물`, `부동산`, `관광`, `아파트`, `게임`, … | keyword values (analysis units) |

## Methodological notes

- **1/√2 imputation for left-censored search volume.** Months in which a keyword
  falls outside the top-50 list are treated as left-censored below a limit of
  detection (LOD). Following Ganser & Hewett (2010), the value LOD × (1/√2) is
  used as the empirical substitution; Helsel (2012) cautions against constant
  substitution in general, which is why the keyword set is restricted to keep
  censoring within the validated range (≤50% of months).
- **`irf_resim` diagnostic** (last notebook cell before the appendix note).
  statsmodels' built-in `irf_resim` returned identical values across all 500
  bootstrap replications (std = 0), so IRF confidence intervals are computed with
  a manual residual-resampling bootstrap following Lütkepohl (2005, Appendix D.3,
  with residual centering; `bootstrap_irf_ci`).

## Environment

Python 3.10, statsmodels 0.14.6 (see `requirements.txt`).

## Citation

If you use this code or data, please cite the paper (bibliographic details to
be finalized upon publication):

```
[Author names] (in review). The lead–lag relationship between news coverage and
open government data portal search in South Korea. Submitted to Government
Information Quarterly.
```
This section will be updated with the full citation and DOI upon acceptance.

## License

- **Code** is released under the MIT License (see `LICENSE`).
- **Search-volume data** originate from Korea's Open Government Data portal
  (data.go.kr); redistribution here was confirmed by the portal operator.
  Please attribute the source when reusing.
- **BigKinds news data** are subject to BigKinds' terms of use and are not
  included in this repository.
