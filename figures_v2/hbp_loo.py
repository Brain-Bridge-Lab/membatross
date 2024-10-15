from pathlib import Path

import numpy
import pandas
import matplotlib.pyplot as plt
from met_brewer import met_brew
from statsmodels.api import add_constant,OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import spearmanr

COLORS = met_brew(name="Juarez", n=27, brew_type="continuous")

hbp_data = pandas.read_csv('hbp_data.csv')
eu_data = pandas.read_csv('eu_data.csv')

distance_file_hbp = 'distances_hbp.npy'
hyperbolic_distances = numpy.load(distance_file_hbp)

def loo_hbp(row):
    rownow = rows[rows != row]
    datanow = hbp_data.iloc[rownow].copy()
    distnow = hyperbolic_distances[rownow, :][:, rownow]

    within_cat_idx = [numpy.argwhere(datanow['bigcat'] == datanow['bigcat'].values[i]) for i in
                      range(datanow.shape[0])]
    within_cat_idx = [within_cat_idx[i][within_cat_idx[i] != i] for i in range(datanow.shape[0])]

    between_cat_idx = [numpy.argwhere(datanow['bigcat'] != datanow['bigcat'].values[i]) for i in
                       range(datanow.shape[0])]

    cats = numpy.unique(datanow['bigcat'])
    avg_distances_hbp = numpy.hstack([distnow[datanow['bigcat'] == cat, :][:, datanow['bigcat'] == cat][
                                          numpy.tril_indices(sum(datanow['bigcat'] == cat), k=-1)].mean() for cat in
                                      cats])
    datanow['within_cat_distance'] = numpy.hstack([distnow[i, :][within_cat_idx[i]].mean() -
                                                   avg_distances_hbp[
                                                       numpy.argwhere(cats == datanow['bigcat'].values[i]).flatten()]
                                                   for i in range(len(within_cat_idx))])
    datanow['between_cat_distance'] = numpy.hstack(
        [distnow[i, :][between_cat_idx[i].flatten()].mean() for i in range(len(between_cat_idx))])

    fbet = OLS(datanow['between_cat_distance'],
                                           add_constant(datanow['radius'])).fit()
    datanow['between_cat_distance'] = fbet.resid
    datanow = datanow[datanow['bigcat'] != 99]
    f = OLS(datanow['cr'],
        add_constant(datanow[['radius', 'between_cat_distance', 'within_cat_distance']])).fit()
    datloo = hbp_data.iloc[row][['radius', 'between_cat_distance', 'within_cat_distance']].values
    datloo[1] = datloo[1]-fbet.predict(numpy.hstack(([1],datloo[0])))[0]
    return f.predict(numpy.hstack(([1],datloo)))[0]

loo_hbp_file = 'loo_pred_hbp.npy'
if not Path(loo_hbp_file).exists():
    rows = numpy.array(list(range(len(hbp_data))))
    loo_hbp(0)
    from multiprocessing import Pool
    p = Pool(7)
    preds = p.map(loo_hbp,rows)
    preds = numpy.array(preds)
    numpy.save(loo_hbp_file,preds)
else:
    preds = numpy.load(loo_hbp_file)
hbp_data['loo_pred'] = preds

hbp_data_og = hbp_data.copy()
hbp_data = hbp_data[hbp_data['bigcat'] != 99]

r2_loo = 1-((hbp_data['cr'] - hbp_data['loo_pred'])**2).sum()/((hbp_data['cr'] - hbp_data['cr'].mean())**2).sum()
r2_loo_boot = [hbp_data.sample(frac=1,replace=True) for i in range(1000)]
# r2_loo_boot = [1-((x['cr'] - x['loo_pred'])**2).sum()/((x['cr'] - x['cr'].mean())**2).sum() for x in r2_loo_boot]
r2_loo_boot = [OLS(x['cr'], add_constant(x['loo_pred'])).fit().rsquared for x in r2_loo_boot]
plt.clf()
plt.hist(r2_loo_boot)
plt.axvline(r2_loo,color='k')
plt.axvline(numpy.percentile(r2_loo_boot,[2.5]),linestyle='--',color='k')
plt.axvline(numpy.percentile(r2_loo_boot,[97.5]),linestyle='--',color='k')
plt.show()

surrogate_file = 'surrogates_hbp.npy'
if not Path(surrogate_file).exists():
    from brainsmash.mapgen.base import Base
    D = hyperbolic_distances.copy()
    D[numpy.isnan(D)] = 0
    numpy.fill_diagonal(D,0)
    b = Base(x = hbp_data_og['cr'].values, D=D, resample=False,n_jobs=7)
    surrogates = b(n=1000)
    numpy.save(surrogate_file,surrogates)
else:
    surrogates = numpy.load(surrogate_file)

r2s = [OLS(surrogates[i,:][hbp_data_og['bigcat'] != 99],add_constant(hbp_data_og[hbp_data_og['bigcat'] != 99][['radius', 'between_cat_distance', 'within_cat_distance']])).fit().rsquared for i in range(1000)]
print((numpy.array(r2s)>.153).sum()/1000)

print()
