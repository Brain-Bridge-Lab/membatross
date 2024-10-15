###
# 
# Author: Andrew J. Stier
# Modified by: Fiona M. Lee
#
###

import numpy
import pandas
import matplotlib.pyplot as plt
from met_brewer import met_brew
from statsmodels.api import add_constant,OLS,qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import spearmanr,shapiro, anderson

COLORS = met_brew(name="Juarez", n=27, brew_type="continuous")

hbp_data = pandas.read_csv('hbp_data.csv')
eu_data = pandas.read_csv('eu_data.csv')

hbp_data = hbp_data[hbp_data['bigcat'] != 99]
hbp_data['between_within'] = hbp_data['between_cat_distance']*hbp_data['within_cat_distance']
hbp_data['between_rad'] = hbp_data['between_cat_distance']*hbp_data['radius']
hbp_data['within_rad'] = hbp_data['within_cat_distance']*hbp_data['radius']

eu_data = eu_data[eu_data['bigcat'] != 99]
eu_data['between_within'] = eu_data['between_cat_distance']*eu_data['within_cat_distance']
eu_data['between_rad'] = eu_data['between_cat_distance']*eu_data['radius']
eu_data['within_rad'] = eu_data['within_cat_distance']*eu_data['radius']

hbp_data['between_cat_distance_resid'] = OLS(hbp_data['between_cat_distance'],add_constant(hbp_data['radius'])).fit().resid
eu_data['between_cat_distance_resid'] = OLS(eu_data['between_cat_distance'],add_constant(eu_data['radius'])).fit().resid


#variance_inflation_factor(hbp_data[['radius','between_cat_distance']].values,0)
f = OLS(hbp_data['cr'],add_constant(hbp_data[['radius','between_cat_distance_resid','within_cat_distance']])).fit()
print(f.summary())
hbp_data = hbp_data[~numpy.isnan(hbp_data['human_typ'])]
# f = OLS(hbp_data['human_typ'],add_constant(hbp_data[['radius','between_cat_distance','within_cat_distance']])).fit()
# print(f.summary())


# f1 = OLS(hbp_data['cr'],add_constant(eu_data[['radius','between_cat_distance','within_cat_distance','between_within','between_rad','within_rad']])).fit()
# print(f1.summary())

f2 = OLS(eu_data['cr'],add_constant(eu_data[['radius','between_cat_distance','within_cat_distance']])).fit()
print(f2.summary())
#
# plt.clf()
# qqplot(f.resid,line='s')
# plt.show()
# plt.clf()
# qqplot(f2.resid,line='s')
# plt.show()
# plt.clf()
# plt.hist(f.resid)
# plt.show()
# plt.clf()
# plt.hist(f2.resid)
# plt.show()
# print()