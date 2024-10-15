import numpy
import matplotlib.pyplot as plt
from statsmodels.api import add_constant,OLS
from scipy.stats import spearmanr, ttest_ind
import pandas
from met_brewer import met_brew
from pingouin import linear_regression

COLORS = met_brew(name="Juarez", n=27, brew_type="continuous")

hbp_data = pandas.read_csv('hbp_data.csv')
eu_data = pandas.read_csv('eu_data.csv')

loo_hbp_file = 'loo_pred_hbp.npy'
preds = numpy.load(loo_hbp_file)
hbp_data['loo_pred'] = preds
hbp_data = hbp_data[hbp_data['bigcat'] != 99]
r2_loo_hbp = 1-((hbp_data['cr'] - hbp_data['loo_pred'])**2).sum()/((hbp_data['cr'] - hbp_data['cr'].mean())**2).sum()
r2_loo_boot_hbp = [hbp_data.sample(frac=1,replace=True) for i in range(1000)]
r2_loo_boot_hbp = [OLS(x['cr'], add_constant(x['loo_pred'])).fit().rsquared for x in r2_loo_boot_hbp]

loo_eu_file = 'loo_pred_eu.npy'
preds = numpy.load(loo_eu_file)
eu_data['loo_pred'] = preds
eu_data = eu_data[eu_data['bigcat'] != 99]
r2_loo_eu = OLS(eu_data['cr'], add_constant(eu_data['loo_pred'])).fit().rsquared
r2_loo_boot_eu = [eu_data.sample(frac=1,replace=True) for i in range(1000)]
r2_loo_boot_eu = [OLS(x['cr'], add_constant(x['loo_pred'])).fit().rsquared for x in r2_loo_boot_eu]

########################################################################
plt.clf()
vp = plt.violinplot([r2_loo_boot_hbp,r2_loo_boot_eu],showmeans=False,showextrema=False,quantiles=[[.025,.975],[.025,.975]])
vp['cquantiles'].set_color('k')
vp['bodies'][0].set_color(COLORS[0])
vp['bodies'][1].set_color(COLORS[-1])
plt.xticks([1,2],['3D Hyperbolic','16D Euclidean'])
plt.ylabel(r'Model $R^2$')
plt.savefig('/home/andrewstier/Downloads/membatross/model_r2s.png',dpi=300)

########################################################################
plt.clf()
plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
t = 'radius'
x = hbp_data[t]
y = eu_data[t]
plt.scatter(x,y,alpha=.25)
rs = spearmanr(x,y)
plt.title(r'$r_s=%.2f$' % rs[0] + '\n' + r'$p=%.2e$' %rs[1])
plt.xlabel('3D Hyperbolic Radius')
plt.ylabel('16D Euclidean Radius')

plt.subplot(1,3,2)
t = 'between_cat_distance'
x = hbp_data[t]
y = eu_data[t]
plt.scatter(x,y,alpha=.25)
rs = spearmanr(x,y)
plt.title(r'$r_s=%.2f$' % rs[0] + '\n' + r'$p=%.2e$' %rs[1])
plt.xlabel('3D Hyperbolic Repulsion')
plt.ylabel('16D Euclidean Repulsion')

plt.subplot(1,3,3)
t = 'within_cat_distance'
x = hbp_data[t]
y = eu_data[t]
plt.scatter(x,y,alpha=.25)
rs = spearmanr(x,y)
plt.title(r'$r_s=%.2f$' % rs[0] + '\n' + r'$p=%.2e$' %rs[1])
plt.xlabel('3D Hyperbolic Attraction')
plt.ylabel('16D Euclidean Attraction')

plt.tight_layout()
plt.savefig('/home/andrewstier/Downloads/membatross/attraction_repulsion_scatter.png',dpi=300)
########################################################################

# hbp_data['between_cat_distance'] = OLS(hbp_data['between_cat_distance'],add_constant(hbp_data['radius'])).fit().resid
# eu_data['between_cat_distance'] = OLS(eu_data['between_cat_distance'],add_constant(eu_data['radius'])).fit().resid
#
# f = linear_regression(hbp_data[['radius','between_cat_distance','within_cat_distance']],
#                         hbp_data['cr'],add_intercept=True,relimp=True)
#

print()