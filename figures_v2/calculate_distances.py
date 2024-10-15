###
# 
# Author: Andrew J. Stier
# Modified by: Fiona M. Lee
#
###

from pathlib import Path

import numpy
import pandas
import matplotlib.pyplot as plt
from met_brewer import met_brew
from scipy.stats import ttest_rel
from statsmodels.api import add_constant,OLS

COLORS = met_brew(name="Juarez", n=27, brew_type="continuous")

human_typ = pandas.read_csv('../data/human_rated_typicality.csv')
eu_data = pandas.read_csv('../data/concept_embeddings_cr_cat_16eu.csv')
hbp_data = pandas.read_csv('../data/concept_embeddings_cr_cat_3hbp.csv')
eu_data['human_typ'] = human_typ['typicality']
hbp_data['human_typ'] = human_typ['typicality']

def radius(x):
    return numpy.sqrt(numpy.sum(x**2,axis=1 if len(x.shape)==2 else 0))

def hyperbolic_distance(x,y):
    x = numpy.hstack(x)
    y = numpy.hstack(y)
    r1 = radius(x)
    r2 = radius(y)
    dtheta = numpy.abs(numpy.arccos(numpy.sum(x*y)/(r1*r2)))
    dtheta = numpy.pi-numpy.abs(numpy.pi-dtheta)
    return numpy.arccosh(numpy.cosh(r1)*numpy.cosh(r2)-numpy.sinh(r1)*numpy.sinh(r2)*numpy.cos(dtheta))

hbp_data['radius'] = radius(hbp_data.values[:,1:4].astype(float))
eu_data['radius'] = radius(eu_data.values[:,1:17].astype(float))
distance_file_hbp = 'distances_hbp.npy'
distance_file_hbp_eu = 'distances_hbp_eu.npy'
distance_file_eu = 'distances_eu.npy'
if not Path(distance_file_eu).exists():
    euclidean_distances = numpy.vstack([[numpy.sqrt(numpy.sum((numpy.array(x)-numpy.array(y))**2)) for x in eu_data.values[:,1:17].tolist()] for y in eu_data.values[:,1:17].tolist()])
    numpy.save(distance_file_eu,euclidean_distances)
else:
    euclidean_distances = numpy.load(distance_file_eu)

if not Path(distance_file_hbp_eu).exists():
    euclidean_distances_hbp = numpy.vstack([[numpy.sqrt(numpy.sum((numpy.array(x)-numpy.array(y))**2)) for x in hbp_data.values[:,1:4].tolist()] for y in hbp_data.values[:,1:4].tolist()])
    numpy.save(distance_file_hbp_eu,euclidean_distances_hbp)
else:
    euclidean_distances_hbp = numpy.load(distance_file_hbp_eu)

if not Path(distance_file_hbp).exists():
    hyperbolic_distances = numpy.vstack([[hyperbolic_distance(x,y) for x in hbp_data.values[:,1:4].tolist()] for y in hbp_data.values[:,1:4].tolist()])
    numpy.save(distance_file_hbp,hyperbolic_distances)
else:
    hyperbolic_distances = numpy.load(distance_file_hbp)

within_cat_idx = [numpy.argwhere(hbp_data['bigcat'].values==hbp_data['bigcat'].values[i]) for i in range(hbp_data.shape[0])]
within_cat_idx = [within_cat_idx[i][within_cat_idx[i]!=i] for i in range(hbp_data.shape[0])]

between_cat_idx = [numpy.argwhere(hbp_data['bigcat'].values!=hbp_data['bigcat'].values[i]) for i in range(hbp_data.shape[0])]

cats = numpy.unique(hbp_data['bigcat'])
avg_distances_hbp = numpy.hstack([hyperbolic_distances[hbp_data['bigcat']==cat,:][:,hbp_data['bigcat']==cat][numpy.tril_indices(sum(hbp_data['bigcat']==cat),k=-1)].mean() for cat in cats])
avg_radius_hbp = numpy.hstack([hbp_data[hbp_data['bigcat']==cat]['radius'].mean() for cat in cats])
avg_distances_hbp_eu = numpy.hstack([euclidean_distances_hbp[hbp_data['bigcat']==cat,:][:,hbp_data['bigcat']==cat][numpy.tril_indices(sum(hbp_data['bigcat']==cat),k=-1)].mean() for cat in cats])
avg_distances_eu = numpy.hstack([euclidean_distances[eu_data['bigcat']==cat,:][:,eu_data['bigcat']==cat][numpy.tril_indices(sum(eu_data['bigcat']==cat),k=-1)].mean() for cat in cats])


hbp_data['within_cat_distance_raw'] = numpy.hstack([hyperbolic_distances[i,:][
                                                    within_cat_idx[i]].mean()-
                                                avg_distances_hbp[numpy.argwhere(cats==hbp_data['bigcat'].values[i]).flatten()]
                                                for i in range(len(within_cat_idx))])

hbp_data['within_cat_distance'] = numpy.hstack([hyperbolic_distances[i,:][
                                                    within_cat_idx[i]].mean()-
                                                avg_distances_hbp[numpy.argwhere(cats==hbp_data['bigcat'].values[i]).flatten()]
                                                for i in range(len(within_cat_idx))])
hbp_data['between_cat_distance'] = numpy.hstack([hyperbolic_distances[i,:][between_cat_idx[i].flatten()].mean() for i in range(len(between_cat_idx))])

hbp_data['between_cat_distance_eu'] = numpy.hstack([euclidean_distances_hbp[i,:][between_cat_idx[i].flatten()].mean() for i in range(len(between_cat_idx))])

eu_data['within_cat_distance'] = numpy.hstack([euclidean_distances[i,:][within_cat_idx[i]].mean()-
                                               avg_distances_eu[numpy.argwhere(cats==eu_data['bigcat'].values[i]).flatten()]
                                               for i in range(len(within_cat_idx))])
eu_data['between_cat_distance'] = numpy.hstack([euclidean_distances[i,:][between_cat_idx[i].flatten()].mean() for i in range(len(between_cat_idx))])

print('Hyperbolic (euclidean) within/between vs. Euclidean within/between')
print(ttest_rel(avg_distances_hbp_eu/[hbp_data[hbp_data['bigcat']==cat]['between_cat_distance_eu'].mean() for cat in cats],avg_distances_eu/[eu_data[eu_data['bigcat']==cat]['between_cat_distance'].mean() for cat in cats]))
hbp_data['cat_name'] = hbp_data['cat_name'].astype(str).map(lambda x: x if x!='nan' else 'other')
hbp_data['cat_name'] = hbp_data['cat_name'].astype(str).map(lambda x: x if x!='nan' else 'other')

plt.clf()
x= avg_distances_hbp_eu/[hbp_data[hbp_data['bigcat']==cat]['between_cat_distance_eu'].mean() for cat in cats]
y = avg_distances_eu/[eu_data[eu_data['bigcat']==cat]['between_cat_distance'].mean() for cat in cats]
plt.scatter(x,y,c=COLORS[:-3])
[plt.scatter([],[],c=COLORS[i],label=hbp_data[hbp_data['bigcat']==cats[i]]['cat_name'].values[0]) for i in range(len(cats))]
plt.xlabel('Hyperbolic within/between (euclidean distance)')
plt.ylabel('Euclidean within/between')
plt.legend(ncol=2,fontsize='x-small')
t = ttest_rel(x,y)
plt.text(.65,.50,r'$t_{rel}=%.2f$' % t[0] + '\n' + r'$p<%.3f$' % t[1],size=14)
plt.savefig('./within_between.png',dpi=300)


plt.clf()
x= avg_distances_hbp/[hbp_data[hbp_data['bigcat']==cat]['between_cat_distance'].mean() for cat in cats]
y = avg_distances_eu/[eu_data[eu_data['bigcat']==cat]['between_cat_distance'].mean() for cat in cats]
plt.scatter(x,y,c=COLORS[:-3])
[plt.scatter([],[],c=COLORS[i],label=hbp_data[hbp_data['bigcat']==cats[i]]['cat_name'].values[0]) for i in range(len(cats))]
plt.xlabel('Hyperbolic within/between (hyperbolic distance)')
plt.ylabel('Euclidean within/between')
plt.legend(ncol=2,fontsize='x-small')
t = ttest_rel(x,y)
plt.text(1,.50,r'$t_{rel}=%.2f$' % t[0] + '\n' + r'$p<%.3f$' % t[1],size=14)
plt.savefig('./within_between_native.png',dpi=300)

print('Hyperbolic (Hyperbolic) within/between vs. Euclidean within/between')
print(ttest_rel(avg_distances_hbp/[hbp_data[hbp_data['bigcat']==cat]['between_cat_distance'].mean() for cat in cats],avg_distances_eu/[eu_data[eu_data['bigcat']==cat]['between_cat_distance'].mean() for cat in cats]))


plt.clf()
x= avg_distances_hbp/[hbp_data[hbp_data['bigcat']==cat]['between_cat_distance'].mean() for cat in cats]
y = [hbp_data[hbp_data['bigcat']==cat]['radius'].mean() for cat in cats]
plt.scatter(x,y,c=COLORS[:-3])
[plt.scatter([],[],c=COLORS[i],label=hbp_data[hbp_data['bigcat']==cats[i]]['cat_name'].values[0]) for i in range(len(cats))]
plt.xlabel('Hyperbolic within/between (hyperbolic distance)')
plt.ylabel('Hyperbolic Radius')
plt.legend(ncol=2,fontsize='xx-small')
plt.savefig('./within_between_radius_hbp_native.png',dpi=300)

plt.clf()
x = avg_distances_hbp_eu/[hbp_data[hbp_data['bigcat']==cat]['between_cat_distance_eu'].mean() for cat in cats]
y = [hbp_data[hbp_data['bigcat']==cat]['radius'].mean() for cat in cats]
plt.scatter(x,y,c=COLORS[:-3])
[plt.scatter([],[],c=COLORS[i],label=hbp_data[hbp_data['bigcat']==cats[i]]['cat_name'].values[0]) for i in range(len(cats))]
plt.xlabel('Hyperbolic within/between (euclidean distance)')
plt.ylabel('Hyperbolic Radius')
plt.legend(ncol=2,fontsize='xx-small')
plt.savefig('./within_between_radius_hbp_euclidean.png',dpi=300)

plt.clf()
x = avg_distances_eu/[eu_data[eu_data['bigcat']==cat]['between_cat_distance'].mean() for cat in cats]
y = [eu_data[eu_data['bigcat']==cat]['radius'].mean() for cat in cats]
plt.scatter(x,y,c=COLORS[:-3])
[plt.scatter([],[],c=COLORS[i],label=eu_data[eu_data['bigcat']==cats[i]]['cat_name'].values[0]) for i in range(len(cats))]
plt.xlabel('Euclidean within/between')
plt.ylabel('Euclidean Radius')
plt.legend(ncol=2,fontsize='xx-small')
plt.savefig('./within_between_radius_euclidean.png',dpi=300)

eu_data['between_cat_distance_resid'] = OLS(eu_data['between_cat_distance'],add_constant(eu_data['radius'])).fit().resid
hbp_data['between_cat_distance_resid'] = OLS(hbp_data['between_cat_distance'],add_constant(hbp_data['radius'])).fit().resid

hbp_data.to_csv('hbp_data.csv')
eu_data.to_csv('eu_data.csv')
print()