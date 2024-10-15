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

cats = numpy.unique(hbp_data['bigcat'])

print()
plt.clf()
ax = plt.figure().add_subplot(projection='3d')
ax.plot(hbp_data[hbp_data['bigcat']==1].values[1],
        hbp_data[hbp_data['bigcat']==1].values[2],
        hbp_data[hbp_data['bigcat']==1].values[3],label=hbp_data[hbp_data['bigcat']==1]['category_name'].values[0],
        color=COLORS[0])
ax.legend()
ax.view_init(elev=20., azim=-35, roll=0)
plt.show()