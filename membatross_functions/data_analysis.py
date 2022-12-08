import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from hyperbolicMDS.mds import poincare_dist_vec
from sklearn import preprocessing 


def within_category_typicality(group, original_embeddings):
    """
    Calculate the within-category typicality of each concept using average similarity within category 
    """
    feat_matrix = []
    for concept in group.index:
        feat_matrix.append(original_embeddings[concept])
    coef_matrix = np.corrcoef(feat_matrix)
    typicality = (np.sum(coef_matrix, axis=1)-1)/len(coef_matrix)
    group['typicality'] = typicality
    return group

def get_contrast_cat_typ(df, measure, cat_col, by):
    """
    Compute the contrast typicality of each concept using its average distance from other categories
    measure = eu (euclidean) or hbp (poincare)
    by = avg (typicality by avg distance) or min (typicality by min distance)
    """
    typ_contrast = {}
    for cat in df[cat_col].unique():
        category = df[df[cat_col] == cat]
        others = df[df[cat_col] != cat]
        for idx in range(len(category)):
            concept = category.iloc[idx]
            typ = 0.0
            if measure == 'eu':
                if by == 'avg':
                    typ = (np.sqrt(((others[[0,1,2]] - concept[[0,1,2]])**2).sum(1))).mean(0)
                else: 
                    typ = (np.sqrt(((others[[0,1,2]] - concept[[0,1,2]])**2).sum(1))).min(0)
            else:
                tmp = pd.concat([pd.DataFrame(concept).T, others])
                tmp[[0,1,2]] = tmp[[0,1,2]].astype(float)
                if by == 'avg':
                    typ = poincare_dist_vec(tmp[[0,1,2]].values)[0].mean(0)
                else:
                    typ = poincare_dist_vec(tmp[[0,1,2]].values)[0][1:].min()
            typ_contrast[concept.name] = typ
    df['typ_contrast'] = pd.Series(typ_contrast)
    # rescale typicality
    min_max_scaler = preprocessing.MinMaxScaler()
    df['typ_contrast'] = min_max_scaler.fit_transform(df[['typ_contrast']].values)
    df['typ_contrast'] = 1 - df['typ_contrast']
    
    return df

