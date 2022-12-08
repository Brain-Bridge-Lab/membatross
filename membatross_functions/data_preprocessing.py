import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.io import loadmat

def generate_feature_embeddings(embeddings_file):
	"""
	Generate feature embeddings from MDS results.
	Input
	    embeddings_file: file that contains 500 layers of embeddings generated from MDS.
	Output
	    mean_feature_embed: feature coordinates of the representational space. 
	"""
	feature_embeddings = np.load(embeddings_file)
	
	### procrustes of feature embeddings
	feature_embed_reference = feature_embeddings[:,:,0]
	feature_embed_output = [feature_embed_reference]
	for i in range(1, feature_embeddings.shape[-1]):
    		R, scale = orthogonal_procrustes(feature_embeddings[:,:,i], feature_embed_reference)
    		feature_embed_output.append(feature_embeddings[:,:,i]@R)
	mean_feature_embed = np.array(feature_embed_output).mean(0)
	return mean_feature_embed


def generate_concept_loadings(embeddings_file, original_loadings_file):
	"""
        Generate concept loadings using the feature embeddings from MDS.
        Input 
            embeddings_file: file that contains 500 layers of embeddings generated from MDS. 
            original_loadings_file: file that contains the 1854 concepts represented by 49 features.
	Output
            mean_concept_in_feat: concept coordinates in the representational space.
        """
	feature_embeddings = np.load(embeddings_file)
	original_embeddings = np.loadtxt(original_loadings_file)
	
	### normalize embeddings and transform concept embeddings to loadings in hyperbolic feature space
	concept_in_feat_coord = []
	normalized_embeddings = (original_embeddings.T/np.sqrt((original_embeddings**2).sum(axis=1))).T
	for layer in range(feature_embeddings.shape[-1]):
    		concept_in_feat_coord.append(normalized_embeddings@feature_embeddings[:,:,layer])
	### procrustes of concept loadings in hyperbolic feature space
	concept_in_feat_reference = concept_in_feat_coord[0]
	concept_in_feat_output = [concept_in_feat_reference]
	for i in range(1, len(concept_in_feat_coord)):
    		R, scale = orthogonal_procrustes(concept_in_feat_coord[i], concept_in_feat_reference)
    		concept_in_feat_output.append(concept_in_feat_coord[i]@R)
	mean_concept_in_feat = pd.DataFrame(np.array(concept_in_feat_output).mean(axis=0))
	
	return mean_concept_in_feat


def merge_memorability(concepts, memorability_file):
	
	"""
        Merge memorability data to concept coordinates.
        Input 
            concepts: concept coordinates in the representational space.
            memorability_file: file that contains memorability scores of all object images in every concept.
        Output
            cat_mapping_mem: concept coordinates with memorability and concept names.
        """
	# load categories and memorability score of each concept
	concept_data = pd.read_csv(memorability_file)
	concept_data['concept_name'] = concept_data['file_path'].apply(lambda x: x.split('/')[1])
	cat_concept_mem = concept_data[['cr', 'smallcat', 'bigcat', 'concept_name']].groupby('smallcat').agg({'cr': 'mean', 'smallcat':'mean', 'bigcat':'mean', 'concept_name':'first'}).drop(columns=['smallcat'])
	cat_concept_mem['bigcat'] = cat_concept_mem['bigcat'].astype(int).replace(0, 99)
	cat_concept_mem = cat_concept_mem.reset_index().drop(columns=['smallcat'])
	cat_mapping_mem = concepts.merge(cat_concept_mem, how='inner', left_index=True, right_index=True)
	return cat_mapping_mem

def merge_category(concepts, category_file):

        """
        Merge category data to concept coordinates.
        Input
            concepts: concept coordinates in the representational space.
            category_file: file that contains category names and mapping.
        Output
            cat_concept_mem: concept coordinates with category and concept names.
        """
        # load categories and memorability score of each concept
        cat = loadmat(category_file)
        cat_names = [item[0] for item in cat['categories'][0]]
        cat_names.append('na')
        cat_names = pd.DataFrame(cat_names, columns=['cat_name']).reset_index()
        cat_names['index'] = cat_names['index']+1
        cat_names.iloc[27] = [99, np.nan]
        cat_concept_mem = concepts.merge(cat_names, how='left', left_on='bigcat', right_on='index').drop(columns=['index'])
        
        return cat_concept_mem
	

def load_rated_typicality(typicality_file):

        """
        Load typicality data for 1619 concept.
        Input
            concepts: concept coordinates in the representational space.
            typicality_file: file that contains human rated typicality scores.
        Output
            concept_typicality_rated: mapping of concept, category, and human rated typicality.
        """
	### load typicality score
        cat_typicality = loadmat(typicality_file)
        concept_idx = []
        concept_typ = []
        concept_cat = []
        for cat_idx in range(27):
                concept_idx.extend(pd.DataFrame(cat_typicality['category27_ind'][0][cat_idx])[0].to_list())
                concept_typ.extend(pd.DataFrame(cat_typicality['category27_typicality_rating_normed'][cat_idx][0])[0].to_list())
                concept_cat.extend([cat_idx+1 for i in range(len(cat_typicality['category27_ind'][0][cat_idx]))])
        concept_typicality_rated = pd.DataFrame({'cat': concept_cat, 'concept': concept_idx, 'typicality': concept_typ})
        concept_typicality_rated['concept'] = concept_typicality_rated['concept']-1

        return concept_typicality_rated

