import hdbscan
import numpy as np

        
def hdbscan_delegator(model, func: str, embeddings: np.ndarray = None):
    """ Function used to select the HDBSCAN-like model for generating 
    predictions and probabilities.

    Arguments:
        model: The cluster model.
        func: The function to use. Options:
                - "approximate_predict"
                - "all_points_membership_vectors"
        embeddings: Input embeddings for "approximate_predict"
    """

    # Approximate predict
    if func == "approximate_predict":
        if isinstance(model, hdbscan.HDBSCAN):
            predictions, probabilities = hdbscan.approximate_predict(model, embeddings)
            return predictions, probabilities
        elif "cuml" and "hdbscan" in str(type(model)).lower():
            from cuml.cluster import approximate_predict
            predictions, probabilities = approximate_predict(model, embeddings)
            return predictions, probabilities
        else:
            predictions = model.predict(embeddings)
            return predictions, None

    # All points membership
    if func == "all_points_membership_vectors":
        if isinstance(model, hdbscan.HDBSCAN):
            return hdbscan.all_points_membership_vectors(model)
        elif "cuml" and "hdbscan" in str(type(model)).lower():
            from cuml import cluster
            return cluster.all_points_membership_vectors(model)
        else:
            return None


def is_supported_hdbscan(model):
    """ Check whether the input model is a supported HDBSCAN-like model """
    if isinstance(model, hdbscan.HDBSCAN):
        return True
    elif "cuml" and "hdbscan" in str(type(model)).lower():
        return True
    else:
        return False
