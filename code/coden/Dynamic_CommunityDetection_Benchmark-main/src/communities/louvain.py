import community
import networkx as nx
from .utils import map_communities, replace_communities
# from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI


class Louvain():
    """ The Louvain class implements the Louvain algorithm for community 
    detection in graphs. It provides methods to run the algorithm and compute 
    evaluation metrics such as the Adjusted Rand Index (ARI). The Louvain 
    algorithm iteratively optimizes community assignments to maximize 
    modularity.
    
    Parameters
    ----------
        seed (int) : Seed for the random number generator.

    Attributes
    ----------
        seed (int): Seed for the random number generator.
        change_detected (bool): Flag indicating whether a change in the community 
            structure was detected.

    Methods
    -------
        run(G, initialization, previous_comms)
            Runs the Louvain algorithm on a graph.
        get_metrics(y_true, y_pred)
            Computes the evaluation metrics for the community detection results.

    """
    def __init__(self, seed):
        self.seed = seed
        self.change_detected = False 

    def run(self, G, initialization, previous_comms):
        """ Run the Louvain algorithm on a graph.

        Parameters
        ----------
            G (networkx.Graph): Input graph.
            initialization (dict): Initial community assignment for each node.
            previous_comms (dict): Community assignments from the previous 
                snapshot.

        Returns
        -------
            current_comms (dict): Community assignments after running the Louvain 
                algorithm.
            y_pred (list): Predicted community labels.

        """
        # Run independent Louvain
        current_comms = community.best_partition(G, 
                                                 random_state=self.seed,
                                                 partition=initialization)
        # Perform the mapping and replace the snapshot t communities
        mapping = map_communities(previous_comms, current_comms)
        current_comms = replace_communities(current_comms, mapping)
        # # nx.set_node_attributes(G, current_comms, name='community')
        # #y_pred = [current_comms[key] for key in sorted(current_comms)]
        # y_pred = {k:current_comms[k] for k in sorted(G.nodes)}

        return current_comms
    
    def get_metrics(self, y_true, y_pred):
        """ Compute evaluation metrics for the community detection results.

        Parameters
        ----------
            y_true (list): True community labels.
            y_pred (list): Predicted community labels.

        Returns
        -------
            
            ami (float): Adjusted Mutual Info Score Index (AMI) score.

        """
        tot_keys = set(y_pred.keys()).intersection(set(y_true.keys()))
        _y_true, _y_pred = [], []
        for k in tot_keys:
            _y_true.append(y_true[k])
            _y_pred.append(y_pred[k])

        ami = AMI(_y_true, _y_pred)
        
        return ami