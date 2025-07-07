from cdlib import algorithms
import networkx as nx
from .utils import map_communities, replace_communities
# from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI


class Leiden():
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

    def run(self, G_str, initialization, previous_comms):
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
      
        G1=G_str.copy()
        
        connected_components = list(nx.connected_components(G1))
        if len(connected_components)>1: #Graph is disconnected, Leiden does not work on disconnected Graph
            for i in range(1,len(connected_components)):
                node=list(connected_components[i])[0]
                target=list(connected_components[i-1])[0]
                G1.add_edge(node, target,weight=0.0001)
            assert nx.is_connected(G1)
        
        if initialization==None:
            reoredered_init=None
        else:
            reoredered_init = list()
            init_cnt = 0
            for node in G1.nodes:
                # c = initialization[int(node)]
                c = initialization[node]
            

                # reoredered_init.append(int(c))
                reoredered_init.append(c)
            # print(G1.number_of_nodes(),len(reoredered_init))
                
            assert len(reoredered_init)==G1.number_of_nodes()
        # print(reoredered_init)
        # Run independent Louvain
        current_comms = algorithms.leiden(G1, 
                                          weights='weight',
                                          initial_membership=reoredered_init)
        current_comms = dict(current_comms.to_node_community_map())
        # current_comms = {int(k):current_comms[k][0] for k in current_comms}
        current_comms = {k:int(current_comms[k][0]) for k in current_comms}
        

        # Perform the mapping and replace the snapshot t communities
        mapping = map_communities(previous_comms, current_comms)
        current_comms = replace_communities(current_comms, mapping)
        
        
        # nx.set_node_attributes(G, current_comms, name='community')

        #y_pred = [current_comms[key] for key in sorted(current_comms)]
        # y_pred = {k:current_comms[k] for k in sorted(G1.nodes)}
        # print(current_comms)
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