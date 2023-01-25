import abc
import torch
from torch import nn
import copy

""""
Fusion: base class for fusion algorithms
FusionAvg: compute average across all parties
FusionRetrain: compute average across all parties except the target one
"""

class Fusion(abc.ABC):

    """
    Base class for Fusion
    """

    def __init__(self, num_parties):
        self.name = "fusion"
        self.num_parties = num_parties
        
    def average_selected_models(self, selected_parties, party_models):
        with torch.no_grad():
            sum_vec = nn.utils.parameters_to_vector(party_models[selected_parties[0]].parameters())
            if len(selected_parties) > 1:
                for i in range(1,len(selected_parties)):
                    sum_vec += nn.utils.parameters_to_vector(party_models[selected_parties[i]].parameters())
                sum_vec /= len(selected_parties)

            model = copy.deepcopy(party_models[0])
            nn.utils.vector_to_parameters(sum_vec, model.parameters())
        return model.state_dict()
            
            
    @abc.abstractmethod
    def fusion_algo(self, party_models, current_model=None):
        raise NotImplementedError


class FusionAvg(Fusion):

    def __init__(self, num_parties):
        super().__init__(num_parties)
        self.name = "Fusion-Average"

    def fusion_algo(self, party_models, current_model=None):
        selected_parties = [i for i in range(self.num_parties)]
        aggregated_model_state_dict = super().average_selected_models(selected_parties, party_models)
        return aggregated_model_state_dict 


class FusionRetrain(Fusion):

    def __init__(self, num_parties):
        super().__init__(num_parties)
        self.name = "Fusion-Retrain"
        
    # Currently, we assume that the party to be erased is party_id = 0
    def fusion_algo(self, party_models, current_model=None):
        selected_parties = [i for i in range(1,self.num_parties)]
        aggregated_model_state_dict = super().average_selected_models(selected_parties, party_models)
        return aggregated_model_state_dict 
