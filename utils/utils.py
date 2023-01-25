import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Utils():
    
    @staticmethod
    def get_distance(model1, model2):
        with torch.no_grad():
            model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
            model2_flattened = nn.utils.parameters_to_vector(model2.parameters())
            distance = torch.square(torch.norm(model1_flattened - model2_flattened))
        return distance

    @staticmethod
    def get_distances_from_current_model(current_model, party_models):
        num_updates = len(party_models)
        distances = np.zeros(num_updates)
        for i in range(num_updates):
            distances[i] = Utils.get_distance(current_model, party_models[i])
        return distances

    def evaluate(testloader, model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

