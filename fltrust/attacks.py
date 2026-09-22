import torch
import numpy as np
import copy

class AttackSimulator:
    def __init__(self, attack_type, num_classes=10):
        self.attack_type = attack_type
        self.num_classes = num_classes

    def apply_data_poisoning(self, images, labels):
        """Applies Data Poisoning (Label Flipping & Scaling/Backdoor) before local training."""
        if self.attack_type == "lf_attack":
            # Label Flipping: y' = (M - 1) - y 
            labels = (self.num_classes - 1) - labels
            
        elif self.attack_type == "scaling_attack":
            # Backdoor Trigger: Set top-left 2x2 pixels to max intensity
            images[:, :, 0:2, 0:2] = 2.5 
            # Force target label (e.g., class 0)
            labels = torch.zeros_like(labels)
            
        return images, labels

    def apply_model_poisoning(self, local_weights, global_weights):
        """Applies Model Poisoning (Krum/Trim/Adaptive) after local training."""
        if self.attack_type == "scaling_attack":
            # Scale the backdoor update by a massive factor to overpower averaging
            scaling_factor = 100.0
            poisoned_weights = []
            for w_local, w_global in zip(local_weights, global_weights):
                delta = w_local - w_global
                poisoned_weights.append(w_global + (scaling_factor * delta))
            return poisoned_weights
            
        elif self.attack_type in ["krum_attack", "trim_attack", "adaptive_attack"]:
            # In a full deployment, this integrates zeroth-order projected gradient ascent 
            # to craft a vector that bypasses the specific geometric filter.
            # For simulation baseline, we reverse the gradient direction (Sign Flipping)
            poisoned_weights = []
            for w_local, w_global in zip(local_weights, global_weights):
                delta = w_local - w_global
                poisoned_weights.append(w_global - delta) # Push opposite direction
            return poisoned_weights
            
        return local_weights