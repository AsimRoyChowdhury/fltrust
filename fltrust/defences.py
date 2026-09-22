import numpy as np

def aggregate_krum(client_updates, num_malicious):
    """Selects the single update with the lowest squared distance to its neighbors."""
    num_clients = len(client_updates)
    neighbors_to_keep = num_clients - num_malicious - 2
    
    scores = []
    for i in range(num_clients):
        distances = []
        for j in range(num_clients):
            if i != j:
                dist = np.linalg.norm(client_updates[i] - client_updates[j]) ** 2
                distances.append(dist)
        distances.sort()
        scores.append(sum(distances[:neighbors_to_keep]))
        
    best_index = np.argmin(scores)
    return client_updates[best_index]

def aggregate_trimmed_mean(client_updates, num_malicious):
    """Sorts each parameter coordinate and trims the extreme values."""
    stacked_updates = np.stack(client_updates, axis=0) # Shape: (Num_Clients, Num_Parameters)
    stacked_updates.sort(axis=0)
    
    # Trim the top and bottom `num_malicious` values for every single parameter
    trimmed_updates = stacked_updates[num_malicious : -num_malicious, :]
    return np.mean(trimmed_updates, axis=0)

def aggregate_median(client_updates):
    """Computes the coordinate-wise median across all client vectors."""
    stacked_updates = np.stack(client_updates, axis=0)
    return np.median(stacked_updates, axis=0)