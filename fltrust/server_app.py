import flwr as fl
from fltrust.strategy import FLTrustStrategy
from fltrust.task import MNISTNet, prepare_datasets

# This file is a fallback for standard CLI execution. 

def server_fn(context: fl.common.Context):
    root_loader, _, _, _ = prepare_datasets(num_clients=100)
    
    # Defaults to FLTrust defense for standard execution
    strategy = FLTrustStrategy(
        defense_type="FLTrust",
        global_model=MNISTNet(),
        root_loader=root_loader,
        fraction_fit=1.0,
        min_fit_clients=100,
        min_available_clients=100
    )
    return fl.server.ServerAppComponents(
        strategy=strategy, 
        config=fl.server.ServerConfig(num_rounds=2000)
    )

app = fl.server.ServerApp(server_fn=server_fn)