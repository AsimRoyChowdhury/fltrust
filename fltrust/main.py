import flwr as fl
import pandas as pd
from fltrust.client_app import MaliciousFLClient
from fltrust.strategy import FLTrustStrategy
from fltrust.attacks import AttackSimulator
from fltrust.task import MNISTNet, prepare_datasets, evaluate_model

# Experiment Matrix Definitions
DEFENSES = ["FedAvg", "Krum", "Trim-mean", "Median", "FLTrust"]
ATTACKS = ["No attack", "LF attack", "Krum attack", "Trim attack", "Scaling attack", "Adaptive attack"]
NUM_CLIENTS = 100
MALICIOUS_FRACTION = 0.20

root_loader, client_loaders, test_loader, backdoor_test_loader = prepare_datasets(num_clients=NUM_CLIENTS)

def run_experiment(attack, defense):
    def client_fn(cid: str):
        client_id = int(cid)
        # Assign malicious status to the first 20 clients
        is_malicious = client_id < int(NUM_CLIENTS * MALICIOUS_FRACTION)
        simulator = AttackSimulator(attack_type=attack if is_malicious else "No attack")
        
        return MaliciousFLClient(
            dataloader=client_loaders[client_id], 
            attack_simulator=simulator
        ).to_client()

    strategy = FLTrustStrategy(
        defense_type=defense,
        global_model=MNISTNet(),
        root_loader=root_loader,
        fraction_fit=1,          
        min_fit_clients=100,         
        min_available_clients=100,
        fraction_evaluate=0.0
    )

    # Restrict concurrent clients to 3 to prevent OOM on 6GB VRAM constraint
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=2000), 
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.035},
    )

    # Calculate Evaluation Metrics
    final_model = strategy.global_model
    ter = evaluate_model(final_model, test_loader)
    
    # Only the Scaling Attack tracks the Attack Success Rate (ASR)
    if attack == "Scaling attack":
        asr = evaluate_model(final_model, backdoor_test_loader, metric="ASR")
        return f"{ter:.2f} / {asr:.2f}"
    
    return f"{ter:.2f}"

def generate_table():
    results = {defense: [] for defense in DEFENSES}
    checkpoint_file = "results.csv"
    
    # Create the CSV and write headers if it doesn't exist yet
    if not os.path.exists(checkpoint_file):
        with open(checkpoint_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Attack", "Defense", "Result"])

    for attack in ATTACKS:
        for defense in DEFENSES:
            print(f"Executing: {defense} under {attack}")
            result_metric = run_experiment(attack, defense)
            results[defense].append(result_metric)
            
            # Instantly append the result to the disk after the round completes
            with open(checkpoint_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([attack, defense, result_metric])
            print(f"--> Saved checkpoint: {defense} + {attack} = {result_metric}")
            
    # Once the entire matrix finishes normally, generate the clean Pandas outputs
    df = pd.DataFrame(results, index=ATTACKS)
    print("\n--- Results ---")
    print(df.to_markdown())
    df.to_csv("final_results.csv")

if __name__ == "__main__":
    generate_results()
