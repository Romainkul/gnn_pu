import numpy as np
import matplotlib.pyplot as plt
from train_NNIF_GNN import run_nnif_gnn_experiment

def experiment_varying_ratio_of_positives(fixed_params):
    results = []
    ratio_list=[0.5,0.4,0.3,0.2]
    
    est_prior=fixed_params['ratio']/((1 - 0.5) + (fixed_params['ratio'] * 0.5))
    
    for train_pct in ratio_list:
        ratio=(est_prior-train_pct*est_prior)/(1-train_pct*est_prior)    
        exp_params = {**fixed_params, 'train_pct': train_pct, 'ratio': ratio}
        
        # Call your training function
        f1,std = run_nnif_gnn_experiment(exp_params)
        results.append((train_pct, f1))
        print(f"Ratio of Positives={train_pct} => F1={f1:.4f}")

    # Plot results
    ratios, f1 = zip(*results)
    plt.figure()
    plt.plot(ratios, f1, marker='o')
    plt.xlabel("Ratio of Positives")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Ratio of Positives")
    plt.show()


def experiment_varying_k(k_values, pollution_ratios, fixed_params):
    metric_matrix = np.zeros((len(k_values), len(pollution_ratios)), dtype=float)

    for i, k_val in enumerate(k_values):
        for j, poll in enumerate(pollution_ratios):
            exp_params = {
                **fixed_params,
                'K': k_val,
                'ratio': poll
            }
            f1,std = run_nnif_gnn_experiment(exp_params)
            metric_matrix[i, j] = f1
            print(f"K={k_val}, pollution={poll:.2f} => F1={f1:.3f}")

    plt.figure()
    plt.imshow(metric_matrix, origin='lower', aspect='auto',
               extent=[0, len(pollution_ratios), 0, len(k_values)])
    plt.colorbar(label="F1 Score")
    plt.xlabel("Index in pollution_ratios")
    plt.ylabel("Index in k_values")
    plt.title("Accuracy for varying K and Pollution Ratio")
    plt.show()

def experiment_varying_class_prior(class_priors, fixed_params):
    """
    Sweeps over a list of class priors, and for each prior we train different
    methods (nnPU, Imb nnPU, Ours), plotting or comparing results side by side.
    
    class_priors : list of floats
    fixed_params : dict
    """
    method_names = ['nnPU', 'Imb_nnPU', 'ours']
    results = {name: [] for name in method_names}
    
    for prior in class_priors:
        for method in method_names:
            if method=="ours":
                ratio=(prior-fixed_params['train_pct']*prior)/(1-fixed_params['train_pct']*prior)
            exp_params = {
                **fixed_params,
                'ratio': prior,
                'methodology': method
            }
            f1,std = run_nnif_gnn_experiment(exp_params)
            results[method].append(f1)
            print(f"Prior={prior}, method={method} => f1={f1:.4f}")

    plt.figure()
    for method in method_names:
        plt.plot(class_priors, results[method], label=method, marker='o')
    plt.xlabel("Class Prior")
    plt.ylabel("Accuracy")
    plt.title("Comparison of Methods vs. Class Prior")
    plt.legend()
    plt.show()

def table_best_results_nnif_if_spy(fixed_params):
    """
    Loops over these methods:
      - NNIF (removal, relabel)
      - IF (removal, relabel)
      - Spy (SCAR)
    Captures the best metrics for each approach, then prints a table.
    """
    combos = [
        ('NNIF', 'removal'),
        ('NNIF', 'relabel'),
        ('IF',   'removal'),
        ('IF',   'relabel'),
        ('Spy',  'Spy')
    ]

    table_rows = []
    for method, variant in combos:
        if method=="Spy":
            exp_params = {
                **fixed_params,
                'method': method
            }
        else:
            exp_params = {
                **fixed_params,
                'method': method,
                'treatment': variant
            }
        f1,std = run_nnif_gnn_experiment(exp_params)
        table_rows.append((method, variant, f1))

    print("\n=== Table: Best Results (NNIF, IF, Spy) ===")
    print("{:>10} | {:>8} | {:>10}".format("Method", "Variant", "F1","STD"))
    print("-" * 35)
    for row in table_rows:
        print("{:>10} | {:>8} | {:>10.4f}".format(row[0], row[1], row[2]))

def table_best_results_sampling_strategies(fixed_params):
    """
    Loops over sampling strategies: ClusterGCN, SHINE, GraphSAGE
    Then prints the best results in a table.
    """
    strategies = ['cluster','neighbor','nearest_neighbor']

    table_rows = []
    for sampling_type in strategies:
        exp_params = {
            **fixed_params,
            'sampling': sampling_type
        }
        f1,std = run_nnif_gnn_experiment(exp_params)
        table_rows.append((sampling_type, f1,std))

    print("\n=== Table: Sampling Strategies Comparison ===")
    print("{:>12} | {:>16} | {:>10}".format("SamplingType", "F1","STD"))
    print("-" * 44)
    for row in table_rows:
        print("{:>12} | {:>16} | {:>10.4f}".format(row[0], row[1], row[2]))

def table_best_results_convs(fixed_params):
    """
    Loops over 4 different GNN convolution layers (GATConv, GCNConv, GINConv,
    SAGEConv) and prints a table with the best results for each.
    """
    convs = ['GATConv', 'GCNConv', 'GINConv', 'SAGEConv','MLP']

    table_rows = []
    for conv_type in convs:
        exp_params = {
            **fixed_params,
            'model_type': conv_type
        }
        f1,std = run_nnif_gnn_experiment(exp_params)
        table_rows.append((conv_type, f1,std))

    print("\n=== Table: Convolution Layers Comparison ===")
    print("{:>10} | {:>10}".format("ConvType", "F1","STD"))
    print("-" * 25)
    for row in table_rows:
        print("{:>10} | {:>10.4f}".format(row[0], row[1]))