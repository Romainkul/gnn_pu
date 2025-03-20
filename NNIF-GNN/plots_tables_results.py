import numpy as np
import matplotlib.pyplot as plt
from train_NNIF_GNN import run_nnif_gnn_experiment

def experiment_varying_ratio_of_positives(ratio_list, fixed_params):
    """
    Sweeps over different ratios of positive samples,
    calls 'run_nnif_gnn_experiment', and plots the final metric vs. ratio.
    
    ratio_list   : list of floats (e.g. [0.1, 0.2, 0.3, 0.5])
    fixed_params : dict of other params that remain constant
    """
    results = []
    for ratio in ratio_list:
        # Merge ratio with the fixed parameters
        exp_params = {**fixed_params, 'ratio': ratio}
        
        # Call your training function
        metrics = run_nnif_gnn_experiment(exp_params)
        results.append((ratio, metrics['f1']))
        print(f"Ratio={ratio} => F1={metrics['f1']:.4f}")

    # Plot results
    ratios, accuracies = zip(*results)
    plt.figure()
    plt.plot(ratios, accuracies, marker='o')
    plt.xlabel("Ratio of Positives")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Ratio of Positives")
    plt.show()

def experiment_varying_k(k_values, pollution_ratios, fixed_params):
    """
    Sweeps over k_values and pollution_ratios to produce a 3D surface or heatmap
    that might show how 'K' and 'pollution ratio' jointly affect performance.
    
    k_values        : list of ints/floats
    pollution_ratios: list of floats
    fixed_params    : dict
    """
    metric_matrix = np.zeros((len(k_values), len(pollution_ratios)), dtype=float)

    for i, k_val in enumerate(k_values):
        for j, poll in enumerate(pollution_ratios):
            exp_params = {
                **fixed_params,
                'K': k_val,
                'pollution_ratio': poll
            }
            metrics = run_nnif_gnn_experiment(exp_params)
            metric_matrix[i, j] = metrics['accuracy']
            print(f"K={k_val}, pollution={poll:.2f} => Accuracy={metrics['accuracy']:.3f}")

    plt.figure()
    plt.imshow(metric_matrix, origin='lower', aspect='auto',
               extent=[0, len(pollution_ratios), 0, len(k_values)])
    plt.colorbar(label="Accuracy")
    plt.xlabel("Index in pollution_ratios")
    plt.ylabel("Index in k_values")
    plt.title("Accuracy for varying K and Pollution Ratio")
    plt.show()

###############################################################################
# 3) Experiment: Plot for varying class prior (compare nnPU, Imb nnPU, Ours)
###############################################################################
def experiment_varying_class_prior(class_priors, fixed_params):
    """
    Sweeps over a list of class priors, and for each prior we train different
    methods (nnPU, Imb nnPU, Ours), plotting or comparing results side by side.
    
    class_priors : list of floats
    fixed_params : dict
    """
    method_names = ['nnPU', 'Imb_nnPU', 'Ours']
    results = {name: [] for name in method_names}
    
    for prior in class_priors:
        for method in method_names:
            exp_params = {
                **fixed_params,
                'class_prior': prior,
                'method': method  # for example
            }
            metrics = run_nnif_gnn_experiment(exp_params)
            results[method].append(metrics['accuracy'])
            print(f"Prior={prior}, method={method} => Accuracy={metrics['accuracy']:.4f}")

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
        exp_params = {
            **fixed_params,
            'method': method,
            'variant': variant
        }
        metrics = run_nnif_gnn_experiment(exp_params)
        table_rows.append((method, variant, metrics['accuracy']))

    print("\n=== Table: Best Results (NNIF, IF, Spy) ===")
    print("{:>10} | {:>8} | {:>10}".format("Method", "Variant", "Accuracy"))
    print("-" * 35)
    for row in table_rows:
        print("{:>10} | {:>8} | {:>10.4f}".format(row[0], row[1], row[2]))

def table_best_results_sampling_strategies(fixed_params):
    """
    Loops over sampling strategies: ClusterGCN, SHINE, GraphSAGE
    Then prints the best results in a table.
    """
    strategies = [
        ('ClusterGCN', 'Cluster-Sampling'),
        ('SHINE',      'NN-Sampling'),
        ('GraphSAGE',  'Random-Sampling')
    ]

    table_rows = []
    for name, sampling_type in strategies:
        exp_params = {
            **fixed_params,
            'strategy': sampling_type,
            'model_name': name
        }
        metrics = run_nnif_gnn_experiment(exp_params)
        table_rows.append((name, sampling_type, metrics['accuracy']))

    print("\n=== Table: Sampling Strategies Comparison ===")
    print("{:>12} | {:>16} | {:>10}".format("Model", "SamplingType", "Accuracy"))
    print("-" * 44)
    for row in table_rows:
        print("{:>12} | {:>16} | {:>10.4f}".format(row[0], row[1], row[2]))

def table_best_results_convs(fixed_params):
    """
    Loops over 4 different GNN convolution layers (GATConv, GCNConv, GINConv,
    SAGEConv) and prints a table with the best results for each.
    """
    convs = ['GATConv', 'GCNConv', 'GINConv', 'SAGEConv']

    table_rows = []
    for conv_type in convs:
        exp_params = {
            **fixed_params,
            'conv_type': conv_type
        }
        metrics = run_nnif_gnn_experiment(exp_params)
        table_rows.append((conv_type, metrics['accuracy']))

    print("\n=== Table: Convolution Layers Comparison ===")
    print("{:>10} | {:>10}".format("ConvType", "Accuracy"))
    print("-" * 25)
    for row in table_rows:
        print("{:>10} | {:>10.4f}".format(row[0], row[1]))