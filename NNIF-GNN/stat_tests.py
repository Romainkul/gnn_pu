import numpy as np
from scipy import stats

def iman_davenport_test(
    num_models: int,
    num_datasets: int,
    model_ranks: np.ndarray
) -> float:
    """
    Performs the Iman-Davenport test (a more powerful alternative 
    to the Friedman test) to determine if there are significant 
    differences among multiple models across multiple datasets.

    Parameters
    ----------
    num_models : int
        Number of models being compared.
    num_datasets : int
        Number of datasets or experimental conditions.
    model_ranks : np.ndarray
        One-dimensional array of length `num_models` 
        containing the average rank of each model 
        (e.g., from a Friedman ranking procedure).

    Returns
    -------
    pvalue : float
        p-value from the F distribution under the Iman-Davenport test.
        If this p-value is below your chosen alpha, you may conclude 
        that at least one model differs significantly from the others.
    """
    # Compute the Iman-Davenport statistic (F-distribution)
    # Freedman statistic adjusted by Iman and Davenport:
    chisq_F = (
        12 * num_datasets / (num_models * (num_models + 1))
        * (np.sum(model_ranks**2) - (num_models * (num_models + 1)**2) / 4.0)
    )

    Ff = ((num_datasets - 1) * chisq_F) / (num_datasets * (num_models - 1) - chisq_F)

    df1 = num_models - 1
    df2 = (num_models - 1) * (num_datasets - 1)

    pvalue = 1.0 - stats.f.cdf(Ff, df1, df2)

    print(f"Iman-Davenport test: F({df1}, {df2}) = {Ff:.4f}, p-value = {pvalue:.4e}")
    return pvalue

def bonferroni_holm_test(
    num_models: int,
    num_datasets: int,
    model_names: np.ndarray,
    model_ranks: np.ndarray,
    alpha: float = 0.05
):
    """
    Performs the Bonferroni-Holm post-hoc test to identify which models
    differ significantly from the top-ranked (lowest-rank) model 
    after an omnibus test (e.g., Iman-Davenport) suggests that at least 
    one model is significantly different.

    Parameters
    ----------
    num_models : int
        Number of models compared.
    num_datasets : int
        Number of datasets.
    model_names : np.ndarray
        Array of model names of length `num_models`.
    model_ranks : np.ndarray
        Array of the average ranks of the models 
        (e.g., from Friedman or Iman-Davenport procedures).
    alpha : float, optional (default=0.05)
        Significance level.

    Returns
    -------
    best_model : str
        Name of the model with the lowest average rank (assumed "best").
    passed_models : np.ndarray
        Array of model names not significantly different from the best model
        after applying Bonferroni-Holm corrections.
    adj_p_values : np.ndarray
        Array of adjusted p-values for each comparison.
    raw_p_values : np.ndarray
        Array of unadjusted (raw) p-values for each comparison.
    
    Notes
    -----
    The test statistic is based on a normal approximation. The formula 
    for comparing model i with the best model is:

        z = (rank_i - min_rank) / sqrt( (num_models*(num_models+1)) / (6*num_datasets) ).

    We then compute one-sided p-values from z, and correct them using 
    the sequential Bonferroni-Holm method.
    """
    # Compute the standard deviation denominator for comparing model ranks
    denominator = np.sqrt((num_models * (num_models + 1)) / (6.0 * num_datasets))

    # Identify the top-ranked model(s)
    min_rank = np.min(model_ranks)
    ix_best = np.where(model_ranks == min_rank)[0]
    best_model = model_names[ix_best][0]  # If tie, pick the first

    # Remove the best model from further comparisons
    # We'll compare all other models against this best model
    ranks_others = np.delete(model_ranks, ix_best)
    names_others = np.delete(model_names, ix_best)

    # Calculate z-scores for each other model vs. best model
    z_scores = (ranks_others - min_rank) / denominator
    # One-sided p-value from normal distribution survival function:
    raw_p_values = stats.norm.sf(z_scores)  

    # Sort p-values ascendingly for the Holm procedure
    ix_sort = np.argsort(raw_p_values)
    sorted_p_values = raw_p_values[ix_sort]

    # Bonferroni-Holm decisions
    decision = np.ones_like(sorted_p_values, dtype=bool)
    adj_p_values = sorted_p_values.copy()

    # Start iterative Holm procedure
    # i: index in the sorted array; m: actual p-value
    for i, p_val in enumerate(sorted_p_values):
        # Since there are (num_models - 1) comparisons in total:
        # We do alpha / (num_models - i - 1)
        if p_val <= alpha / (num_models - 1 - i):
            # We reject the null -> difference is significant
            decision[ix_sort[i]] = False
        
        # Adjusted p-values for reporting (Holm's multiplication)
        adj_p_values[ix_sort[i]] *= (num_models - 1 - i)

    # decision == True -> we fail to reject the null (i.e., not significantly different)
    # decision == False -> we reject the null (significantly different)
    passed_models = names_others[decision]  # models that are effectively "as good" as best

    # Return:
    #  - best_model: the "control" (lowest rank)
    #  - passed_models: those not significantly different from the best model
    #  - adj_p_values: final Holm-corrected p-values
    #  - raw_p_values: original p-values
    print(f"\nBonferroni-Holm with alpha={alpha}:")
    print(f"Best model: {best_model}")
    print(f"Models not significantly different from {best_model}: {passed_models.tolist()}\n")

    return best_model, passed_models, adj_p_values, raw_p_values
