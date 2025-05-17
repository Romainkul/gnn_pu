import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

def train_binary_classifier_test(x, y_psd_neg, y_true, train_mask, val_mask, test_mask):
    x = normalize(x, norm='l2')

    x_train = x[train_mask]
    x_test = x[test_mask]
    x_val = x[val_mask]

    y_train = y_psd_neg[train_mask]  # We do not allow the leakage of the ground-true of negative nodes to the trainer
    y_val = y_psd_neg[val_mask]      # We do not allow the leakage of the ground-true of negative nodes to the evaluator
    y_test = y_true[test_mask]       # But we use the ground-true of negative nodes to test

    # train a linear binary classifier
    clf = LogisticRegression(penalty='l2')
    clf.fit(x_train, y_train)

    # evaluate
    f1_scores = []
    ap_scores = []
    y_pred = clf.predict_proba(x_val)
    y_pred = np.argmax(y_pred, axis=1)
    val_f1 = f1_score(y_val, y_pred, pos_label=1)
    val_ap = average_precision_score(y_val, y_pred)
    ap_scores.append(val_ap)
    f1_scores.append(val_f1)

    # test
    y_pred = clf.predict_proba(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    test_f1 = f1_score(y_test, y_pred, pos_label=1)
    test_ap = average_precision_score(y_test, y_pred)
    ap_scores.append(test_ap)
    f1_scores.append(test_f1)
    return f1_scores, ap_scores