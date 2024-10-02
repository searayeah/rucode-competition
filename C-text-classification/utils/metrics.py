from sklearn.metrics import f1_score


def calculate_f1_scores_multilabel(y_true, y_pred, num_labels, threshold):
    return [f1_score(y_true[:, i], y_pred[:, i] >= threshold) for i in range(num_labels)]


def calculate_f1_score_multilabel(y_true, y_pred, threshold, average):
    return f1_score(y_true, y_pred >= threshold, average=average)


def f1_df_for_each_threshold(y_true, y_pred, num_labels):
    scores = {}
    for threshold in [x / 10 for x in range(11)]:
        scores[threshold] = calculate_f1_scores_multilabel(
            y_true, y_pred, num_labels, threshold
        )
        scores[threshold].append(
            calculate_f1_score_multilabel(y_true, y_pred, threshold, "macro")
        )
    return scores
