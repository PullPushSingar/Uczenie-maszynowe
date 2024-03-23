def MSE(targets, predictions):
    return ((targets - predictions) ** 2).mean()

def classification_error(targets, predictions):
    y_pred = np.where(predictions > 0.5, 1, 0)
    return (targets != y_pred).mean()