from sklearn.metrics import accuracy_score #Import accuracy_score from scikit-learn's metrics module

def compute_accuracy(predicted: str, reference: str) -> float:
    return accuracy_score([reference], [predicted])

# Function takes two string arguments: predicted and reference, returns a float value which is accuracy score between 0.0 and 1.0
# Accuracy_score expects lists (or arrays) of true and predicted labels so reference and predicted are wrapped in square brackets. 