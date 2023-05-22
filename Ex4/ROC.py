import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# load ground-truth values and detector outputs
y_true = np.loadtxt('detector_groundtruth.dat') # Ground truth values (1 for positive and 0 for negative)
y_scores = np.loadtxt('detector_output.dat') # Detector outputs

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Wikipedia: tp / (tp + fp)
precision = np.zeros_like(tpr)
# Wikipedia: tp / (tp + fn)
recall = np.zeros_like(fpr)

for i in range(len(thresholds)):
    threshold = thresholds[i]
    y_pred = y_scores >= threshold
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision[i] = tp / (tp + fp)
    recall[i] = tp / (tp + fn)

# Plot the ROC curve
plt.plot(recall, fpr)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (1-precision)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.show()