from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


## Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

## Calculate sensitivity and specificity
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

## Calculate positive predictive value and negative predictive value
ppv = tp / (tp + fp)
npv = tn / (tn + fn)

## Calculate prevalence
prevalence = (tp + fn) / (tp + tn + fp + fn)

## Calculate detection rate and detection prevalence
detection_rate = tp / (tp + fn)
detection_prevalence = (tp + fp) / (tp + tn + fp + fn)

## Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

## Calculate balanced accuracy
b_accuracy = balanced_accuracy_score(y_true, y_pred)

## Calculate ROC AUC score
roc_auc = roc_auc_score(y_true, y_pred)

## Calculate precision, recall, f1
precision = precision_score(y_true_task, y_pred_task)
recall = recall_score(y_true_task, y_pred_task)
f1 = f1_score(y_true_task, y_pred_task)


