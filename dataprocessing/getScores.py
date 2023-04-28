from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
def main(y_test, y_pred):

    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')

    # Print the metrics in two columns
    print('Metric\t\tValue')
    print('------------------')
    print('accuracy\t{:.4f}'.format(accuracy))
    print('f1-score\t{:.4f}'.format(f1))
    print('recall\t\t{:.4f}'.format(recall))
    print('precision\t{:.4f}'.format(precision))

#how to add this:
    #import sys
    #sys.path.append('./dataprocessing')
    #import getScores

    #getScores.main(y_test,y_pred)