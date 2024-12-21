"To begin the script in the command line put python shopping.py shopping.csv"
import csv
import sys
import calendar
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score

TEST_SIZE = 0.2

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # find better hyperparametre for k-neighbors
    K = find_hyper(X_train, y_train)

    # Train model and make predictions
    model = train_model(X_train, y_train, K)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []
    month_mapping = {
        **{month: idx for idx, month in enumerate(calendar.month_abbr) if month},
        **{month: idx for idx, month in enumerate(calendar.month_name) if month},
    }

    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                month = month_mapping.get(row["Month"], -1)
                if month == -1:
                    print(f"Skipping invalid month: {row['Month']}")
                    continue

                evidence.append([
                    int(row["Administrative"]),
                    float(row["Administrative_Duration"]),
                    int(row["Informational"]),
                    float(row["Informational_Duration"]),
                    int(row["ProductRelated"]),
                    float(row["ProductRelated_Duration"]),
                    float(row["BounceRates"]),
                    float(row["ExitRates"]),
                    float(row["PageValues"]),
                    float(row["SpecialDay"]),
                    month,
                    int(row["OperatingSystems"]),
                    int(row["Browser"]),
                    int(row["Region"]),
                    int(row["TrafficType"]),
                    1 if row["VisitorType"] == "Returning_Visitor" else 0,
                    1 if row["Weekend"] == "TRUE" else 0,
                ])
                labels.append(1 if row["Revenue"] == "TRUE" else 0)
            except (ValueError, KeyError) as e:
                print(f"Skipping invalid row: {e}")
    return evidence, labels

def train_model(evidence, labels, K):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=K)
    model.fit(evidence, labels)
    return model

def find_hyper(X_train, y_train):

    model = KNeighborsClassifier()
    param_grid = {'n_neighbors': range(2, 21)}

    scoring = make_scorer(recall_score)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring= scoring,
        cv=5
    )

    grid.fit(X_train, y_train)

    print("best K:", grid.best_params_['n_neighbors'])

    print("Precision:", grid.best_score_)

    return grid.best_params_['n_neighbors']


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positive = sum(1 for actual, predicted in zip(labels, predictions) if actual == 1 and predicted == 1)
    true_negative = sum(1 for actual, predicted in zip(labels, predictions) if actual == 0 and predicted == 0)
    total_positive = sum(1 for label in labels if label == 1)
    total_negative = sum(1 for label in labels if label == 0)

    sensitivity = true_positive / total_positive if total_positive else 0
    specificity = true_negative / total_negative if total_negative else 0

    return sensitivity, specificity


if __name__ == "__main__":
    main()
