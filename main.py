import pandas as pd
from data_processing import load_data, normalize_data, shuffle_data, split_data, preprocess_text_data
from svm_model import SVM
from svm_utils import save_model, load_model, calculate_accuracy, log_metrics

# Load and preprocess data
data = load_data('spam_ham_dataset.csv')

# Preprocess text data
text_column = 'text'
label_column = 'label_num'
X, y = preprocess_text_data(data, text_column, label_column)

# Shuffle and split the data
X, y = shuffle_data(pd.DataFrame(X)), pd.Series(y)
X_train, X_test, y_train, y_test = split_data(X, y)

# Normalize data
X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

# Initialize and train SVM model
svm = SVM()
svm.train(pd.DataFrame(X_train).assign(label=y_train.values))

# Predict and evaluate model
y_pred = svm.predict(X_test)
accuracy = calculate_accuracy(y_test.values, y_pred)
print(f'Accuracy: {accuracy}')

# Save the model
save_model(svm, 'svm_model.pkl')

# Log metrics
log_metrics(f'Accuracy: {accuracy}', 'metrics.log')
