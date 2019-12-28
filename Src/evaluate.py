import keras
import itertools
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def print_confusion_matrix(model, generator, model_name="model"):
	data = [x for x in itertools.islice(generator, 1)]
	data = data[0]


	X, Y = data

	y_pred = model.predict(X)
	y_pred = np.round(y_pred)

	print("Confusion Matrix for "+model_name+" is:")
	print(confusion_matrix(Y, y_pred))

	print("Classification Report for "+model_name+" is:")
	print(classification_report(Y, y_pred))
