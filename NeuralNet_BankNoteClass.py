import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def main():
	bank_note_df = pd.read_csv("./bank_note_data.csv")
	sns.countplot(x='Class',data=bank_note_df)
	plt.show()
	scalar = StandardScaler()
	scalar.fit(X=bank_note_df.drop('Class',axis=1))
	scaled_features = scalar.transform(X=bank_note_df.drop('Class',axis=1))
	scaled_df = pd.DataFrame(data = scaled_features,columns=bank_note_df.columns[:-1])
	X =scaled_df
	y=bank_note_df['Class']
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

	#a list of feature column objects
	feat_col = []
	for col in scaled_df.columns:
		feat_col.append(tf.feature_column.numeric_column(col))

	#object called classifier which is a DNNClassifier from learn with 2 classes and a [10,20,10] hidden unit layer structure
	classifier = tf.estimator.DNNClassifier(hidden_units=[10,20,10],n_classes=2,feature_columns=feat_col)
	input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,shuffle=True)
	#train classifier to the input function using steps=200
	classifier.train(input_fn=input_func,steps=200)
	#create prediction function and predict test data
	pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
	predictions = list(classifier.predict(input_fn=pred_fn))
	#create final prediction object with only classifier as 'predictions' object has many other values
	final_pred = []
	for pred in predictions:
		final_pred.append(pred['class_ids'][0])
	print_evaluation_report(y_test,final_pred,"DNNClassifier")

	#using random forest classifier and comparing with DNN Classifier
	rfc = RandomForestClassifier(n_estimators=200)
	rfc.fit(X_train,y_train)
	rfc_pred = rfc.predict(X_test)
	print_evaluation_report(y_test,rfc_pred,"Random Forest Classifier")

def print_evaluation_report(y_true,y_pred,title):
	print("-------------"+title+"-------------\n")
	print("Confusion_matrix:\n")
	print(confusion_matrix(y_true,y_pred))
	print('\nClassification report:\n')
	print(classification_report(y_true,y_pred))


if __name__ == '__main__':
	main()