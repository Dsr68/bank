
from sklearn import svm
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd



colunas = ['V', 'Ass', 'Curtose', 'Entropia', 'Verificacao']
notas = pd.read_csv("bank.csv", header=None, names=colunas)
notas.head()


columns = ['V', 'Ass', 'Curtose', 'Entropia']

X = notas[columns]
y = notas.Verificacao


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=13)

clf = svm.SVC(C=1.0)
clf.fit(X_train, y_train)




