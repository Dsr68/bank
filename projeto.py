# %%
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd



# %%
colunas = ['v', 'ass', 'curtose', 'entropia', 'class']
notas = pd.read_csv("bank.csv", header=None, names=colunas)
notas.head()

# %%
variaveis = ['v', 'ass', 'curtose', 'entropia']
X = notas.drop(columns=variaveis)
y = notas['class']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=13)


# %%
clf = svm.SVC(C=1.0)
clf.fit(X_train, y_train)

# %%
clf.score(X_test, y_test)
#clf.predict(X_test)
#y_test

# %%


# %% [markdown]
# 


