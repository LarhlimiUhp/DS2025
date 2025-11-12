<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# detialle plus ce support,

utilise des jeux de donnees sous forme de lien url pris sur internet

Voici une version **encore plus détaillée**, intégrant des jeux de données accessibles sur internet via des liens (datasets publics, Kaggle, UCI, etc.). Chaque séance inclut des cas d’application réelle, des exemples de code, et des références précises pour télécharger les données ou s’entraîner sur Colab/Jupyter. Le plan suit la progression proposée précédemment, optimisé pour une lecture sur 20 pages Markdown.

***

# Introduction à l’Intelligence Artificielle

*Support complet (20 pages) pour l’enseignement spécialisé*

***

## Table des matières

1. Présentation et objectifs du cours
2. Séance 1 : Histoire et concepts fondamentaux
3. Séance 2 : Apprentissage supervisé
4. Séance 3 : Apprentissage non supervisé
5. Séance 4 : Réseaux de neurones profonds
6. Séance 5 : IA responsable
7. Séance 6 : Projets et perspectives
8. Bibliographie sélective

***

## 1. Présentation et objectifs du cours

**Objectifs généraux :**

- Comprendre et utiliser les principaux concepts et techniques de l’IA
- Maîtriser l’alternance théorie/pratique sur datasets réels
- S’initier à la réflexion éthique et l’explicabilité

**Prérequis** : Bon niveau Python et statistiques de base

***

## 2. Séance 1 : Histoire et concepts fondamentaux

**Objectifs :**

- Découvrir l’évolution historique et les définitions clés

**Contenus :**

- Histoire de l’IA (Dartmouth, Perceptron, Deep Learning)
- ANI vs AGI
- Enjeux industriels/sociétaux

**Quiz d'ouverture :**

- Qu’est-ce que l’ANI ?
- Quel événement marque la naissance de l’IA ?

**Étude de cas : Reconnaissance d’image médicale vs. tri automatique de mails**

**Dataset :**

- [MNIST - Chiffres manuscrits](https://www.openml.org/d/554)
- [UCI Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- [Skin Cancer MNIST: HAM10000 (Kaggle)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

**Code Python : Visualisation MNIST**

```python
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
mnist = fetch_openml('mnist_784', version=1)
plt.imshow(mnist.data[0].reshape(28, 28), cmap='gray')
plt.title(f"Chiffre: {mnist.target[0]}")
plt.show()
```


***

## 3. Séance 2 : Apprentissage supervisé

**Objectifs :**

- Mettre en oeuvre régression et classification

**Théorie :**

- Variable cible (= label)
- Métriques (MSE, Accuracy, F1-score, ROC-AUC)

**Étude de cas : Prédiction du salaire à partir de compétences ; Détection d’e-mails spam**

**Datasets :**

- [Titanic (Kaggle)](https://www.kaggle.com/competitions/titanic/data)
- [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

**Code Python : Classification (Titanic)**

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df = df.dropna(subset=['Age','Embarked'])
X = df[['Pclass', 'Age', 'Fare']]
y = df['Survived']
clf = RandomForestClassifier().fit(X, y)
print("Score:", clf.score(X, y))
```

**Quiz fin de séance :**

- Différence régression/classification ?
- Décrivez la validation croisée.

***

## 4. Séance 3 : Apprentissage non supervisé

**Objectifs :**

- Utiliser les algorithmes de clustering et de réduction de dimension

**Théorie :**

- Regroupement de données (K-Means, DBSCAN)
- PCA pour simplifier la visualisation

**Étude de cas : Segmentation clients/Marketing ; Détection de fraudes bancaires**

**Datasets :**

- [Wholesale customers data (UCI)](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)
- [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Mall Customers Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)

**Code : Clustering K-means**

```python
import pandas as pd
from sklearn.cluster import KMeans

url = "https://raw.githubusercontent.com/sharmaroshan/Customer-Segmentation-Dataset/master/dataset.csv"
df = pd.read_csv(url)
X = df[['Age','Annual Income (k$)','Spending Score (1-100)']]
kmeans = KMeans(n_clusters=3).fit(X)
df['Cluster'] = kmeans.labels_
print(df.groupby('Cluster').mean())
```

**Quiz :**

- Qu’est-ce que le clustering ?
- Donnez une application industrielle.

***

## 5. Séance 4 : Réseaux de neurones profonds

**Objectifs :**

- Comprendre architectures CNN/RNN et rétropropagation

**Théorie :**

- Perceptron multicouche
- CNN (Vision), RNN (Séries), Introduction Transformers (NLP)

**Étude de cas : Reconnaissance de chiffres, Analyse de sentiment**

**Datasets :**

- [MNIST (OpenML)](https://www.openml.org/d/554)
- [IMDB Movie Review Sentiment (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)

**Code simple CNN (Keras)**

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
(X_train, y_train), (X_test, y_test) = mnist.load_data()
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    Flatten(), Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train.reshape(-1,28,28,1), y_train, epochs=3)
```

**Quiz :**

- Décrivez le principe du perceptron.
- Différence entre CNN et RNN ?

***

## 6. Séance 5 : IA responsable

**Objectifs :**

- Explorer la question de l’éthique, de la explicabilité et des biais

**Théorie :**

- XAI : SHAP, LIME
- Réglementation européenne (AI Act), principes de responsabilité

**Étude de cas : Reconnaissance faciale, Prêts bancaires**

**Datasets :**

- [German Credit Data](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- [COMPAS Recidivism Data](https://github.com/propublica/compas-analysis)

**Code explicabilité (LIME)**

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
df = pd.read_csv(url, delim_whitespace=True, header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
clf = RandomForestClassifier().fit(X, y)
explainer = LimeTabularExplainer(X.values, mode="classification")
exp = explainer.explain_instance(X.values[0], clf.predict_proba)
exp.show_in_notebook()
```

**Quiz :**

- Qu’est-ce qu’un biais algorithmique ? Exemple réel.

***

## 7. Séance 6 : Projets et perspectives

**Objectifs :**

- Concevoir un projet IA miniature, introduire le déploiement et les tendances IA

**Thèmes :**

- Pipeline du projet IA (acquisition, nettoyage, modélisation, évaluation, déploiement, MLOps)
- IA générative (GAN, diffusion, LLM)

**Datasets proposés :**

- [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [CelebA Faces (GAN training)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [OpenAI GPT-2 output dataset](https://github.com/openai/gpt-2-output-dataset)

**Code Pipeline Sklearn**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd

url = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download"
df = pd.read_csv(url)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', IsolationForest())
])
pipeline.fit(df.drop('Class', axis=1))
```

**Quiz final de cours**

- Quels risques en cas de mauvais déploiement IA ?
- Définir l’IA générative et donner une application concrète.

***

## 8. Bibliographie sélective

- *Pattern Recognition and Machine Learning*, C. Bishop
- *Deep Learning*, Goodfellow, Bengio, Courville
- [Scikit-learn documentation](https://scikit-learn.org/stable/index.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [OpenML](https://www.openml.org)
- [AI Act (EU)](https://artificialintelligenceact.eu/)

***

## Suggestions pour alternance théorie/pratique

- Pour chaque séance, débuter par la théorie, illustrer par un cas industriel, puis proposer une manipulation de dataset réel en Python.
- Mettre l’accent sur l’analyse critique des résultats et la réflexion sur les limites/biais des approches utilisées.

***

### _Ce support peut être enrichi selon les besoins, en générant des notebooks complets à partir des extraits et en développant les quiz pour une session interactive.

Prêt pour adaptation Beamer, PowerPoint, ou Google Slides grâce à la structure modulaire !_

