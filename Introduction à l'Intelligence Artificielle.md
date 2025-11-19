# Introduction à l'Intelligence Artificielle

> _Cours magistral pour étudiants de 4ème année cycle ingénieur/Master 1, alternant théorie et pratique, conçu pour favoriser la compréhension et l’esprit critique._

***

## Table des matières

1. [Présentation et objectifs](#presentation-et-objectifs)
2. [Séance 1 : Histoire et concepts fondamentaux](#seance-1-histoire-concepts)
3. [Séance 2 : Apprentissage supervisé](#seance-2-apprentissage-supervise)
4. [Séance 3 : Apprentissage non supervisé](#seance-3-apprentissage-non-supervise)
5. [Séance 4 : Réseaux de neurones profonds](#seance-4-reseaux-profonds)
6. [Séance 5 : IA responsable](#seance-5-ia-responsable)
7. [Séance 6 : Projets et perspectives](#seance-6-projets-perspectives)
8. [Bibliographie sélective](#bibliographie)

***

```
## Présentation et objectifs<a name="presentation-et-objectifs"></a>
```

**Ce cours vise à :**

- Définir les concepts majeurs de l'intelligence artificielle (IA)
- Démystifier les méthodes d'apprentissage automatique
- Illustrer les applications concrètes et les enjeux industriels
- Initier à l'éthique et à l'explicabilité en IA
- Guider la réalisation de mini-projets en Python


### Prérequis

- Maîtrise de Python
- Bases de statistiques (moyenne, variance, corrélation)

***

```
# Séance 1 : Histoire et concepts fondamentaux<a name="seance-1-histoire-concepts"></a>
```

**Objectifs d'apprentissage :**

- Comprendre l'évolution et le vocabulaire central de l’IA
- Saisir les enjeux et typologies

***

### Définitions clés

- **Intelligence Artificielle (IA)** : Science et ensemble de techniques visant à faire réaliser à des machines des tâches considérées comme intelligentes (ex : compréhension du langage, reconnaissance d’image).
- **ANI (Artificial Narrow Intelligence)** : IA spécialisée dans une tâche précise (ex : jouer aux échecs, classifier emails).
- **AGI (Artificial General Intelligence)** : IA généraliste, capable de raisonner comme un humain sur divers sujets. AGI n’existe que sous forme de concept.


#### Typologie des IA

| Type | Définition | Exemple |
| :-- | :-- | :-- |
| ANI | Spécialisée, efficace pour une tâche. | Siri, traducteur automatique |
| AGI | Question conceptuelle, cognitive générale. | Non-atteint aujourd'hui |

### Bref historique

- **1956** : Terme

IA" introduit, conférence de Dartmouth

- **Années 1960-1970** : Premiers systèmes experts, découverte de l'apprentissage automatique
- **Années 1980** : Perceptron, début des réseaux de neurones
- **2000s-2010s** : Explosion du deep learning

**Étude de cas** : Evolution de la reconnaissance d’image, du simple filtrage à la détection d’objets multi-classes dans les smartphones.

### Enjeux sociétaux et perspectives

- Impacts sur l’emploi, la médecine, le transport
- Questions éthiques : confidentialité, biais, explicabilité

***

#### Quiz d’ouverture (à réaliser)

1. Qu’est-ce que l’ANI ?
2. En quelle année le terme IA a-t-il été proposé ?
3. Citez une application industrielle de l’IA.

***

#### Extrait Python : Visualiser le jeu de données MNIST

```python
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
mnist = fetch_openml('mnist_784', version=1)
plt.imshow(mnist.data[0].reshape(28, 28), cmap='gray')  # Affiche un chiffre manuscrit
plt.title(f"Chiffre: {mnist.target[0]}")
plt.show()
```

Ce code illustre une application emblématique du deep learning.

***

```
# Séance 2 : Apprentissage supervisé<a name="seance-2-apprentissage-supervise"></a>
```

**Objectifs d'apprentissage :**

- Saisir la différence régression/classification
- Manipuler des données tabulaires en Python
- Évaluer un algorithme avec des métriques adaptées

***

### Principes fondamentaux

- **Supervisé** : présence d'une variable cible ("label")


#### Exemples

- Prédire le prix d’un appartement (régression)
- Détecter des emails spam/non-spam (classification)


### Modélisation

- Régression linéaire, arbres de décision, k-plus proches voisins (KNN)
- **Métriques** : MSE (Mean Squared Error) en régression ; précision (accuracy), rappel (recall) en classification


#### Démonstration Python : Régression linéaire

```python
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
model = LinearRegression().fit(X, y)
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
```


#### Notion de biais/variance et validation croisée

- **Biais** : erreur due à la simplification excessive du modèle
- **Variance** : sensibilité aux variations des données
- **Validation croisée** : technique d’évaluation robuste, ex : K-fold


#### Étude de cas : Industrie bancaire

- Prédire le risque de crédit avec des arbres de décision et analyse des biais

***

#### Quiz

1. Quelle différence entre régression et classification ?
2. Citez une métrique de performance pour la classification.
3. Quel est le but de la validation croisée ?

***

```
# Séance 3 : Apprentissage non supervisé<a name="seance-3-apprentissage-non-supervise"></a>
```

**Objectifs d'apprentissage :**

- Détecter des groupes cachés ou des anomalies
- Appliquer clustering et réduction de dimension

***

### Clustering : regrouper sans label

- **Algorithmes**: K-means, DBSCAN, hiérarchique


#### Cas concret : Segmenter des clients selon leurs achats

#### Code Python simple : K-means

```python
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print("Labels:", kmeans.labels_)
```


### Réduction de dimensionnalité

- **PCA (Principal Component Analysis)** : compression informative


#### Code Python PCA

```python
from sklearn.decomposition import PCA
X = np.random.rand(100, 5)
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)
print(X_transformed.shape)  # (100,2)
```


### Détection d'anomalies

- IsoForest, LOF : repérer valeurs atypiques


#### Étude de cas : Détection de fraudes bancaires


***

#### Quiz

1. Qu’est-ce que le clustering ?
2. À quoi sert la réduction de dimensionnalité ?
3. Donnez une méthode de détection d’anomalies.

***

```
# Séance 4 : Réseaux de neurones profonds<a name="seance-4-reseaux-profonds"></a>
```

**Objectifs d'apprentissage :**

- Comprendre la structure des réseaux de neurones
- Manipuler les notions de couches, fonctions d’activation

***

### Perceptron simple

- Modélisation d’un neurone : somme pondérée suivie d’une activation


#### Illustration Python

```python
import numpy as np
def perceptron(X, y, lr=0.1, epochs=10):
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        for i in range(len(y)):
            y_pred = np.dot(X[i], w) > 0
            w += lr * (y[i] - y_pred) * X[i]
    return w
# Données fictives
X = np.array([[2, 3], [1, 5], [2, 8], [9, 6]])
y = np.array([0, 0, 0, 1])
w = perceptron(X, y)
print(w)
```


### Rétropropagation

- Calcul du gradient pour ajuster les paramètres


### Architectures avancées

- **CNN (Convolutional Neural Network)** : vision/computing image
- **RNN (Recurrent Neural Network)** : séquences (texte, séries temporelles)
- **Introduction aux transformers** : modèles séquentiels avancés (BERT, GPT)


#### Étude de cas

- Reconnaissance d’image médicale (CNN)
- Analyse de texte automatisée (RNN, transformers)

***

#### Quiz

1. Qu’est-ce qu’un perceptron ?
2. Donnez une différence entre CNN et RNN.
3. À quoi servent les transformers ?

***

```
# Séance 5 : IA responsable<a name="seance-5-ia-responsable"></a>
```

**Objectifs d'apprentissage :**

- Comprendre les enjeux éthiques et législatifs
- Manipuler les outils d’explicabilité

***

### Éthique et biais

- Sources de biais : données, algorithmes
- Limites des algorithmes : discrimination, généralisation
- Éthique : responsabilité, contrôle, respect vie privée


### Explicabilité en IA (XAI)

- SHAP, LIME : interpréter résultats


#### Exemple Python : Interprétation avec LIME

```python
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
X = pd.DataFrame({'A':[1,2,3],'B':[4,5,6]})
y = [0,1,0]
clf = RandomForestClassifier().fit(X, y)
explainer = LimeTabularExplainer(X.values, mode="classification")
exp = explainer.explain_instance(X.values[0], clf.predict_proba)
exp.show_in_notebook()
```


### Réglementation

- **AI Act** : cadre légal européen pour IA


#### Étude de cas : Reconnaissance faciale publique

- Défi éthique : consentement, surveillance

***

#### Quiz

1. Citez un type de biais rencontré en IA.
2. Quel est le rôle du AI Act ?
3. Donnez une méthode d’explicabilité.

***

```
# Séance 6 : Projets et perspectives<a name="seance-6-projets-perspectives"></a>
```

**Objectifs d'apprentissage :**

- Concevoir un projet IA de bout en bout
- S’initier au déploiement et à MLOps
- Identifier les avancées de l’IA générative

***

### Mini-projet Python

- Exemple : Détection d’anomalies de transactions bancaires


#### Étapes d’un projet

1. Définir le problème et valider les données
2. Sélectionner/entraîner le modèle
3. Évaluer et interpréter les résultats
4. Déployer (Docker, Streamlit, etc.)
5. Maintenir et monitorer (MLOps)

#### Code de base

```python
# Simple pipeline sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
clf = Pipeline([
    ("model", IsolationForest())
])
X = np.random.rand(100, 5)
clf.fit(X)
pred = clf.predict(X)
print(pred)
```


### IA générative

- GAN (Generative Adversarial Networks)
- Grandes avancées en images, textes (DALL-E, GPT)


#### Étude de cas : Génération d’images synthétiques

- Application : santé, publicité, jeux vidéo

***

#### Quiz

1. Quelles étapes essentielles dans un projet IA ?
2. Qu’est-ce qu’une IA générative ?

***

# Bibliographie sélective<a name="bibliographie"></a>

### Livres et articles fondateurs

- *Pattern Recognition and Machine Learning* — C. Bishop
- *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman
- *Deep Learning* — Goodfellow, Bengio, Courville
- Articles majeurs (perceptron, CNN, transformers)


### Ressources en ligne

- Coursera : "Machine Learning" — Andrew Ng
- GitHub "Awesome AI" : catalogue d’outils et projets Python
- OpenAI : blog, documentation technique

***

## Conseils pédagogiques et design

- **Déconstruire chaque concept** avec schémas (ex: workflow ML), analogies visuelles
- **Utiliser de vraies données** pour les TP
- **Favoriser les échanges critiques** : débats, questions ouvertes en quiz

> _À chaque séance : penser alternance théorie/pratique, enjeux réels, quiz interactif, et ouverture sur l’industrie/la recherche._

