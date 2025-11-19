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






# Définitions Complètes des Termes de la Figure

---

## 🔷 DOMAINES PRINCIPAUX

### **Artificial Intelligence (Intelligence Artificielle)**
> Discipline scientifique visant à créer des systèmes capables d'effectuer des tâches nécessitant normalement l'intelligence humaine : raisonnement, apprentissage, perception, compréhension du langage et prise de décision.

### **Machine Learning (Apprentissage Automatique)**
> Sous-domaine de l'IA permettant aux systèmes d'apprendre à partir de données et d'améliorer leurs performances sans être explicitement programmés pour chaque tâche spécifique.

### **Neural Networks/Deep Learning (Réseaux de Neurones/Apprentissage Profond)**
> Technique de ML utilisant des réseaux de neurones artificiels à multiples couches pour apprendre des représentations hiérarchiques complexes des données.

### **Data Science (Science des Données)**
> Domaine interdisciplinaire combinant statistiques, informatique et expertise métier pour extraire des connaissances et insights à partir de données structurées et non structurées.

### **Big Data (Mégadonnées)**
> Ensembles de données extrêmement volumineux, complexes et variés qui nécessitent des technologies spécialisées pour leur stockage, traitement et analyse (caractérisés par les 5V : Volume, Vélocité, Variété, Véracité, Valeur).

---

## 📊 SUPERVISED LEARNING (Apprentissage Supervisé)

### **Classification/Regression**
> **Classification** : Tâche de prédiction d'une catégorie discrète (ex: spam/non-spam).  
> **Regression** : Tâche de prédiction d'une valeur continue (ex: prix immobilier).

### **Linear Regression (Régression Linéaire)**
> Modèle qui établit une relation linéaire entre variables d'entrée et sortie : y = β₀ + β₁x₁ + ... + βₙxₙ. Utilisé pour prédire des valeurs continues.

### **Logistic Regression (Régression Logistique)**
> Algorithme de classification binaire utilisant la fonction sigmoïde pour prédire des probabilités d'appartenance à une classe (0 ou 1).

### **Linear Neural Network (Réseau de Neurones Linéaire)**
> Réseau de neurones simple à une ou plusieurs couches sans fonctions d'activation non-linéaires, équivalent à une régression linéaire multiple.

### **Naive Bayes**
> Classificateur probabiliste basé sur le théorème de Bayes avec l'hypothèse "naïve" d'indépendance conditionnelle entre les features. Très efficace pour la classification de texte.

### **K-Nearest Neighbors (K Plus Proches Voisins)**
> Algorithme qui classe un point en fonction de la classe majoritaire de ses K voisins les plus proches dans l'espace des features. Non paramétrique et basé sur la distance.

### **Decision Trees (Arbres de Décision)**
> Structure arborescente où chaque nœud interne représente un test sur un attribut, chaque branche le résultat du test, et chaque feuille une décision finale (classe ou valeur).

### **Random Forest (Forêts Aléatoires)**
> Ensemble d'arbres de décision entraînés sur des sous-ensembles aléatoires des données (bagging). Les prédictions sont agrégées par vote majoritaire (classification) ou moyenne (régression).

### **Support Vector Machines (Machines à Vecteurs de Support)**
> Algorithme qui cherche l'hyperplan optimal séparant les classes avec la marge maximale. Utilise le "kernel trick" pour gérer les problèmes non-linéaires.

---

## 🔍 UNSUPERVISED LEARNING (Apprentissage Non-Supervisé)

### **Dimensionality Reduction (Réduction de Dimensionnalité)**
> Techniques visant à réduire le nombre de variables (features) tout en préservant l'information essentielle. Utilisé pour visualisation, compression et élimination du bruit.

### **PCA (Principal Component Analysis - Analyse en Composantes Principales)**
> Méthode de réduction de dimensionnalité qui transforme les données en un nouvel espace orthogonal où les axes (composantes principales) capturent la variance maximale.

### **Manifold Learning (Apprentissage de Variétés)**
> Ensemble de techniques (t-SNE, UMAP, Isomap) qui découvrent la structure géométrique sous-jacente de données haute dimension en les projetant sur une variété de dimension inférieure.

### **Clustering (Partitionnement)**
> Regroupement automatique de données similaires en clusters (groupes) sans labels préexistants. Utilisé pour segmentation, détection de patterns et organisation de données.

### **K-Means**
> Algorithme de clustering qui partitionne les données en K groupes en minimisant la variance intra-cluster. Assigne chaque point au centroïde le plus proche itérativement.

### **Hierarchical Clustering (Clustering Hiérarchique)**
> Méthode qui construit une hiérarchie de clusters sous forme d'arbre (dendrogramme) par agrégation successive (bottom-up) ou division (top-down) des groupes.

---

## 🧠 DEEP LEARNING ARCHITECTURES

### **Deep Neural Network (DNN - Réseau de Neurones Profond)**
> Réseau de neurones artificiels avec plusieurs couches cachées entre l'entrée et la sortie. Chaque couche apprend des représentations de plus en plus abstraites des données.

### **Convolutional Neural Network (CNN - Réseau de Neurones Convolutif)**
> Architecture spécialisée pour traiter des données structurées en grille (images). Utilise des convolutions pour détecter des features locales (contours, textures, objets) de manière hiérarchique.

### **Recurrent Neural Network (RNN - Réseau de Neurones Récurrent)**
> Architecture conçue pour traiter des séquences (texte, séries temporelles, audio) en maintenant une mémoire interne. Les connexions forment des cycles permettant de capturer des dépendances temporelles.

### **Autoencoder**
> Réseau de neurones non-supervisé composé d'un encodeur (compression) et d'un décodeur (reconstruction). Apprend des représentations compactes des données, utilisé pour réduction de dimensionnalité, débruitage et génération.

---

## 📈 VISUALISATION DU RÉSEAU DE NEURONES (Centre)

La figure centrale montre :

- **Couche d'entrée (jaune)** : Reçoit les données brutes (features)
- **Couches cachées (rouge/orange)** : Transformations non-linéaires successives, extraction de features hiérarchiques
- **Couche de sortie (bleu-vert)** : Prédiction finale (classe ou valeur)
- **Connexions** : Poids synaptiques ajustés durant l'apprentissage par rétropropagation

---

## 🔗 RELATIONS ENTRE LES DOMAINES

1. **IA ⊃ ML ⊃ DL** : Inclusion hiérarchique (du plus général au plus spécifique)
2. **Data Science ∩ AI** : La Data Science utilise les outils d'IA/ML pour analyser les données
3. **Big Data → ML** : Le Big Data fournit les données massives nécessaires pour entraîner les modèles de ML
4. **Supervised ∪ Unsupervised = ML** : Les deux paradigmes couvrent l'essentiel du Machine Learning classique
5. **Deep Learning ⊂ ML** : Le DL est une technique spécialisée du ML basée sur les réseaux de neurones profonds

---

## 📚 TABLEAU RÉCAPITULATIF

| Domaine | Type | Complexité | Cas d'usage typique |
|---------|------|------------|---------------------|
| Linear Regression | Supervisé | ⭐ | Prédiction de prix |
| Logistic Regression | Supervisé | ⭐⭐ | Classification binaire |
| Decision Trees | Supervisé | ⭐⭐ | Décisions explicables |
| Random Forest | Supervisé | ⭐⭐⭐ | Classification/Régression robuste |
| SVM | Supervisé | ⭐⭐⭐ | Classification haute dimension |
| K-Means | Non-supervisé | ⭐⭐ | Segmentation client |
| PCA | Non-supervisé | ⭐⭐ | Réduction de dimension |
| DNN | Deep Learning | ⭐⭐⭐⭐ | Classification complexe |
| CNN | Deep Learning | ⭐⭐⭐⭐ | Vision par ordinateur |
| RNN | Deep Learning | ⭐⭐⭐⭐ | Traitement de séquences |
| Autoencoder | Deep Learning | ⭐⭐⭐⭐ | Génération, compression |

---

Cette figure constitue une **carte conceptuelle complète** de l'écosystème moderne de l'Intelligence Artificielle et de la Data Science, montrant clairement les relations, hiérarchies et applications de chaque technique. 🎯

---

## 💡 NOTES COMPLÉMENTAIRES

### Quand utiliser quoi ?

- **Données étiquetées disponibles** → Supervised Learning
- **Pas de labels, recherche de patterns** → Unsupervised Learning
- **Images, vision** → CNN
- **Texte, séquences temporelles** → RNN/Transformers
- **Interprétabilité importante** → Decision Trees, Linear Models
- **Performance maximale** → Deep Learning, Random Forest
- **Peu de données** → Algorithmes simples (Linear, K-NN)
- **Beaucoup de données** → Deep Learning

### Évolution historique

1. **1950s-1980s** : IA symbolique, premiers algorithmes (Perceptron)
2. **1990s-2000s** : ML classique (SVM, Random Forest, PCA)
3. **2010s** : Révolution Deep Learning (AlexNet 2012)
4. **2020s** : Transformers, IA générative, LLMs

---

**Auteur** : Support de cours Data Science  
**Date** : 2025  
**Version** : 1.0
