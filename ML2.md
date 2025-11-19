# SCIENCE DES DONNÉES
## Cours Avancé - Niveau Master 1

**Destiné aux étudiants de 4ᵉ année**  
**Volume horaire estimé : 60h (cours + TP)**  
**Document de référence : Version complète**

---

## TABLE DES MATIÈRES

1. [Introduction à l'Intelligence Artificielle](#1-introduction-à-lintelligence-artificielle)
2. [Fondements du Machine Learning](#2-fondements-du-machine-learning)
3. [Deep Learning et Réseaux de Neurones](#3-deep-learning-et-réseaux-de-neurones)
4. [Vision par Ordinateur](#4-vision-par-ordinateur)
5. [Traitement du Langage Naturel](#5-traitement-du-langage-naturel)
6. [Analyse Prédictive](#6-analyse-prédictive)
7. [Bonnes Pratiques et Tendances](#7-bonnes-pratiques-et-tendances)
8. [Exercices et Mini-Projets](#8-exercices-et-mini-projets)
9. [Bibliographie et Ressources](#9-bibliographie-et-ressources)
10. [Annexes](#annexes)

---

## 1. INTRODUCTION À L'INTELLIGENCE ARTIFICIELLE

### 1.1 Historique de l'IA

#### 1.1.1 Les Prémices (1943-1956)

L'intelligence artificielle trouve ses racines dans les travaux précurseurs de plusieurs disciplines :

- **1943** : Warren McCulloch et Walter Pitts développent le premier modèle mathématique de neurone artificiel, posant les bases théoriques des réseaux de neurones.
- **1950** : Alan Turing propose le "Test de Turing" dans son article fondateur "Computing Machinery and Intelligence", questionnant la capacité des machines à "penser".
- **1956** : Conférence de Dartmouth - John McCarthy forge le terme "Intelligence Artificielle", marquant officiellement la naissance du domaine.

**[SCHÉMA 1.1 : Frise chronologique de l'évolution de l'IA (1943-2025) avec événements clés]**

#### 1.1.2 Les Périodes Clés

**L'âge d'or (1956-1974)**
- Développement des premiers programmes de raisonnement symbolique (Logic Theorist, General Problem Solver)
- Création de LISP par John McCarthy (1958), langage phare de l'IA symbolique
- Premiers succès : résolution de problèmes mathématiques, jeux d'échecs basiques
- Optimisme débordant : prédictions de machines intelligentes dans les 20 ans

**Le premier hiver de l'IA (1974-1980)**
- Limitations computationnelles majeures : manque de puissance de calcul et de mémoire
- Échec des promesses optimistes (traduction automatique, reconnaissance vocale)
- Rapport Lighthill (1973) critique sévèrement les progrès de l'IA
- Réduction drastique des financements publics et privés

**L'essor des systèmes experts (1980-1987)**
- MYCIN, DENDRAL : premiers systèmes experts médicaux performants
- Investissements massifs au Japon (projet 5ᵉ génération d'ordinateurs)
- Boom commercial des systèmes experts en entreprise
- Limitation : connaissance codée manuellement, manque de généralisation

**Le second hiver (1987-1993)**
- Effondrement du marché des machines LISP spécialisées
- Limites intrinsèques des systèmes experts révélées
- Désillusion suite aux promesses non tenues
- Passage aux approches connexionnistes et probabilistes

**Renaissance moderne (1993-2011)**
- Approches statistiques et probabilistes (réseaux bayésiens, SVMs)
- Augmentation exponentielle de la puissance de calcul
- IBM Deep Blue bat Garry Kasparov aux échecs (1997)
- Émergence du Machine Learning comme paradigme dominant

**L'ère du Deep Learning (2012-présent)**
- **2012** : AlexNet révolutionne la vision par ordinateur (ImageNet challenge)
- **2016** : AlphaGo de DeepMind bat Lee Sedol au jeu de Go
- **2017** : Introduction des Transformers ("Attention Is All You Need")
- **2020** : GPT-3 démontre des capacités étonnantes de génération de texte
- **2022-2025** : Explosion de l'IA générative (ChatGPT, DALL-E, Stable Diffusion)

### 1.2 Concepts Fondamentaux

#### 1.2.1 Définitions

**Intelligence Artificielle (IA)**
> Discipline scientifique visant à créer des systèmes capables d'effectuer des tâches nécessitant normalement l'intelligence humaine : raisonnement logique, apprentissage automatique, perception sensorielle, compréhension du langage naturel, résolution de problèmes complexes.

**Machine Learning (Apprentissage Automatique)**
> Sous-domaine de l'IA permettant aux systèmes d'apprendre à partir de données sans être explicitement programmés pour chaque tâche. Le système améliore ses performances avec l'expérience.

**Deep Learning (Apprentissage Profond)**
> Branche du Machine Learning utilisant des réseaux de neurones artificiels à multiples couches (profonds) pour apprendre des représentations hiérarchiques des données, du plus simple (contours) au plus complexe (objets entiers).

**Data Science (Science des Données)**
> Domaine interdisciplinaire combinant statistiques, informatique, et expertise métier pour extraire des connaissances et insights à partir de données structurées et non structurées.

**[SCHÉMA 1.2 : Diagramme de Venn montrant les relations entre IA, ML, DL et Data Science]**

#### 1.2.2 Types d'Intelligence Artificielle

**IA Faible (Narrow AI/ANI)**
- Spécialisée dans une tâche spécifique unique
- Tous les systèmes actuels appartiennent à cette catégorie
- Exemples : reconnaissance vocale (Siri), recommandations (Netflix), diagnostic médical
- Performance souvent supérieure à l'humain dans son domaine restreint
- Incapable de généraliser hors de son contexte d'entraînement

**IA Forte (General AI/AGI)**
- Capacités cognitives équivalentes à celles d'un humain
- Adaptabilité à diverses tâches sans réentraînement spécifique
- Compréhension contextuelle et raisonnement abstrait
- Objectif théorique non atteint à ce jour
- Estimations variables : 2030-2050 selon les experts

**Super-Intelligence (ASI)**
- Intelligence largement supérieure à celle de l'humain dans tous les domaines
- Concept hypothétique et spéculatif
- Débats éthiques et existentiels (Nick Bostrom, "Superintelligence", 2014)
- Questions de contrôle et d'alignement des objectifs

#### 1.2.3 Paradigmes d'IA

**IA Symbolique (GOFAI - Good Old-Fashioned AI)**
- Manipulation de symboles et règles logiques
- Représentation explicite de la connaissance
- Avantages : interprétabilité, raisonnement explicite
- Inconvénients : rigidité, difficulté à gérer l'incertitude

**IA Connexionniste (Réseaux de Neurones)**
- Inspiration du fonctionnement du cerveau humain
- Apprentissage par ajustement de poids synaptiques
- Avantages : apprentissage automatique, robustesse au bruit
- Inconvénients : boîte noire, nécessite beaucoup de données

**IA Hybride**
- Combinaison des approches symboliques et connexionnistes
- Exemple : systèmes neuro-symboliques
- Objectif : combiner raisonnement explicite et apprentissage

### 1.3 Applications Actuelles

#### 1.3.1 Secteur Santé
- **Diagnostic médical assisté** : Détection précoce de cancers (mammographie, radiologie), analyse d'IRM et scanners
- **Découverte de médicaments** : AlphaFold pour la prédiction de structures protéiques, accélération du screening de molécules
- **Chirurgie assistée** : Robots chirurgicaux (Da Vinci), planification d'interventions
- **Médecine personnalisée** : Traitement adapté au profil génétique du patient
- **Surveillance des patients** : Détection précoce de détérioration, prédiction de complications

#### 1.3.2 Transport et Mobilité
- **Véhicules autonomes** : Tesla Autopilot, Waymo, Cruise (niveau 2-4 d'autonomie)
- **Optimisation de trafic** : Gestion intelligente des feux, prédiction de congestion
- **Logistique** : Optimisation de routes de livraison, gestion d'entrepôts automatisés
- **Maintenance prédictive** : Prévention de pannes sur véhicules et infrastructures
- **Transport aérien** : Optimisation de trajectoires de vol, pilotage automatique

#### 1.3.3 Finance
- **Détection de fraudes** : Analyse en temps réel de transactions suspectes
- **Trading algorithmique** : High-frequency trading, stratégies quantitatives
- **Évaluation de risques** : Scoring de crédit, analyse de portefeuilles
- **Robo-advisors** : Conseil en investissement automatisé
- **Conformité réglementaire** : Détection de blanchiment d'argent (AML/KYC)

#### 1.3.4 Industrie et Manufacturing
- **Contrôle qualité automatisé** : Vision par ordinateur pour détection de défauts
- **Robotique industrielle intelligente** : Cobots adaptatifs, préhension flexible
- **Jumeaux numériques** : Simulation et optimisation de processus de production
- **Maintenance prédictive** : Prévention de pannes machines, réduction des temps d'arrêt
- **Optimisation énergétique** : Gestion intelligente de la consommation

#### 1.3.5 Services et Commerce
- **Systèmes de recommandation** : Netflix (films), Amazon (produits), Spotify (musique)
- **Assistants virtuels** : Alexa, Siri, Google Assistant
- **Chatbots** : Support client automatisé, disponibilité 24/7
- **Analyse de sentiment** : Écoute des réseaux sociaux, satisfaction client
- **Personnalisation** : Expérience utilisateur sur mesure, pricing dynamique

#### 1.3.6 Éducation
- **Tuteurs intelligents** : Adaptation au niveau de l'élève
- **Évaluation automatisée** : Correction de copies, feedback personnalisé
- **Détection de plagiat** : Analyse de similarité de textes
- **Prédiction de décrochage** : Intervention précoce auprès d'étudiants en difficulté

---

## 2. FONDEMENTS DU MACHINE LEARNING

### 2.1 Paradigmes d'Apprentissage

#### 2.1.1 Apprentissage Supervisé

**Principe Fondamental**

L'algorithme apprend à partir d'un ensemble de données étiquetées (X, y), où X représente les features (variables d'entrée) et y les labels (sorties attendues). L'objectif est d'apprendre une fonction de mapping f telle que :

$$
f(X) \approx y
$$

**Processus d'apprentissage**
1. **Initialisation** : Paramètres aléatoires du modèle
2. **Prédiction** : Application du modèle sur les données d'entraînement
3. **Calcul de l'erreur** : Comparaison prédictions vs labels réels
4. **Optimisation** : Ajustement des paramètres pour réduire l'erreur
5. **Itération** : Répétition jusqu'à convergence

**Types de problèmes**

**Classification**
- **Binaire** : Deux classes (spam/non-spam, malade/sain)
- **Multi-classes** : Plus de deux classes mutuellement exclusives (reconnaissance de chiffres 0-9)
- **Multi-labels** : Plusieurs labels possibles simultanément (tags d'articles)

**Régression**
- Prédiction de valeurs continues (prix immobilier, température, ventes)
- Relation fonctionnelle entre variables d'entrée et sortie

**[SCHÉMA 2.1 : Pipeline complet d'apprentissage supervisé avec train/test split et validation]**

**Exemple Complet : Régression Linéaire**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Génération de données synthétiques avec bruit
np.random.seed(42)
n_samples = 100
X = 2 * np.random.rand(n_samples, 1)  # Feature unique entre 0 et 2
# Relation linéaire : y = 4 + 3x + bruit gaussien
y = 4 + 3 * X + np.random.randn(n_samples, 1)

# Division des données : 80% entraînement, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Création et entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation des performances
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("RÉSULTATS DE LA RÉGRESSION LINÉAIRE")
print("=" * 50)
print(f"Coefficient (pente) : {model.coef_[0][0]:.3f} (attendu: 3.0)")
print(f"Ordonnée à l'origine : {model.intercept_[0]:.3f} (attendu: 4.0)")
print(f"MSE : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"R² Score : {r2:.3f}")
print("=" * 50)

# Visualisation complète
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Données et prédictions
axes[0].scatter(X_train, y_train, color='blue', alpha=0.6, label='Train')
axes[0].scatter(X_test, y_test, color='green', alpha=0.6, label='Test')
axes[0].plot(X_test, y_pred, color='red', linewidth=2, label='Prédictions')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].set_title('Régression Linéaire')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Résidus
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Valeurs prédites')
axes[1].set_ylabel('Résidus')
axes[1].set_title('Analyse des Résidus')
axes[1].grid(True, alpha=0.3)

# 3. Prédictions vs Réalité
axes[2].scatter(y_test, y_pred, alpha=0.6)
axes[2].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Prédiction parfaite')
axes[2].set_xlabel('Valeurs réelles')
axes[2].set_ylabel('Valeurs prédites')
axes[2].set_title(f'Prédictions vs Réalité (R²={r2:.3f})')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Explication mathématique détaillée**

La régression linéaire cherche à minimiser la fonction de coût des moindres carrés :

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
$$

Où :
- $m$ : nombre d'échantillons d'entraînement
- $h_\theta(x) = \theta_0 + \theta_1 x$ : hypothèse linéaire (fonction de prédiction)
- $\theta = [\theta_0, \theta_1]$ : paramètres à optimiser (ordonnée, pente)

**Solution analytique (Équation Normale)** :

$$
\theta = (X^TX)^{-1}X^Ty
$$

**Solution itérative (Descente de gradient)** :

$$
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
$$

Avec $\alpha$ le taux d'apprentissage (learning rate).

#### 2.1.2 Apprentissage Non-Supervisé

**Principe Fondamental**

L'algorithme cherche à découvrir des structures cachées, des patterns ou des relations dans des données **non étiquetées**. Pas de labels de référence, le modèle doit auto-organiser les données.

**Applications principales**

1. **Clustering (Partitionnement)**
   - Regroupement d'objets similaires
   - Segmentation de clientèle, compression d'images
   - Algorithmes : K-means, DBSCAN, Hierarchical Clustering

2. **Réduction de dimensionnalité**
   - Compression de données en préservant l'information essentielle
   - Visualisation de données haute dimension
   - Algorithmes : PCA, t-SNE, UMAP, Autoencoders

3. **Détection d'anomalies**
   - Identification d'observations atypiques
   - Fraude, défauts de fabrication, intrusions réseau
   - Algorithmes : Isolation Forest, One-Class SVM, LOF

4. **Apprentissage de représentations**
   - Extraction automatique de features pertinentes
   - Word embeddings, feature learning
   - Algorithmes : Word2Vec, Autoencoders

**[SCHÉMA 2.2 : Comparaison visuelle apprentissage supervisé vs non-supervisé avec exemples]**

**Exemple Détaillé : K-Means Clustering**

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np

# Génération de données synthétiques avec 4 clusters naturels
X, y_true = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=0.60,
    random_state=42
)

# Méthode du coude pour déterminer k optimal
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Visualisation de la méthode du coude
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Nombre de clusters (k)')
axes[0].set_ylabel('Inertie (somme des distances²)')
axes[0].set_title('Méthode du Coude')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Nombre de clusters (k)')
axes[1].set_ylabel('Score de Silhouette')
axes[1].set_title('Score de Silhouette vs k')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Application de K-Means avec k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Métriques d'évaluation
silhouette_avg = silhouette_score(X, y_kmeans)
davies_bouldin = davies_bouldin_score(X, y_kmeans)

print("=" * 50)
print("RÉSULTATS DU CLUSTERING K-MEANS")
print("=" * 50)
print(f"Nombre de clusters : 4")
print(f"Inertie : {kmeans.inertia_:.2f}")
print(f"Score de Silhouette : {silhouette_avg:.3f}")
print(f"Davies-Bouldin Index : {davies_bouldin:.3f}")
print("=" * 50)

# Visualisation comparative
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Vraies classes
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', 
                s=50, edgecolors='black', linewidth=0.5)
axes[0].set_title('Vraies Classes')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].grid(True, alpha=0.3)

# Résultats du clustering
axes[1].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', 
                s=50, edgecolors='black', linewidth=0.5)
axes[1].scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c='red',
    marker='X',
    s=300,
    edgecolors='black',
    linewidth=2,
    label='Centroïdes'
)
axes[1].set_title(f'Clustering K-Means (Silhouette={silhouette_avg:.2f})')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Algorithme K-Means - Détails**

1. **Initialisation** : Choisir k centroïdes aléatoirement (ou avec K-means++)
2. **Affectation** : Assigner chaque point au centroïde le plus proche
   $$c^{(i)} = \arg\min_j ||x^{(i)} - \mu_j||^2$$
3. **Mise à jour** : Recalculer les centroïdes comme moyenne des points assignés
   $$\mu_j = \frac{1}{|C_j|}\sum_{i \in C_j} x^{(i)}$$
4. **Convergence** : Répéter 2-3 jusqu'à stabilisation des centroïdes

**Complexité** : O(n·k·i·d) où n=échantillons, k=clusters, i=itérations, d=dimensions

#### 2.1.3 Apprentissage par Renforcement

**Principe Fondamental**

Un **agent** apprend à prendre des décisions optimales en interagissant avec un **environnement**. L'apprentissage se fait par essai-erreur, guidé par des récompenses (feedback différé).

**Composants clés du RL**

- **État (State, s)** : Situation actuelle de l'environnement
- **Action (Action, a)** : Choix effectué par l'agent
- **Récompense (Reward, r)** : Feedback immédiat de l'environnement
- **Politique (Policy, π)** : Stratégie de décision π(a|s)
- **Fonction de valeur (Value function, V)** : Récompense cumulée attendue
- **Fonction Q (Q-function)** : Valeur d'une action dans un état donné

**Équation de Bellman**

$$
V(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]
$$

Où $\gamma$ est le facteur de discount (0 < γ < 1).

**[SCHÉMA 2.3 : Boucle agent-environnement en apprentissage par renforcement avec états, actions, récompenses]**

**Exemple Détaillé : Q-Learning sur GridWorld**

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GridWorld:
    """
    Environnement de grille 5x5 où l'agent doit atteindre un objectif.
    """
    def __init__(self, size=5):
        self.size = size
        self.agent_pos = [0, 0]  # Position initiale (coin supérieur gauche)
        self.goal_pos = [4, 4]   # Objectif (coin inférieur droit)
        self.obstacles = [[2, 2], [3, 2]]  # Obstacles
        self.actions = ['up', 'down', 'left', 'right']
        self.action_to_delta = {
            0: [-1, 0],  # up
            1: [1, 0],   # down
            2: [0, -1],  # left
            3: [0, 1]    # right
        }
        
    def reset(self):
        """Réinitialise l'environnement."""
        self.agent_pos = [0, 0]
        return tuple(self.agent_pos)
    
    def step(self, action):
        """
        Exécute une action et retourne (nouvel_état, récompense, terminé).
        
        Args:
            action: Entier 0-3 représentant l'action
        
        Returns:
            next_state: Tuple (x, y) de la nouvelle position
            reward: Récompense immédiate
            done: Boolean indiquant si l'épisode est terminé
        """
        # Calcul de la nouvelle position
        delta = self.action_to_delta[action]
        new_pos = [
            self.agent_pos[0] + delta[0],
            self.agent_pos[1] + delta[1]
        ]
        
        # Vérification des limites
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size and
            new_pos not in self.obstacles):
            self.agent_pos = new_pos
        
        # Calcul de la récompense
        if self.agent_pos == self.goal_pos:
            reward = 100  # Grande récompense pour atteindre l'objectif
            done = True
        elif self.agent_pos in self.obstacles:
            reward = -10  # Pénalité pour obstacle
            done = False
        else:
            reward = -1   # Petite pénalité pour encourager rapidité
            done = False
        
        return tuple(self.agent_pos), reward, done
    
    def render(self, q_table=None):
        """Visualise l'environnement et optionnellement la politique."""
        grid = np.zeros((self.size, self.size))
        grid[self.goal_pos[0], self.goal_pos[1]] = 2  # Objectif
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = -1  # Obstacles
        grid[self.agent_pos[0], self.agent_pos[1]] = 1  # Agent
        
        plt.figure(figsize=(8, 8))
        sns.heatmap(grid, annot=True, fmt='.0f', cmap='RdYlGn', 
                    cbar=False, square=True, linewidths=2)
        plt.title('GridWorld Environment')
        plt.show()


def q_learning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Implémentation de l'algorithme Q-Learning.
    
    Args:
        env: Environnement GridWorld
        episodes: Nombre d'épisodes d'entraînement
        alpha: Taux d'apprentissage (learning rate)
        gamma: Facteur de discount (importance du futur)
        epsilon: Probabilité d'exploration (epsilon-greedy)
    
    Returns:
        q_table: Table Q apprise
        rewards_per_episode: Liste des récompenses cumulées par épisode
    """
    # Initialisation de la Q-table à zéro
    q_table = np.zeros((env.size, env.size, len(env.actions)))
    rewards_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        max_steps = 100  # Limite pour éviter les boucles infinies
        
        while not done and steps < max_steps:
            # Stratégie epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(0, len(env.actions))  # Exploration
            else:
                action = np.argmax(q_table[state[0], state[1]])  # Exploitation
            
            # Exécution de l'action