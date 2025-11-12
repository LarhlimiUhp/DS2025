Voici un exemple détaillé d’étude statistique de corrélation appliquée à la finance, incluant chaque étape et du code utilisable :

***

**1. Collecte et préparation des données réelles**
Supposons que l’on veut étudier la corrélation entre le rendement quotidien de l’action Apple (AAPL) et celui de l’indice S\&P 500 sur les 5 dernières années. On récupère ces données à partir de Yahoo Finance (via la librairie `yfinance` en Python ou le package `quantmod` en R).

***

**2. Calcul du coefficient de corrélation de Pearson**
Ce coefficient mesure la force et le sens de la relation linéaire entre deux variables quantitatives (\$ r \$ varie de -1 à 1).

***

**3. Interprétation dans le contexte financier**

- Un \$ r \$ proche de 1 indique que l’action tend à évoluer comme l’indice.
- Un \$ r \$ proche de 0 indique une absence de lien linéaire.
- Un \$ r \$ négatif indique des mouvements opposés.

Exemple typique : les grandes actions américaines sont souvent fortement corrélées avec le S\&P 500.
La corrélation est utile pour une gestion de portefeuille : elle permet de diversifier les actifs et de mieux gérer le risque.[^1]

***

**4. Visualisation graphique**
Le nuage de points (scatter plot) permet de visualiser la relation, et une droite de tendance montre la force de la corrélation.

***

**5. Exemple de code Python utilisant des données réelles (AAPL vs S\&P 500) :**

```python
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Téléchargement des données
aapl = yf.download('AAPL', start='2020-01-01', end='2025-01-01')['Adj Close']
sp500 = yf.download('^GSPC', start='2020-01-01', end='2025-01-01')['Adj Close']

# Calcul des rendements quotidiens
returns = pd.DataFrame({
    'AAPL': aapl.pct_change(),
    'SP500': sp500.pct_change()
}).dropna()

# Calcul de la corrélation de Pearson
corr = returns['AAPL'].corr(returns['SP500'])
print(f"Corrélation de Pearson : {corr}")

# Visualisation
plt.scatter(returns['SP500'], returns['AAPL'], alpha=0.5)
plt.xlabel('Rendement S&P 500 (%)')
plt.ylabel('Rendement AAPL (%)')
plt.title('Corrélation journalière : AAPL vs S&P 500')
plt.show()
```


***

**6. Exemple de code R (équivalent à l’exemple Python ci-dessus) :**

```r
library(quantmod)

# Téléchargement des données
getSymbols(c("AAPL", "^GSPC"), from="2020-01-01", to="2025-01-01")

# Calcul des rendements quotidiens
aapl_ret <- dailyReturn(Cl(AAPL))
sp500_ret <- dailyReturn(Cl(GSPC))

# Fusion des données
df <- merge(aapl_ret, sp500_ret, by="date")
colnames(df) <- c("Date", "AAPL", "SP500")

# Corrélation de Pearson
correlation <- cor(df$AAPL, df$SP500)
print(paste("Corrélation de Pearson :", correlation))

# Visualisation
plot(df$SP500, df$AAPL,
     xlab="Rendement S&P 500 (%)",
     ylab="Rendement AAPL (%)",
     main="Corrélation journalière : AAPL vs S&P 500")
abline(lm(df$AAPL~df$SP500), col="blue")
```


***

**Résumé :**

- **Étape 1** : Extraction des données financières réelles (actions et indices).
- **Étape 2** : Calcul des rendements et du coefficient de corrélation de Pearson.
- **Étape 3** : Interprétation dans un contexte de gestion de portefeuille, mesure du lien entre titres et marché.
- **Étape 4** : Visualisation graphique pour illustrer la relation.
- **Étape 5** : Exemples de code Python et R prêts à l’emploi.

Cet exemple illustre la méthodologie classique de la corrélation en finance et peut être adapté à toute paire d’actifs ou d’indices financiers.[^2][^1]
<span style="display:none">[^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://www.datacamp.com/fr/tutorial/coefficient-of-determination

[^2]: https://fr.linkedin.com/learning/les-statistiques-pour-la-finance-en-python/mesurer-et-visualiser-la-correlation-entre-actifs-financiers

[^3]: https://brightdata.fr/blog/donnees-web/analysis-with-python

[^4]: https://eric.univ-lyon2.fr/ricco/cours/cours/Analyse_de_Correlation.pdf

[^5]: https://corpus.ulaval.ca/server/api/core/bitstreams/aeb4c26e-3af8-4c0d-afb1-5b3e7f59ea64/content

[^6]: https://sonar.ch/rerodoc/328774/files/20190911_-_H_ni_Victor_-_Travail_de_bachelor_Corrig_.pdf

[^7]: https://www.youtube.com/watch?v=ijbJB1pPiak

[^8]: https://www.math.univ-toulouse.fr/~besse/pub/Appren_stat.pdf

[^9]: http://web.univ-ubs.fr/lmba/lardjane/python/c2.pdf


