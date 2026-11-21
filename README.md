# Machine Learning model for ATP match prediction

Model that estimates the win probability in ATP tour matches using only pre-match information, with a tournament simulator.

### Main idea:
Predict who wins a professional tennis match and explain why based on each player's history, and test whether that history outperforms the official ranking as a predictor.


### Technologies used:
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook


### Architectural aspects:
- ETL on more than 78,000 ATP matches (2000–2026)
- Custom surface-specific Elo rating and features built only from previous matches
- Ensemble of Logistic Regression, Random Forest and Gradient Boosting with temporal validation
- Tournament simulator based on the model's probabilities
