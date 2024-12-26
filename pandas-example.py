import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Завантаження набору даних про вино
dataset = load_wine()
target_names = dataset.target_names
X = dataset.data  # Особливості
y = dataset.target  # Цільові класи

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення конвеєра для нормалізації та моделі класифікації
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=200))
])

# Крос-валідація для оцінки моделі
model_score = cross_validate(pipeline, X_train, y_train, return_train_score=True)
print("Train score:", round(model_score["train_score"].mean(), 4))
print("Test score:", round(model_score["test_score"].mean(), 4))

# Оцінка різних моделей класифікації
classifiers = [DecisionTreeClassifier(), SVC(), KNeighborsClassifier(), MLPClassifier(max_iter=500)]  # Increased max_iter

for classifier in classifiers:
    pipeline.set_params(model=classifier)
    model_score = cross_validate(pipeline, X_train, y_train)
    print(f"Score for {classifier.__class__.__name__} is {model_score['test_score'].mean():.4f}")

# Налаштування гіперпараметрів для моделі дерева рішень
pipeline.set_params(model=DecisionTreeClassifier())
params_grid = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": range(1, 5),
    "model__min_samples_leaf": [1, 5],
    "model__class_weight": [None, "balanced"]
}

grid_search_cv = GridSearchCV(pipeline, params_grid)
grid_search_cv.fit(X_train, y_train)

print("Best parameters:", grid_search_cv.best_params_)
print("Best score:", grid_search_cv.best_score_)

# Тестування найкращої моделі на тестових даних
best_estimator = grid_search_cv.best_estimator_
y_predicted = best_estimator.predict(X_test)

# Аналіз результатів
mean_accuracy = np.mean(y_predicted == y_test)
f1_weighted = f1_score(y_test, y_predicted, average='weighted')

print(f"Mean accuracy: {mean_accuracy:.4f}")
print(f"F1 weighted: {f1_weighted:.4f}")

# Візуалізація матриці невідповідностей
ConfusionMatrixDisplay.from_predictions(y_test, y_predicted)
plt.show()
