import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix


class EnsembleModels:
    """
    Classe responsável por treinar e avaliar modelos Ensemble (Bagging e Boosting).
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Dicionário com os modelos Ensemble
        self.models = {
            "Random Forest (Bagging)": RandomForestClassifier(
                n_estimators=100,
                max_depth=20,  # Limitamos a profundidade para não decorar (overfitting)
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ),
            "Gradient Boosting (Boosting)": HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                class_weight='balanced',
                random_state=42
            )
        }
        self.predictions = {}

    def train_and_evaluate(self):
        """
        Treina os modelos Ensemble e imprime o Classification Report.
        """
        print("--- Iniciando o Treino dos Modelos Ensemble ---\n")

        for name, model in self.models.items():
            print(f"A treinar o modelo {name}...")
            # Treinar o modelo
            model.fit(self.X_train, self.y_train)

            # Fazer previsões
            y_pred = model.predict(self.X_test)
            self.predictions[name] = y_pred

            # Imprimir as métricas
            print(f"\nResultados para {name}:")
            print(classification_report(self.y_test, y_pred, target_names=["On-time", "Short Delay", "Long Delay"]))
            print("-" * 50)

    def plot_confusion_matrices(self):
        """
        Gera os gráficos das Matrizes de Confusão.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        for ax, (name, y_pred) in zip(axes, self.predictions.items()):
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
                        xticklabels=["On-time", "Short", "Long"],
                        yticklabels=["On-time", "Short", "Long"])
            ax.set_title(f'{name}')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')

        plt.tight_layout()
        plt.show()