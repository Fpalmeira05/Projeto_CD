import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


class SupervisedModels:
    """
    Classe responsável por treinar e avaliar modelos supervisionados clássicos.
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Dicionário com os modelos, já protegidos contra o desequilíbrio das classes!
        self.models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42,
                                                      class_weight='balanced', n_jobs=-1)
        }
        self.predictions = {}

    def train_and_evaluate(self):
        """
        Treina os modelos e imprime o Classification Report (Precision, Recall, F1-Score).
        """
        print("--- Iniciando o Treino dos Modelos Supervisionados ---\n")

        for name, model in self.models.items():
            print(f"A treinar o modelo: {name} (isto pode demorar uns minutos)...")
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
        Gera os gráficos das Matrizes de Confusão para visualizar os erros do modelo.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, (name, y_pred) in zip(axes, self.predictions.items()):
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=["On-time", "Short", "Long"],
                        yticklabels=["On-time", "Short", "Long"])
            ax.set_title(f'Confusion Matrix: {name}')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')

        plt.tight_layout()
        plt.show()