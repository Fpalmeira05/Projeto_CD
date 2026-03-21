import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, filename, test_size=0.2, random_state=42):
        self.filename = filename
        self.test_size = test_size
        self.random_state = random_state
        self.data_train = None
        self.labels_train = None
        self.data_test = None
        self.labels_test = None

        # Load data immediately
        self._load_data()

    def _load_data(self):
        try:
            # 1. Carregar o dataset
            df = pd.read_csv(self.filename, low_memory=False)
            print(f"Original shape: {df.shape}")

            # 2. Remover cancelados e divergidos
            df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]
            print(f"Shape after removing cancelled/diverted: {df.shape}")

            # --- INSTRUÇÕES DO PROFESSOR ---

            # A) Converter todos os atrasos negativos (voos adiantados) para 0
            df['ARR_DELAY'] = df['ARR_DELAY'].clip(lower=0)

            # B) Amostragem Equilibrada (Undersampling)
            df_atrasados = df[df['ARR_DELAY'] > 0]
            df_pontuais = df[df['ARR_DELAY'] == 0]

            print(f"Voos atrasados encontrados: {len(df_atrasados)}")
            print(f"Voos pontuais/adiantados encontrados: {len(df_pontuais)}")

            # Escolher aleatoriamente um número de voos pontuais igual ao número de voos atrasados
            df_pontuais_amostra = df_pontuais.sample(n=len(df_atrasados), random_state=self.random_state)

            # Juntar as duas metades e baralhar o dataset (frac=1)
            df = pd.concat([df_atrasados, df_pontuais_amostra]).sample(frac=1, random_state=self.random_state)
            print(f"Shape final após amostragem 50/50: {df.shape}")

            # -------------------------------

            # 3. Separar o Target (y) e as Features (X)
            y = df['ARR_DELAY']
            X = df.drop(columns=['ARR_DELAY'])

            # 4. Dividir em Treino e Teste
            self.data_train, self.data_test, self.labels_train, self.labels_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            print("Data loaded, balanced and split successfully.")

        except FileNotFoundError:
            print("File not found. Please check the file path.")