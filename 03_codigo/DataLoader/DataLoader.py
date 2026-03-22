import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.errors import EmptyDataError


class DataLoader:
    """
    Class responsible for loading the Flight dataset and splitting it.
    """

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
        # --- VERIFICAÇÃO 1: O ficheiro existe? ---
        if not os.path.exists(self.filename):
            print(f"Erro Crítico: O ficheiro '{self.filename}' não foi encontrado no caminho especificado.")
            return

        try:
            # 1. Carregar o dataset
            df = pd.read_csv(self.filename, low_memory=False)

            # --- VERIFICAÇÃO 2: O dataset está vazio? ---
            if df.empty:
                print("Erro Crítico: O dataset carregado não tem dados (está vazio).")
                return

            print(f"Original shape: {df.shape}")

            # --- VERIFICAÇÃO 3: As colunas obrigatórias existem? ---
            colunas_obrigatorias = ['CANCELLED', 'DIVERTED', 'ARR_DELAY']
            colunas_em_falta = [col for col in colunas_obrigatorias if col not in df.columns]
            if colunas_em_falta:
                print(f"Erro Crítico: Faltam colunas essenciais no dataset: {colunas_em_falta}")
                return

            # 2. Remover cancelados e divergidos
            df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]

            # --- VERIFICAÇÃO 4: Ficaram dados após a limpeza? ---
            if df.empty:
                print("Erro Crítico: O dataset ficou sem dados após remover os voos cancelados e divergidos.")
                return

            print(f"Shape after removing cancelled/diverted: {df.shape}")

            # --- INSTRUÇÕES DO PROFESSOR ---

            # A) Converter todos os atrasos negativos (voos adiantados) para 0
            df['ARR_DELAY'] = df['ARR_DELAY'].clip(lower=0)

            # B) Amostragem Equilibrada (Undersampling)
            df_atrasados = df[df['ARR_DELAY'] > 0]
            df_pontuais = df[df['ARR_DELAY'] == 0]

            # --- VERIFICAÇÃO 5: Temos dados suficientes para a amostragem? ---
            if len(df_atrasados) == 0 or len(df_pontuais) == 0:
                print(
                    "Erro Crítico: Não existem dados suficientes de voos atrasados ou pontuais para fazer a amostragem 50/50.")
                return

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

        except EmptyDataError:
            print("Erro Crítico: O ficheiro CSV existe, mas está completamente vazio ou corrompido.")
        except Exception as e:
            print(f"Ocorreu um erro inesperado: {e}")