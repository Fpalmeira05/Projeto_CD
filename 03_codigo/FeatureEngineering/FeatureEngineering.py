import pandas as pd
import numpy as np


class FeatureEngineering:
        def __init__(self):
                # Dicionário para guardar as médias de atraso por hora calculadas no Treino
                self.hourly_delay_map = {}

        def perform_feature_engineering(self, df, target=None):
                """
                Create new features from the existing variables (no data leakage)
                :param df: DataFrame com as features
                :param target: Series com os labels (atrasos). Só é passado no Treino!
                :return: DataFrame com as novas features
                """

                df_fe = df.copy()

                # Garantir que a data está no formato correto de DateTime
                df_fe['FL_DATE'] = pd.to_datetime(df_fe['FL_DATE'])

                # 1. Mês do ano (1 a 12)
                df_fe['MONTH'] = df_fe['FL_DATE'].dt.month

                # 2. Dia da semana (0 a 5, 0 = Segunda e 6 = Domingo)
                df_fe['DAY_OF_WEEK'] = df_fe['FL_DATE'].dt.dayofweek

                # 3. Verifica se é fim de semana (sim = 1 nao = 0)
                df_fe['IS_WEEKEND'] = np.where(df_fe['DAY_OF_WEEK'] >= 5, 1, 0)

                # 4. Estação do ano (1=Inverno, 2=Primavera, 3=Verão, 4=Outono)
                df_fe['SEASON'] = df_fe['MONTH'].apply(
                        lambda x: 1 if x in [12, 1, 2] else (2 if x in [3, 4, 5] else (3 if x in [6, 7, 8] else 4))
                )

                # 5. Verifica se está na altura de férias (julho = 7, agosto = 8 e dezembro = 12)
                df_fe['IS_HOLIDAY_MONTH'] = np.where(df_fe['MONTH'].isin([7, 8, 12]), 1, 0)

                # 6. Hora Planeada de Partida
                df_fe['CRS_DEP_HOUR'] = df_fe['CRS_DEP_TIME'] // 100
                df_fe['CRS_DEP_HOUR'] = df_fe['CRS_DEP_HOUR'].replace(24, 0)

                # 7. Período do Dia (0 = Madrugada, 1 = Manha, 2 = Tarde, 3 = Noite)
                bins = [-1, 5, 11, 18, 25]
                labels_num = [0, 1, 2, 3]
                df_fe['TIME_OF_DAY'] = pd.cut(df_fe['CRS_DEP_HOUR'], bins=bins, labels=labels_num)

                # 8. Rota (Origem -> Destino)
                df_fe['ROUTE'] = df_fe['ORIGIN'] + "_" + df_fe['DEST']

                # 9. Tipo de Voo (Curto, Médio ou Longo Curso)
                bins_dist = [-1, 500, 1500, df_fe['DISTANCE'].max() + 1]
                labels_dist = [0, 1, 2]
                df_fe['FLIGHT_TYPE'] = pd.cut(df_fe['DISTANCE'], bins=bins_dist, labels=labels_dist)
                df_fe['FLIGHT_TYPE'] = df_fe['FLIGHT_TYPE'].astype(float)

                # 10. Hora Planeada de Chegada
                df_fe['CRS_ARR_HOUR'] = df_fe['CRS_ARR_TIME'] // 100
                df_fe['CRS_ARR_HOUR'] = df_fe['CRS_ARR_HOUR'].replace(24, 0)

                # 11. Velocidade Planeada (Milhas por Minuto)
                df_fe['PLANNED_SPEED'] = df_fe['DISTANCE'] / (df_fe['CRS_ELAPSED_TIME'] + 0.001)

                # --- NOVA FEATURE: Média de Atrasos por Hora (Target Encoding) ---

                # Se o 'target' foi passado, significa que estamos a processar os dados de Treino
                if target is not None:
                        # Juntamos temporariamente o target para calcular as médias
                        temp_df = df_fe.copy()
                        temp_df['TARGET_DELAY'] = target

                        # Calcula a média de atraso para cada hora e guarda no dicionário da classe (memória)
                        self.hourly_delay_map = temp_df.groupby('CRS_DEP_HOUR')['TARGET_DELAY'].mean().to_dict()

                # Calcula a média global de atrasos caso apareça no teste uma hora que não existia no treino
                global_mean = sum(self.hourly_delay_map.values()) / len(
                        self.hourly_delay_map) if self.hourly_delay_map else 0

                # Mapeia as médias guardadas para a nova coluna (aplica-se ao Treino e ao Teste)
                df_fe['AVG_DELAY_PER_HOUR'] = df_fe['CRS_DEP_HOUR'].map(self.hourly_delay_map).fillna(global_mean)

                return df_fe
