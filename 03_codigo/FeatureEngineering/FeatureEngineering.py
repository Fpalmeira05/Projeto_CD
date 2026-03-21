import pandas as pd
import numpy as np

class FeatureEngineering:

    def perform_feature_engineering(self, df):
            """
            Create 10 new features from the existing variables(no data leakage)
            :param self:
            :param df:
            :return:
            """

            df_fe = df.copy()

            # Garantir que a data está no formato correto de DateTime
            df_fe['FL_DATE'] = pd.to_datetime(df_fe['FL_DATE'])

            # 1. Mês do ano (1 a 12)
            df_fe['MONTH'] = df_fe['FL_DATE'].dt.month

            #2. Dia da semana(0 a 5 , 0 = Segunda  e 6 = Domingo)
            df_fe['DAY_OF_WEEK'] = df_fe['FL_DATE'].dt.dayofweek

            #3. Verifica se e fim de semana(sim = 1 nao = 0)
            df_fe['IS_WEEKEND'] = np.where(df_fe['DAY_OF_WEEK'] >= 5, 1, 0)

            # 4. Estação do ano (1=Inverno, 2=Primavera, 3=Verão, 4=Outono)
            df_fe['SEASON'] = df_fe['MONTH'].apply(
                lambda x: 1 if x in [12, 1, 2] else (2 if x in [3, 4, 5] else (3 if x in [6, 7, 8] else 4))
            )

            # 5. Verifica se esta na altura de ferias(julho = 7,agosto = 8 e dezembro = 12), 1 = sim 0 = nao
            df_fe['IS_HOLIDAY_MONTH'] = np.where(df_fe['MONTH'].isin([7, 8, 12]), 1, 0)

            # 6. Hora Planeada de Partida
            # O CRS_DEP_TIME vem frequentemente no formato hhmm (ex: 1530 para 15:30).
            # A divisão inteira por 100 extrai apenas a hora (15).
            df_fe['CRS_DEP_HOUR'] = df_fe['CRS_DEP_TIME'] // 100
            # Correção para casos raros onde a hora passa as 24h
            df_fe['CRS_DEP_HOUR'] = df_fe['CRS_DEP_HOUR'].replace(24,0)

            #7. Periodo do Dia (0 = Madrugada, 1 = Manha, 2 = Tarde, 3 = Noite)
            bins = [-1,5,11,18,25] #(0 as 5 horas = Madrugada, 5 as 11 = Manha, 11 as 18 = Tarde, 18 as 23 = Noite)
            labels_num = [0, 1, 2, 3]
            df_fe['TIME_OF_DAY'] = pd.cut(df_fe['CRS_DEP_HOUR'], bins=bins, labels=labels_num)

            # 8. Rota (Origem -> Destino)
            # O modelo pode descobrir que a rota "JFK_LAX" se atrasa mais que outras.
            df_fe['ROUTE'] = df_fe['ORIGIN'] + "_" + df_fe['DEST']

            # 10. Tipo de Voo (Curto, Médio ou Longo Curso)
            # Curto < 500 milhas | Médio 500 a 1500 milhas | Longo > 1500 milhas
            bins_dist = [-1, 500, 1500, df_fe['DISTANCE'].max() + 1]
            labels_dist = [0, 1, 2]  # 0: Curto, 1: Médio, 2: Longo
            df_fe['FLIGHT_TYPE'] = pd.cut(df_fe['DISTANCE'], bins=bins_dist, labels=labels_dist)
            df_fe['FLIGHT_TYPE'] = df_fe['FLIGHT_TYPE'].astype(float)

            # 2. Hora Planeada de Chegada
            df_fe['CRS_ARR_HOUR'] = df_fe['CRS_ARR_TIME'] // 100
            df_fe['CRS_ARR_HOUR'] = df_fe['CRS_ARR_HOUR'].replace(24, 0)  # Corrige as 24h para 0h

            # 3. Velocidade Planeada (Milhas por Minuto)
            # Somamos 0.001 para evitar um erro fatal caso algum tempo seja 0
            df_fe['PLANNED_SPEED'] = df_fe['DISTANCE'] / (df_fe['CRS_ELAPSED_TIME'] + 0.001)





            return df_fe
