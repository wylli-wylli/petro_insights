import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.diagnostics import cross_validation, performance_metrics

def carregar_dados(arquivo):
    """
    Carrega e trata os dados de um arquivo CSV.

    Parâmetros:
    - arquivo: Objeto de arquivo (pode ser um caminho ou um objeto de arquivo similar a file-like).

    Retorna:
    - DataFrame tratado.
    """
    try:
        
        if isinstance(arquivo, str):
            df = pd.read_csv(arquivo, delimiter=';')
        else:
            df = pd.read_csv(arquivo, delimiter=',')
    except Exception as e:
        raise ValueError(f"Erro ao ler o arquivo CSV: {e}")

    required_columns = [
        'Data',
        'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366'
    ]
    if not set(required_columns).issubset(df.columns):
        raise ValueError("Arquivo CSV não possui as colunas esperadas.")

    df.rename(columns={
        'Data': 'ds',
        'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366': 'y'
    }, inplace=True)

    # Converter colunas para tipos adequados
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(subset=['ds', 'y'], inplace=True)
    df = df[df['ds'] >= '2000-01-01']
    df = df[df['ds'].dt.dayofweek < 5]  # Considerar apenas dias úteis
    df.sort_values('ds', inplace=True)  # Garantir que os dados estão ordenados
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError("Após o processamento, o DataFrame está vazio. Verifique os dados de entrada.")

    return df

def dividir_dados(df, proporcao_treino=0.8):
    """
    Divide os dados em conjuntos de treino e teste.

    Parâmetros:
    - df: DataFrame completo.
    - proporcao_treino: Proporção de dados para treino (entre 0 e 1).

    Retorna:
    - dados_treino: DataFrame de treino.
    - dados_teste: DataFrame de teste.
    """
    if not 0 < proporcao_treino < 1:
        raise ValueError("A proporção de treino deve estar entre 0 e 1.")

    tamanho_treino = int(len(df) * proporcao_treino)
    dados_treino = df.iloc[:tamanho_treino].copy()
    dados_teste = df.iloc[tamanho_treino:].copy()

    if dados_treino.empty or dados_teste.empty:
        raise ValueError("Conjuntos de treino ou teste estão vazios após a divisão.")

    return dados_treino, dados_teste

def treinar_modelo_prophet(dados_treino):
    """
    Treina o modelo Prophet com ajustes de hiperparâmetros.

    Parâmetros:
    - dados_treino: DataFrame de treino com colunas 'ds' e 'y'.

    Retorna:
    - modelo: Modelo treinado do Prophet.
    """
    if dados_treino.empty:
        raise ValueError("O conjunto de dados de treino está vazio.")

    modelo = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='additive', 
        changepoint_prior_scale=0.05,  # Ajuste da flexibilidade da tendência
        interval_width=0.95  # Intervalo de confiança de 95%
    )
    modelo.add_country_holidays(country_name='BR')

    # Se houver regressoras exógenas, adicionar aqui
    # Exemplo:
    # modelo.add_regressor('nome_da_regressora')

    # Treinar o modelo
    try:
        modelo.fit(dados_treino)
    except Exception as e:
        raise ValueError(f"Erro ao treinar o modelo Prophet: {e}")

    return modelo

def cross_validation_prophet(modelo, horizon='30 days', period='15 days', initial='365 days'):
    """
    Realiza validação cruzada no modelo Prophet.

    Parâmetros:
    - modelo: Modelo treinado do Prophet.
    - horizon: Horizonte para a previsão na validação cruzada.
    - period: Intervalo entre as previsões na validação cruzada.
    - initial: Período inicial de dados para a validação cruzada.

    Retorna:
    - df_cv: DataFrame com as previsões da validação cruzada.
    - df_p: DataFrame com as métricas de desempenho da validação cruzada.
    """
    try:
        df_cv = cross_validation(modelo, initial=initial, period=period, horizon=horizon)
        df_p = performance_metrics(df_cv)
    except Exception as e:
        raise ValueError(f"Erro durante a validação cruzada: {e}")

    return df_cv, df_p

def calcular_metricas(dados_teste, previsoes):
    """
    Calcula métricas de erro e acurácia usando o conjunto de teste.

    Parâmetros:
    - dados_teste: DataFrame de teste com colunas 'ds' e 'y'.
    - previsoes: DataFrame com previsões do Prophet.

    Retorna:
    - mae: Erro Absoluto Médio.
    - rmse: Root Mean Squared Error.
    - acuracia: Acurácia em porcentagem.
    """
    if dados_teste.empty:
        raise ValueError("O conjunto de dados de teste está vazio.")
    if previsoes.empty:
        raise ValueError("Nenhuma previsão foi gerada.")

    # Filtrar previsões correspondentes ao conjunto de teste
    previsoes_teste = previsoes[previsoes['ds'].isin(dados_teste['ds'])]

    if previsoes_teste.empty:
        raise ValueError("As previsões não contêm datas correspondentes ao conjunto de teste.")

    # Alinhar os dados pelas datas
    dados_teste = dados_teste.set_index('ds')
    previsoes_teste = previsoes_teste.set_index('ds')

    df_merged = dados_teste.join(previsoes_teste[['yhat']], how='inner')

    if df_merged.empty:
        raise ValueError("As previsões não contêm datas correspondentes ao conjunto de teste.")

    # Garantir que os índices estão alinhados
    real_values = df_merged['y'].values
    predictions = df_merged['yhat'].values

    # Calcular métricas
    mae = mean_absolute_error(real_values, predictions)
    rmse = np.sqrt(mean_squared_error(real_values, predictions))
    accuracy = 1 - mae / np.mean(np.abs(real_values))
    acuracia = accuracy * 100  # Converter para porcentagem

    return mae, rmse, acuracia

def analisar_residuos(df_merged):
    """
    Analisa os resíduos do modelo.

    Parâmetros:
    - df_merged: DataFrame resultante da junção dos dados de teste com as previsões.

    Retorna:
    - DataFrame com uma coluna adicional 'residuo'.
    """
    df_merged['residuo'] = df_merged['y'] - df_merged['yhat']
    return df_merged

if __name__ == "__main__":
    caminho_arquivo = r"C:\Users\ozzy\OneDrive - sonardd.com.br\Documents\FIAP\tech_challenge 4\data\ipeadata[04-11-2024-09-30].csv"

    try:
        df = carregar_dados(caminho_arquivo)
        dados_treino, dados_teste = dividir_dados(df, proporcao_treino=0.8)

        modelo = treinar_modelo_prophet(dados_treino)

        total_periods = len(dados_teste) + 365  # Exemplo: previsão de 1 ano
        futuro = modelo.make_future_dataframe(periods=total_periods, freq='D')
        previsoes = modelo.predict(futuro)

        mae, rmse, acuracia = calcular_metricas(dados_teste, previsoes)
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, Acurácia: {acuracia:.2f}%")

        # Realizar validação cruzada
        df_cv, df_p = cross_validation_prophet(modelo)

        # Análise dos resíduos
        dados_teste = dados_teste.set_index('ds')
        previsoes_teste = previsoes.set_index('ds').loc[dados_teste.index]
        df_merged = dados_teste.join(previsoes_teste[['yhat']])
        df_residuos = analisar_residuos(df_merged)

    except Exception as e:
        print(f"Erro: {e}")
