import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.diagnostics import cross_validation, performance_metrics

# Função para converter arquivo Excel em CSV
def converter_excel_para_csv(arquivo_excel, arquivo_csv):
    try:
        df = pd.read_excel(arquivo_excel, engine='openpyxl')
        df.to_csv(arquivo_csv, sep=',', index=False)
    except Exception as e:
        raise ValueError(f"Erro ao converter o arquivo Excel para CSV: {e}")

# Função para carregar e tratar dados do arquivo CSV
def carregar_dados(arquivo):
    """
    Carrega e trata os dados de um arquivo Excel ou CSV e transforma em um DataFrame.

    Parâmetros:
    - arquivo: Caminho do arquivo (Excel ou CSV).

    Retorna:
    - DataFrame tratado.
    """
    try:
        # Converter o arquivo para CSV, se for um Excel
        if isinstance(arquivo, str) and arquivo.endswith(('.xlsx', '.xls')):
            arquivo_csv = arquivo.replace('.xlsx', '.csv').replace('.xls', '.csv')
            converter_excel_para_csv(arquivo, arquivo_csv)
            arquivo = arquivo_csv

        # Carregar o CSV gerado ou enviado, sempre considerando vírgula como delimitador
        df = pd.read_csv(arquivo, delimiter=',')
    except Exception as e:
        raise ValueError(f"Erro ao ler o arquivo CSV: {e}")

    # Verificar se o DataFrame possui as colunas esperadas
    required_columns = [
        'Data',
        'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366'
    ]
    if not set(required_columns).issubset(df.columns):
        raise ValueError("Por favor, verifique o arquivo e envie um compatível com a base de dados esperada.")

    # Renomear colunas
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

# Função para dividir os dados em treino e teste
def dividir_dados(df, proporcao_treino=0.8):
    if not 0 < proporcao_treino < 1:
        raise ValueError("A proporção de treino deve estar entre 0 e 1.")

    tamanho_treino = int(len(df) * proporcao_treino)
    dados_treino = df.iloc[:tamanho_treino].copy()
    dados_teste = df.iloc[tamanho_treino:].copy()

    if dados_treino.empty or dados_teste.empty:
        raise ValueError("Conjuntos de treino ou teste estão vazios após a divisão.")

    return dados_treino, dados_teste

# Função para treinar o modelo Prophet
def treinar_modelo_prophet(dados_treino):
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

    try:
        modelo.fit(dados_treino)
    except Exception as e:
        raise ValueError(f"Erro ao treinar o modelo Prophet: {e}")

    return modelo

# Função para realizar validação cruzada no modelo Prophet
def cross_validation_prophet(modelo, horizon='30 days', period='15 days', initial='365 days'):
    try:
        df_cv = cross_validation(modelo, initial=initial, period=period, horizon=horizon)
        df_p = performance_metrics(df_cv)
    except Exception as e:
        raise ValueError(f"Erro durante a validação cruzada: {e}")

    return df_cv, df_p

# Função para calcular métricas de erro e acurácia
def calcular_metricas(dados_teste, previsoes):
    if dados_teste.empty:
        raise ValueError("O conjunto de dados de teste está vazio.")
    if previsoes.empty:
        raise ValueError("Nenhuma previsão foi gerada.")

    previsoes_teste = previsoes[previsoes['ds'].isin(dados_teste['ds'])]

    if previsoes_teste.empty:
        raise ValueError("As previsões não contêm datas correspondentes ao conjunto de teste.")

    dados_teste = dados_teste.set_index('ds')
    previsoes_teste = previsoes_teste.set_index('ds')

    df_merged = dados_teste.join(previsoes_teste[['yhat']], how='inner')

    if df_merged.empty:
        raise ValueError("As previsões não contêm datas correspondentes ao conjunto de teste.")

    real_values = df_merged['y'].values
    predictions = df_merged['yhat'].values

    mae = mean_absolute_error(real_values, predictions)
    rmse = np.sqrt(mean_squared_error(real_values, predictions))
    accuracy = 1 - mae / np.mean(np.abs(real_values))
    acuracia = accuracy * 100  # Converter para porcentagem

    return mae, rmse, acuracia

# Função para analisar os resíduos
def analisar_residuos(df_merged):
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
