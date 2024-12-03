import os
import numpy as np
import postech_TC4
import pandas as pd
import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from prophet.plot import plot_cross_validation_metric

# Função para exibir EDA
def exibir_eda(df):

    st.header("Análise Exploratória de Dados (EDA)")
    
    # Estatísticas Descritivas
    st.subheader("Estatísticas Descritivas")
    st.write(df['y'].describe())
    st.write(
        """
        Esta seção apresenta as estatísticas descritivas básicas dos preços do petróleo, 
        incluindo média, desvio padrão, mínimo, máximo e quartis.
        """
    )
    
    # Histograma
    st.subheader("Histograma do Preço do Petróleo")
    st.write(
        """
        O histograma mostra a distribuição dos preços do petróleo Brent ao longo do período analisado, 
        permitindo visualizar a frequência de diferentes faixas de preço.
        """
    )
    fig_hist = plt.figure(figsize=(10, 6))
    plt.hist(df['y'], bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribuição dos Preços do Petróleo")
    plt.xlabel("Preço (USD)")
    plt.ylabel("Frequência")
    st.pyplot(fig_hist)

    # Boxplot
    st.subheader("Boxplot do Preço do Petróleo")
    st.write(
        """
        O boxplot ilustra a dispersão dos preços do petróleo, destacando a mediana, quartis e possíveis outliers.
        """
    )
    fig_box = plt.figure(figsize=(4, 3))
    plt.boxplot(df['y'], vert=True)
    plt.title("Boxplot do Preço do Petróleo", fontsize=6)
    plt.ylabel("Preço (USD)", fontsize=5)
    plt.xticks(fontsize=5)  
    plt.yticks(fontsize=5)
    plt.tight_layout()
    st.pyplot(fig_box)

    # Autocorrelação e Autocorrelação Parcial
    st.subheader("Autocorrelação (ACF) e Autocorrelação Parcial (PACF)")
    st.write(
        """
        Os gráficos de ACF e PACF ajudam a identificar a presença de autocorrelações nos dados, 
        o que é útil para entender a dependência temporal e ajustar modelos de séries temporais.
        """
    )
    fig_acf, ax_acf = plt.subplots(2, 1, figsize=(12, 8))
    sm.graphics.tsa.plot_acf(df['y'], lags=50, ax=ax_acf[0])
    sm.graphics.tsa.plot_pacf(df['y'], lags=50, ax=ax_acf[1])
    ax_acf[0].set_title('Autocorrelação (ACF)')
    ax_acf[1].set_title('Autocorrelação Parcial (PACF)')
    st.pyplot(fig_acf)

# Função para exibir análise dos resíduos
def exibir_analise_residuos(df_residuos):
    st.subheader("Análise dos Resíduos")
    st.write(
        """
        A análise dos resíduos avalia a diferença entre os valores reais e as previsões do modelo. 
        Resíduos bem distribuídos indicam um bom ajuste do modelo.
        """
    )
    
    fig_resid, ax_resid = plt.subplots(2, 1, figsize=(12, 8))

    ax_resid[0].plot(df_residuos.index, df_residuos['residuo'], label='Resíduos')
    ax_resid[0].hlines(0, xmin=df_residuos.index.min(), xmax=df_residuos.index.max(), colors='r', linestyles='--')
    ax_resid[0].set_title("Resíduos ao longo do tempo")
    ax_resid[0].set_xlabel("Data")
    ax_resid[0].set_ylabel("Resíduo")
    ax_resid[0].legend()

    ax_resid[1].hist(df_residuos['residuo'], bins=50, color='skyblue', edgecolor='black')
    ax_resid[1].set_title("Histograma dos Resíduos")
    ax_resid[1].set_xlabel("Resíduo")
    ax_resid[1].set_ylabel("Frequência")

    plt.tight_layout()
    st.pyplot(fig_resid)

# Função para exibir o gráfico e métricas
def exibir_previsao_detalhada(
    dados_treino, dados_teste, previsoes, mae, rmse, acuracia
):
    st.subheader("Previsões do Modelo Prophet")
    st.write(
        """
        Este gráfico apresenta os dados históricos de treino, os dados reais de teste e as previsões geradas pelo modelo Prophet, 
        permitindo visualizar a precisão das previsões em relação aos dados reais.
        """
    )

    # Plotar os dados históricos e as previsões
    plt.figure(figsize=(10, 6))
    plt.plot(
        dados_treino['ds'],
        dados_treino['y'],
        label='Dados Históricos (Treino)'
    )
    plt.plot(
        dados_teste['ds'],
        dados_teste['y'],
        label='Dados Reais (Teste)',
        color='orange'
    )
    plt.plot(
        previsoes['ds'],
        previsoes['yhat'],
        label='Previsões',
        color='green'
    )

    # Destacar o período de previsão
    plt.axvline(
        x=dados_treino['ds'].iloc[-1],
        color='red',
        linestyle='--',
        label='Início das Previsões'
    )

    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.title('Previsões do Modelo Prophet')
    plt.grid(":")
    plt.legend()
    plt.tight_layout()  # Ajustar layout para evitar cortes
    st.pyplot(plt.gcf())

    # Exibir métricas de erro
    st.write(
        """
        **Métricas de Desempenho:**
        - **Erro Absoluto Médio (MAE):** Indica a média das diferenças absolutas entre os valores reais e previstos.
        - **Root Mean Squared Error (RMSE):** Mede a raiz quadrada da média dos erros ao quadrado, penalizando mais os grandes erros.
        - **Acurácia:** Representa a proporção de previsões corretas em relação aos valores reais.
        """
    )
    st.write(f"**Erro Absoluto Médio (MAE):** {mae:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**Acurácia:** {acuracia:.2f}%")

    # Preparar o CSV com as previsões futuras
    previsoes_futuras = previsoes[previsoes['ds'] > dados_teste['ds'].max()]
    previsoes_futuras_csv = previsoes_futuras[["ds", "yhat"]].to_csv(index=False)
    st.download_button(
        "Baixar Previsões Futuras em CSV",
        data=previsoes_futuras_csv,
        file_name="previsoes_futuras_petroleo.csv",
        mime="text/csv",
    )

# Função para exibir insights
def exibir_insights():
    st.subheader("Insights sobre o Preço do Petróleo")
    st.write(
        """
        **Insights Derivados das Análises:**
        
        1. **Tendência de Alta nos Preços:** A análise da tendência mostra que os preços do petróleo Brent têm apresentado uma tendência de alta nos últimos anos, 
        indicando um aumento consistente no valor do barril.
        
        2. **Sazonalidade Evidente:** Os gráficos de sazonalidade revelam padrões sazonais claros, com aumentos nos preços durante determinados meses do ano, 
        possivelmente relacionados a fatores climáticos e de demanda.
        
        3. **Volatilidade nos Preços:** A autocorrelação e a análise dos resíduos indicam que há períodos de alta volatilidade nos preços do petróleo, 
        sugerindo que eventos externos têm um impacto significativo nas variações diárias.
        
        4. **Ajuste Adequado do Modelo:** A análise dos resíduos mostra que os resíduos estão bem distribuídos e próximos de zero, 
        o que sugere que o modelo Prophet está capturando eficazmente os padrões nos dados.
        
        5. **Previsões Sólidas para o Futuro:** As previsões geradas pelo modelo Prophet indicam que os preços do petróleo Brent continuarão a seguir a tendência de alta, 
        com variações sazonais moderadas.
        """
    )

# Função para exibir informações dos desenvolvedores
def exibir_creditos():
    st.subheader("Desenvolvedores")
    st.write("Este projeto foi desenvolvido por:")
    st.write("- **Nome da Desenvolvedora:** Clara Crizio de Araujo Torres ")
    st.write("- **Nome da Desenvolvedora:** Isabela de Jesus Santos")
    st.write("- **Nome do Desenvolvedor:**  Willian C. Rodrigues")
    st.write(
        "**Turma:** Grupo 17 / **Módulo:** Data Viz and Production Models / FIAP - Pós Tech"
    )

# Função principal que contém a lógica do seu dashboard
def main():
    st.title("Dashboard de Previsão do Preço do Petróleo Brent")
    st.write(
        """
        ### Bem-vindo ao Dashboard de Previsão do Petróleo
        Este painel interativo foi desenvolvido como parte de um projeto de análise e previsão dos preços do petróleo Brent.
        A previsão dos preços é baseada em dados históricos e realizada através do modelo Prophet, uma técnica de séries temporais.

        O objetivo deste dashboard é auxiliar na compreensão dos principais fatores que influenciam o preço do petróleo, como:
        - Eventos geopolíticos
        - Crises econômicas
        - Demanda global por energia
        - Transição para energias renováveis

        A ferramenta permite explorar o histórico de preços e visualizar previsões futuras, facilitando insights e suporte à tomada de decisão.
        """
    )

    # Entrada para definir o período de previsão
    st.sidebar.header("Configurações do Modelo")
    st.sidebar.write(
        "Ajuste os parâmetros do modelo e o período de previsão."
    )
    periodo_previsao = st.sidebar.number_input(
        "Período de previsão (em dias)",
        min_value=1,
        max_value=365 * 10,
        value=365,
        step=1,
    )

    # Upload do arquivo de dados
    st.sidebar.header("Upload do Arquivo de Dados")
    arquivo = st.sidebar.file_uploader("Escolha o arquivo CSV", type=["csv"])

    if arquivo is not None:
        # Carregar os dados usando o módulo postech_TC4
        try:
            df = postech_TC4.carregar_dados(arquivo)
            st.write("### Dados Históricos do Preço do Petróleo Brent")

            # Exibir algumas informações sobre os dados carregados
            st.write(f"**Total de registros carregados:** {len(df)}")
            st.write("**Visualização dos dados carregados:**")
            st.write(df.head())

            # EDA
            exibir_eda(df)

            # Decomposição da série temporal
            st.write("### Decomposição da Série Temporal do Preço do Petróleo Brent")
            st.write(
                """
                A decomposição da série temporal separa os dados em componentes de tendência, sazonalidade e resíduo, 
                permitindo uma análise mais detalhada dos padrões presentes nos preços do petróleo.
                """
            )

            # Realizar a decomposição
            decomposicao = sm.tsa.seasonal_decompose(
                df.set_index('ds')['y'], model='additive', period=30
            )

            # Criar subplots
            fig_decomposicao, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))

            decomposicao.observed.plot(ax=ax1)
            ax1.set_ylabel('Observado')
            ax1.set_title('Observado')

            decomposicao.trend.plot(ax=ax2)
            ax2.set_ylabel('Tendência')
            ax2.set_title('Tendência')

            decomposicao.seasonal.plot(ax=ax3)
            ax3.set_ylabel('Sazonalidade')
            ax3.set_title('Sazonalidade')

            decomposicao.resid.plot(ax=ax4)
            ax4.set_ylabel('Resíduo')
            ax4.set_title('Resíduo')

            plt.tight_layout()

            # Exibir o gráfico no Streamlit
            st.pyplot(fig_decomposicao)

        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")
            return
    else:
        st.info("Por favor, faça o upload do arquivo CSV contendo os dados.")
        return

    # Dividir dados com proporção fixa de 80% treino e 20% teste
    proporcao_treino = 0.8
    try:
        dados_treino, dados_teste = postech_TC4.dividir_dados(
            df, proporcao_treino=proporcao_treino
        )
    except Exception as e:
        st.error(f"Erro ao dividir os dados: {e}")
        return

    st.write(f"**Dados de Treino:** {len(dados_treino)} registros")
    st.write(f"**Dados de Teste:** {len(dados_teste)} registros")

    # Treinar o modelo e obter previsões para todo o período
    with st.spinner('Treinando o modelo, por favor aguarde...'):
        try:
            modelo = postech_TC4.treinar_modelo_prophet(dados_treino)

            # Criar DataFrame com datas futuras (incluindo datas do teste e previsões futuras)
            total_periods = len(dados_teste) + periodo_previsao
            futuro = modelo.make_future_dataframe(periods=total_periods, freq='D')  # Especificar a frequência
            previsoes = modelo.predict(futuro)

            # Verificar se todas as datas de dados_teste estão em previsoes
            previsoes_set = set(previsoes['ds'])
            dados_teste_set = set(dados_teste['ds'])

            datas_faltando = dados_teste_set - previsoes_set

            if datas_faltando:
                st.warning(f"Existem {len(datas_faltando)} datas no conjunto de teste que estão faltando nas previsões.")
                st.write("**Datas faltando:**", sorted(datas_faltando))
            else:
                st.success("Todas as datas do conjunto de teste estão presentes nas previsões.")

            # Exibir algumas informações sobre as previsões
            st.write("### Previsões Geradas")
            st.write(previsoes.tail())

            # Calcular métricas
            mae, rmse, acuracia = postech_TC4.calcular_metricas(dados_teste, previsoes)

            exibir_previsao_detalhada(
                dados_treino, dados_teste, previsoes, mae, rmse, acuracia
            )

            # Análise dos resíduos
            dados_teste = dados_teste.set_index('ds')
            previsoes_teste = previsoes.set_index('ds').reindex(dados_teste.index)

            # Verificar e remover linhas com NaN
            if previsoes_teste['yhat'].isnull().any():
                st.warning("Existem previsões ausentes para algumas datas do conjunto de teste. Essas entradas serão removidas da análise dos resíduos.")
                df_merged = dados_teste.join(previsoes_teste[['yhat']], how='inner').dropna(subset=['yhat'])
            else:
                df_merged = dados_teste.join(previsoes_teste[['yhat']], how='inner')

            st.write(f"**Dados Merged:** {len(df_merged)} registros após alinhar as previsões com os dados de teste.")

            df_residuos = postech_TC4.analisar_residuos(df_merged)
            exibir_analise_residuos(df_residuos)

            st.success('Modelo treinado com sucesso!')
        except Exception as e:
            st.error(f"Ocorreu um erro durante o treinamento: {e}")
            return

    exibir_insights()
    exibir_creditos()
