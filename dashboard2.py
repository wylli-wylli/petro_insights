import os
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from annotated_text import annotated_text

# Função principal que contém a lógica do dashboard2
def main():
    st.title("Análises e Insights")
    st.write("""
        <hr style="margin: 5px 0;">
    Visando atender à requisição do cliente, elaboramos um dashboard para melhor visualização dos dados históricos de 
    preços do petróleo Brent. Com o apoio do relatório desenvolvido, foi possível realizar a análise exploratória dos dados, 
    a fim de compreender as principais métricas e suas oscilações.

    """, unsafe_allow_html=True) 

    st.image("assents/view1.png", caption="Figura 1 - Painel interativo com dados da série histórica de preços do Petróleo Brent")
    
    st.write("""
    A partir da observação do comportamento dos dados no tempo foram identificados os picos e vales, 
    períodos correspondentes a crises macroeconômicas e globais, bem como os fatores subjacentes a essas crises que 
    poderiam estar afetando as tendências e sazonalidades normais de variação de preços dessa _commodity_.

    """) 
    
    st.image("assents/view2.png", caption="Figura 2 - Série histórica de preços do Petróleo Brent") 

    st.write("""
    Os principais períodos de crise identificados, com reflexos visíveis na série histórica, foram: 
    - **Guerra do Golfo (2 de agosto de 1990 a 28 de fevereiro de 1991):** A invasão do Kuwait pelo Iraque em agosto de 1990 e a 
    subsequente Guerra do Golfo provocaram temor de escassez, pois ambos os países eram grandes produtores de petróleo. 
    O preço do petróleo subiu rapidamente, mas a crise foi relativamente curta, pois a produção foi retomada rapidamente após o fim do conflito.
             
    - **Crise Financeira de 2008 (1 de fevereiro de 2007 a 31 de dezembro de 2008):** A alta demanda global, especialmente da China, 
    e restrições de produção mantiveram os preços em alta. O barril de petróleo atingiu o recorde de 143,95 dólares em 15 de julho de 2008, 
    no entanto, a crise financeira global fez com que os preços caíssem abruptamente nos meses subsequentes. 
    O aumento extremo nos preços contribuiu para agravar a crise financeira de 2008, ao elevar os custos de transporte e produção, 
    contribuindo para o prolongamento do período crítico.
             
    - **Pandemia da Covid-19 (1 de março de 2020 a 31 de dezembro de 2020):** A pandemia de COVID-19 levou a uma queda acentuada na demanda por petróleo devido ao isolamento social, 
    restrições de viagem e menor atividade econômica. O preço do petróleo caiu para níveis históricos, 
    com o West Texas Intermediate (WTI) (referência dos EUA) chegando a valores negativos em abril de 2020. Foi uma crise sem precedentes, 
    pois a baixa demanda e os estoques elevados resultaram em problemas de armazenamento e oscilações drásticas.
             
    - **Guerra Rússia-Ucrânia (1 de fevereiro de 2022 a 31 de dezembro de 2022):** A invasão da Ucrânia pela Rússia gerou sanções contra a Rússia, 
    um dos maiores produtores de petróleo e gás natural, afetando o fornecimento global. Os preços do petróleo dispararam, 
    ultrapassando 100 dólares por barril e atingindo 133,18 dólares no pico em 8 de março de 2022. 
    O aumento dos preços de energia contribuiu para a alta inflação global e gerou incertezas econômicas, especialmente na Europa.

    Desde o primeiro registro da série, em 20 de maio de 1987 (U\$ 18,63), até o último valor registrado, 
    em 28 de outubro de 2024 (U\$ 71,87), o preço do produto variou mais de 285%. A aceleração na escalada dos valores, 
    bem como a intensificação das variações ocorreram próximo ao final do milênio.
    
    Com a virada dos anos 2000, ocorreu uma baixa nos investimentos e na capacidade de produção de
    petróleo devido aos preços baixos e baixa rentabilidade. Com o aumento da demanda, a oferta global encontrou dificuldades para acompanhar, 
    pressionando os preços para cima.
    
    Para além dessa questão, os anos 2000 foram marcados por conflitos em regiões produtoras de petróleo, especialmente no Oriente Médio. 
    A Guerra do Iraque em 2003, tensões no Irã, e a instabilidade em países como Venezuela e 
    Líbia contribuíram para aumentar as incertezas no fornecimento de petróleo, elevando os preços. Tais fatores tornaram o mercado de petróleo mais complexo,
    com preços reagindo não apenas à oferta e demanda, mas também a fatores geopolíticos, ambientais e de investimento. 
    O mercado passou a ser mais influenciado por eventos globais e regionais,
    o que explica a alta volatilidade e as flutuações significativas a partir deste período.
    
    Apesar das grandes oscilações resultantes dos períodos de crises e da variação de oferta e demanda, 
    ao longo dos anos, a tendência aparenta ser de um aumento gradual no preço do produto. 
    A fim de verificar esta hipótese e prover previsões mais robustas, foram estudados alguns modelos de predição, que serão descritos abaixo.
          
    """)       
    
    # Título da seção
    st.title("Modelos de Previsão")

    # Seção Modelo 1
    st.subheader("Modelo 1", divider="gray")
    st.write("""
    **Algoritmo utilizado:** Prophet  
    **Descrição do modelo:** 
    O modelo foi desenvolvido com ajuste de sazonalidade e imputação de períodos de crise para ajuste de tendência. 
    Optou-se pela realização de um recorte temporal na série a partir do ano de 2000 (01/01/2000), 
    devido à alteração identificada no comportamento da série a partir deste período. Os dados pregressos foram desconsiderados e descartados.  

    Para a definição dos períodos de treino e teste, foi utilizada a data de corte em 18/05/2023, 
    a fim de considerar o período de teste posterior à última crise identificada na série histórica, 
    por este não se tratar de um comportamento típico da série.  

    Após os ajustes realizados, foi obtida uma acurácia de aproximadamente 93%.
    """)
    
    st.subheader("Resultado")
    st.image("assents/view3.png", caption="Figura 3 - Gráfico de Resultados do modelo 1")

    # Resultados e métricas
    st.subheader("Métricas e Acurácia")
    st.markdown("""
    - **RMSE:** 6.87 
    - **MAE:** 5.52 
    - **MAPE:** 0.06 
    - **Accuracy:** 0.93
    """)

    # Seção Modelo 2
    st.subheader("Modelo 2", divider="gray")
    st.write("""
    **Algoritmo utilizado:** Prophet  
    **Descrição do modelo:**  O modelo de previsão de preços de petróleo foi desenvolvido usando dados diários desde 2000, 
    carregados de um arquivo CSV e pré-processados. A coluna "Data" foi convertida para o formato de data e a coluna "Preço" para tipo numérico. 
    A série foi filtrada para incluir apenas dias úteis e configurada com a data como índice.
    A divisão foi feita em 80% para treinamento e 20% para teste, utilizando o modelo Prophet com ajustes para sazonalidade diária, 
    semanal e anual, além de feriados nacionais brasileiros. Parâmetros como changepoint_prior_scale e intervalo de confiança de 95% foram configurados para melhorar as previsões. 
    O modelo gerou previsões para 1 ano.
    O desempenho foi avaliado com as métricas MAE, RMSE e uma métrica de acurácia, usando um conjunto de teste separado. 
    Validações cruzadas com um horizonte de 30 dias e frequência de 15 dias verificaram a robustez do modelo. 
    A análise dos resíduos permitiu identificar padrões não capturados. O modelo demonstrou bom desempenho com erros baixos e alta acurácia.
    """)

    st.subheader("Resultado")
    st.image("assents/view5.png", caption="Figura 4 - Gráfico de Resultados do modelo 2")

    # Resultados e métricas
    st.subheader("Métricas e Acurácia")
    st.markdown("""
    - **RMSE:** 24.63  
    - **MAE:** 20.92 
    - **Accuracy:** 71.75
    """)          
       
    # Seção Modelo 3
    st.subheader("Modelo 3", divider="gray")
    st.write("""
    **Algoritmo utilizado:** ARIMA  
    **Descrição do Modelo:** O modelo foi desenvolvido usando as datas dos anos de 2012 até 2014. A técnica de previsão escolhida foi ARIMA, 
    adequada para séries temporais, com o objetivo de analisar e prever os valores de preço de uma série histórica de dados anuais. 
    Primeiramente, os dados foram preparados e transformados, com a coluna Data configurada como índice da série e a coluna Preço convertida para o tipo numérico apropriado. 
    A série foi dividida em conjunto de treinamento e teste, sendo 80% para treinamento e 20% para teste.
    O modelo ARIMA foi ajustado com parâmetros p=1, d=1 e q=2, usado para gerar previsões para o período de teste. 
    Para avaliar o desempenho do modelo, a previsão foi estendida para um período futuro de 100 dias. A precisão do modelo foi avaliada por métricas como 
    RMSE, MAE, MAPE e uma métrica de precisão percentual, oferecendo uma visão detalhada do seu desempenho.       
    """)
    
    #Grafico
    st.image("assents/view4.jpeg", caption="Figura 5 - Gráfico de Resultados do modelo 3")
    # Resultados e métricas
    st.subheader("Métricas e Acurácia")
    st.markdown("""
    - **RMSE:** 6.45 
    - **MAE:** 4.32
    - **MAPE:** 4.42
    - **Accuracy:** 95.11
    """)

    # Comparação do resultado
    st.title("Análise de desempenho")
    st.write("""
        <hr style="margin: 5px 0;">
    A análise de desempenho dos modelos Prophet e ARIMA revelou as seguintes métricas de precisão:
    """, unsafe_allow_html=True)

    st.subheader("Modelo 1 (Prophet):")
    st.markdown("""
    - **RMSE (Root Mean Squared Error):** 6.88
    - **MAE (Mean Absolute Error):** 5.53
    - **MAPE (Mean Absolute Percentage Error):** 6.74%
    - **Accuracy:** <span style="background-color: lightyellow; padding: 0 4px 0 5px;">93.29%</span>
    """, unsafe_allow_html=True)

    st.subheader("Modelo 2 (Prophet):")
    st.markdown("""
    - **RMSE (Root Mean Squared Error):** 24.63
    - **MAE (Mean Absolute Error):** 20.92
    - **Accuracy:** <span style="background-color: lightyellow; padding: 0 4px 0 5px;">71.75%</span>
    """, unsafe_allow_html=True)

    st.subheader("Modelo 3 (ARIMA):")
    st.markdown("""
    - **RMSE (Root Mean Squared Error):** 6.45
    - **MAE (Mean Absolute Error):** 4.32
    - **MAPE (Mean Absolute Percentage Error):** 4.42%
    - **Accuracy:** <span style="background-color: lightyellow; padding: 0 4px 0 5px;">95.11%</span>
    """, unsafe_allow_html=True)

    
    st.title("Prophet x ARIMA")
    st.write("""
        <hr style="margin: 5px 0;">
    Este projeto foi desenvolvido utilizando duas ferramentas de Machine Learning: Prophet e ARIMA.
    Embora o ARIMA tenha demonstrado uma leve superioridade em termos de precisão, 
    com métricas como RMSE e Accuracy mais altas, optamos por utilizar o Prophet como base para o modelo final. 
    Essa escolha foi motivada por fatores como:
             
    - **Flexibilidade:** O Prophet lida bem com sazonalidades complexas e incorpora fatores externos de forma intuitiva.
             
    - **Facilidade de uso:** Sua interface e abordagem mais moderna tornam o ajuste de modelos mais rápido e eficiente.
             
    - **Manutenção futura:** O Prophet oferece maior facilidade para ajustes e implementações contínuas, algo valioso no contexto do cliente.
    
    Apesar do desempenho ligeiramente superior do ARIMA nas métricas, 
    o Prophet foi escolhido como a base do modelo de previsão devido à sua versatilidade e capacidade de atender a cenários futuros de maneira mais eficaz. 
    Essa decisão visa garantir que o cliente tenha à disposição uma solução robusta, escalável e alinhada com as necessidades de negócios.     
    """, unsafe_allow_html=True)

    st.title("Aplicação no Streamlit")
    st.write("""
        <hr style="margin: 5px 0;">
    O modelo desenvolvido conta com funcionalidades personalizáveis para realização de previsões a partir do modelo Prophet. 
    Na aplicação é possível definir o período para previsão, bem como outros argumentos para input no modelo, 
    retornando com as previsões em formato gráfico e as métricas relativas
     ao resultado do modelo rodado com os parâmetros definidos pelo cliente.
    """, unsafe_allow_html=True)   

if __name__ == "__main__":
    main()
