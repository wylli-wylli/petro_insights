# Preto Insights - Análise e Previsão de Preço do Petróleo Brent 📊

## Sobre Nós

**Preto Insights** é uma consultoria especializada em fornecer análises estratégicas e suporte na tomada de decisões para o mercado de petróleo Brent. Capacitando nossos clientes a entender e navegar em um ambiente global dinâmico, com base em dados precisos e insights acionáveis. Nosso objetivo é transformar desafios complexos em oportunidades por meio de análises claras e projeções confiáveis.

## Desafio

O desafio consistiu em criar um **dashboard interativo** para apresentar insights sobre as variações do preço do petróleo Brent, com base em dados históricos. A partir de uma base de dados com duas colunas: **data** e **preço**, buscamos analisar fatores geopolíticos, crises econômicas, e a demanda global por energia para fornecer insights valiosos.

Além disso, foi desenvolvido um modelo de **Machine Learning** para previsão diária do preço do petróleo, utilizando **análise de séries temporais**. O projeto foi implementado com foco na produção de um **MVP** que foi publicado utilizando **Streamlit**.

## Funcionalidades

- **Dashboard Interativo**: Visualização de dados históricos e insights sobre o comportamento do preço do petróleo.
- **Forecasting do Preço do Petróleo**: Previsão diária do preço do petróleo utilizando modelos de **Machine Learning** (Prophet e ARIMA).
- **Análises**: 4 insights detalhados sobre os fatores que impactam o preço do petróleo, como crises econômicas, eventos geopolíticos e tendências globais de demanda.
- **Deployment**: O modelo em produção foi disponibilizado via **Streamlit** e hospedado no **Heroku**.

## Tecnologias Utilizadas

Neste projeto, foram utilizadas as seguintes tecnologias:

- [**Streamlit**](https://streamlit.io/): Framework de código aberto para criação de dashboards interativos e visualizações.
- [**NumPy**](https://numpy.org/): Biblioteca para manipulação de arrays multidimensionais e operações matemáticas em Python.
- [**Pandas**](https://pandas.pydata.org/): Biblioteca fundamental para análise de dados e manipulação de DataFrames.
- [**Prophet**](https://facebook.github.io/prophet/): Modelo de previsão de séries temporais desenvolvido pelo Facebook, utilizado para previsão do preço do petróleo.
- [**ARIMA**](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html): Modelo estatístico de séries temporais usado para modelagem e previsão.
- [**Seaborn**](https://seaborn.pydata.org/): Biblioteca de visualização de dados baseada no Matplotlib, utilizada para criar gráficos informativos e de alta qualidade.
- [**Matplotlib**](https://matplotlib.org/): Biblioteca de visualização em 2D para criar gráficos estáticos, interativos e animados.
- [**Heroku**](https://www.heroku.com/): Plataforma como serviço (PaaS) para deploy e hospedagem de aplicações, utilizado para disponibilizar o modelo em produção.

## Estrutura do Projeto

1. **Data Preprocessing**: Limpeza e preparação dos dados históricos de preços do petróleo.
2. **Modelos de Previsão**:
   - Utilização de **ARIMA** e **Prophet** para prever os preços futuros.
3. **Dashboard Interativo**: Desenvolvimento de visualizações interativas com **Streamlit** para exibir insights e previsões.
4. **Deployment**: Publicação do modelo preditivo utilizando **Heroku**.

## Como Rodar o Projeto

### Requisitos

- Python 3.7 ou superior
- Bibliotecas: `numpy`, `pandas`, `matplotlib`, `seaborn`, `prophet`, `statsmodels`, `streamlit`

### Passos

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu_usuario/preto-insights.git
   cd preto-insights

2. Clone este repositório:
   ```bash
   pip install -r requirements.txt

3. Clone este repositório:
   ```bash
   streamlit run app.py

4. Acesse o dashboard interativo no seu navegador localmente.

### Deploy
O projeto está disponível em produção via **Heroku**. Para acessar, clique no link abaixo:

[**Visite o Dashboard no Heroku**](https://petro-insights-c1c2ea8bf549.herokuapp.com/)

### Contribuidores
Este projeto foi desenvolvido por:

Clara Crizio de Araujo Torres.<br>Isabela de Jesus Santos.<br>Willian C. Rodrigues

<div align="center">
  Desenvolvido com muito carinho 🧩
</div>
