# Petro Insights - Previsão e Análise do Preço do Petróleo Brent

## 🚀 Sobre o Projeto

**Petro Insights** é uma consultoria especializada em análises preditivas e estratégicas para o setor de petróleo Brent. Criamos um dashboard interativo para análise de dados históricos do preço do petróleo, com insights baseados em fatores econômicos, geopolíticos e crises globais. Nosso objetivo é fornecer previsões precisas usando modelos de Machine Learning e apoiar a tomada de decisões para um mercado dinâmico e volátil.

### 🎯 Desafio

O desafio consistiu em desenvolver uma solução que unisse análise de dados históricos e previsões futuras dos preços do petróleo, integrando insights relevantes com base em dados concretos. Para isso, foi criado um modelo de Machine Learning utilizando **Séries Temporais** para previsão diária do preço do petróleo. A solução foi construída e está disponível em produção via **Streamlit**.

### 💡 O que foi desenvolvido:

- **Dashboard interativo**: Utilizando o Streamlit, foi construído um dashboard para visualização de dados históricos e insights.
- **Insights de mercado**: A análise envolveu fatores como crises econômicas, geopolítica, e tendências de demanda global por energia.
- **Modelo de Previsão**: Desenvolvemos e treinamos modelos de séries temporais (Prophet e ARIMA) para prever os preços futuros do petróleo.
- **Deploy em produção**: A aplicação foi disponibilizada em produção através da plataforma Heroku.

## 🌐 Como Usar

### 1. **Clone este repositório**:

```bash
git clone https://github.com/wylli-wylli/petro_insights.git
cd petro_insights
2. Instale as dependências:
Este projeto requer Python 3.8 ou superior. Você pode criar um ambiente virtual e instalar as dependências necessárias com o seguinte comando:

bash
Copiar código
pip install -r requirements.txt
3. Execute o aplicativo:
Para rodar o dashboard localmente, basta executar o seguinte comando:

bash
Copiar código
streamlit run app.py
Isso abrirá o dashboard em seu navegador.

🛠 Tecnologias Utilizadas
Este projeto faz uso das seguintes tecnologias e bibliotecas:

Streamlit - Framework utilizado para criação do dashboard interativo.
NumPy - Biblioteca fundamental para computação científica e manipulação de arrays.
Prophet - Algoritmo de séries temporais utilizado para previsões de preços.
ARIMA - Modelo de séries temporais para previsão de dados futuros.
Pandas - Biblioteca para manipulação e análise de dados.
Seaborn - Biblioteca para visualização de dados estatísticos.
Matplotlib - Biblioteca de visualização gráfica.
Heroku - Plataforma utilizada para deploy da aplicação em produção.
📊 Dashboard
O dashboard interativo desenvolvido permite a visualização e análise do histórico de preços do petróleo Brent, além de fornecer previsões diárias para o futuro do preço com base nos modelos de Machine Learning. Acesse a versão em produção do dashboard clicando no link abaixo:

🔗 Acessar o Dashboard no Heroku

🔮 Modelo de Machine Learning
Prophet:
O modelo Prophet foi utilizado para previsão de séries temporais, focando na previsão do preço do petróleo Brent. A utilização de Prophet permite modelar de forma eficiente a sazonalidade e tendências nos dados de séries temporais.

Mais sobre Prophet: Link para a documentação
ARIMA:
O modelo ARIMA (AutoRegressive Integrated Moving Average) também foi explorado para prever os preços futuros, utilizando o histórico de dados. ARIMA é amplamente utilizado para dados que apresentam padrões de tendência e sazonalidade.

Mais sobre ARIMA: Link para a documentação
📝 Contribuições
Este projeto foi desenvolvido por:

Clara Crizio de Araujo Torres
Isabela de Jesus Santos
Willian C. Rodrigues
Agradecemos a contribuição e o esforço de todos os envolvidos no desenvolvimento do Petro Insights.

🚀 Deploy
O modelo foi implantado em produção usando a plataforma Heroku, garantindo acessibilidade e escalabilidade para os usuários interagirem com a aplicação de forma rápida e eficiente.

Documentação do Heroku
📈 Resultados Esperados
O modelo de previsão oferece uma visão precisa do comportamento futuro dos preços do petróleo, o que pode ser fundamental para ajudar empresas e investidores a tomarem decisões informadas em um mercado global volátil.
