import os
import dashboard
import dashboard2
import sobre_nos
import streamlit as st

# Função para exibir imagem e legenda
def exibir_imagem_e_legenda(imagem_path, legenda_texto, largura_imagem=150):
    # Verifique se o arquivo existe antes de carregar
    if os.path.exists(imagem_path):
        st.image(imagem_path, width=largura_imagem)
        
        # Centraliza e formata a legenda em negrito
        st.markdown(
            f'<p style="text-align: left; font-weight: bold;">{legenda_texto}</p>', 
            unsafe_allow_html=True
        )
    else:
        st.error(f"A imagem '{imagem_path}' não foi encontrada!")

# Funções para cada página
def dashboard_previsao_petroleo():
    dashboard.main()  

def dashboard2_previsao_petroleo():
    dashboard2.main()  

def sobre_nos_previsao_petroleo():
    sobre_nos.main()  

st.set_page_config(
    page_title="Petro Insights", 
    page_icon="📊", 
)

# Definir a imagem e legenda para as páginas
imagem_path = "assents/fiap.png"
legenda = "Pós Tech"

# Exibir imagem e legenda nas páginas certas
paginas = {
    "Previsão do Preço do Petróleo Brent": dashboard_previsao_petroleo,
    "Análise do grupo": dashboard2_previsao_petroleo,
    "Sobre nós": sobre_nos_previsao_petroleo,
}

st.sidebar.title("Navegação")
pagina_selecionada = st.sidebar.selectbox("Selecione uma página", paginas.keys())

# Exibir a imagem e legenda apenas nas páginas que não são 'Sobre nós'
if pagina_selecionada != "Sobre nós":
    exibir_imagem_e_legenda(imagem_path, legenda)

# Chamar a função da página selecionada
paginas[pagina_selecionada]()
