import os
import streamlit as st

def main(): 
    """
    Função principal que exibe a página "Sobre Nós".
    """
    # Função para exibir imagem e opcionalmente legenda
    def exibir_imagem(imagem_path, legenda_texto=None, largura_imagem=150):
        """
        Exibe uma imagem com uma legenda opcional.
        """
        # Verifique se o arquivo existe antes de carregar
        if os.path.exists(imagem_path):

            st.image(imagem_path, width=largura_imagem)  # Ajusta a largura da imagem conforme necessário
            
            # Se a legenda for fornecida, exibe-a
            if legenda_texto:
                st.markdown(
                    f'<p style="text-align: left; font-weight: bold;">{legenda_texto}</p>', 
                    unsafe_allow_html=True
                )
        else:
            st.error(f"A imagem '{imagem_path}' não foi encontrada!")

    # Exibir imagem sem legenda
    imagem_path = "assents/view6.png"
    exibir_imagem(imagem_path, "Petro Insights")  # Não passamos a legenda, pois é opcional

    # Exibir conteúdo da página "Sobre Nós"
    st.title("Sobre Nós")

    st.write("""
        <hr style="margin: 5px 0;">
    Somos uma consultoria consolidada há anos no mercado, 
    especializada em fornecer análises estratégicas e suporte na tomada de decisões para o setor de petróleo Brent.

    Capacitamos nossos clientes a tomar decisões informadas em um mercado global dinâmico e desafiador, 
    com base em dados precisos e insights valiosos.

    Nossa missão é descomplicar a complexidade do mercado, oferecendo análises claras, 
    projeções confiáveis e estratégias eficazes que geram resultados concretos. 
    Com uma equipe formada por especialistas em análise de dados, economia e energia, 
    estamos comprometidos em ser sua principal fonte de confiança, ajudando a transformar desafios em oportunidades.
    """, unsafe_allow_html=True)

    st.header("Desafio")

    st.write("""
        <hr style="margin: 5px 0;">
    Geramos insights relevantes sobre as variações do mercado de petróleo, 
    analisando fatores econômicos, geopolíticos e tendências de demanda global. 
    Utilizamos modelos avançados de análise de dados e 
    algoritmos preditivos para realizar previsões diárias precisas sobre os preços do petróleo, 
    ajudando nossos clientes a tomar decisões estratégicas em um mercado dinâmico e volátil.
             
    """, unsafe_allow_html=True)

    exibir_creditos()

def exibir_creditos():
    """
    Exibe os créditos do projeto e permite que o usuário adicione uma string Markdown.
    """
    st.subheader(":green[Desenvolvedores]", divider="green")
    st.write("Este projeto foi desenvolvido por:")
    st.write("- **Nome da Desenvolvedora:** Clara Crizio de Araujo Torres")
    st.write("- **Nome da Desenvolvedora:** Isabela de Jesus Santos")
    st.write("- **Nome do Desenvolvedor:** Willian C. Rodrigues")
    st.write("**Turma:** Grupo 17 / **Módulo:** Data Viz and Production Models / FIAP - Pós Tech")

if __name__ == "__main__":
    main()
