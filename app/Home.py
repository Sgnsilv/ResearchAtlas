import streamlit as st

st.set_page_config(
    page_title="ResearchAtlas",
    page_icon="🔬",
    layout="wide"
)

st.title("ResearchAtlas")

st.subheader(
    "Explorando conexões entre Inteligência Artificial "
    "e diferentes áreas científicas"
)

st.info(
    "Projeto em desenvolvimento — "
    "Aprendizagem de Máquina e Clusterização"
)

st.markdown("""
### Sobre o projeto

O ResearchAtlas investiga padrões e comunidades
científicas interdisciplinares utilizando:

- Clusterização semântica
- Embeddings de textos científicos
- Análise de grafos de citações
- Visualização interativa de dados

**Fonte dos dados:** OpenAlex.
""")