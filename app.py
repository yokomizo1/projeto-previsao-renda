import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


model = joblib.load('best_rf_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.set_page_config(page_title="Previsão de Renda", page_icon="💰")

st.markdown("<h1 style='text-align: center; font-size: 3em;'>Previsão de Renda 💰</h1>",
            unsafe_allow_html=True)
st.write('Este aplicativo prevê a renda anual com base nas características inseridas.')

st.sidebar.title("Parâmetros de Entrada")

# inputs do usuário na barra lateral
tempo_emprego = st.sidebar.slider('Tempo de Emprego', 0, 48, 5)
sexo = st.sidebar.selectbox('Sexo', ['Masculino', 'Feminino'])
idade = st.sidebar.slider('Idade', 18, 78, 30)
educacao = st.sidebar.selectbox('Nível de Educação', [
                                'Primário', 'Secundário', 'Superior incompleto', 'Superior completo', 'Pós graduação'])
qtd_filhos = st.sidebar.slider('Quantidade de Filhos', 0, 10, 2)
qt_pessoas_residencia = st.sidebar.slider(
    'Quantidade de Pessoas na Residência', 1, 15, 3)
posse_de_veiculo = st.sidebar.selectbox('Possui Veículo?', ['Sim', 'Não'])
posse_de_imovel = st.sidebar.selectbox('Possui Imóvel?', ['Sim', 'Não'])
estado_civil = st.sidebar.selectbox(
    'Estado Civil', ['Solteiro', 'Casado', 'Separado', 'Divorciado', 'Viúvo', 'União'])
tipo_renda = st.sidebar.selectbox('Tipo de Renda', [
                                  'Assalariado', 'Autônomo', 'Empresário', 'Servidor público', 'Pensionista', 'Bolsa'])
tipo_residencia = st.sidebar.selectbox('Tipo de Residência', [
                                       'Casa', 'Aluguel', 'Estúdio', 'Com os pais', 'Comunitário', 'Governo'])

# entradas categóricas para os valores corretos
sexo_map = {'Masculino': 'M', 'Feminino': 'F'}
posse_de_veiculo_map = {'Sim': True, 'Não': False}
posse_de_imovel_map = {'Sim': True, 'Não': False}

# inputs do usuário
input_data = {
    'qtd_filhos': [qtd_filhos],
    'idade': [idade],
    'tempo_emprego': [tempo_emprego],
    'qt_pessoas_residencia': [qt_pessoas_residencia],
    'idade_ao_quadrado': [idade ** 2],
    'sexo': [sexo_map[sexo]],
    'posse_de_veiculo': [posse_de_veiculo_map[posse_de_veiculo]],
    'posse_de_imovel': [posse_de_imovel_map[posse_de_imovel]],
    'tipo_renda': [tipo_renda],
    'educacao': [educacao],
    'estado_civil': [estado_civil],
    'tipo_residencia': [tipo_residencia]
}

input_df = pd.DataFrame(input_data)

# botao
if st.sidebar.button('Prever Renda', key="predict_button"):
    try:
        input_processed = preprocessor.transform(input_df)

        prediction = model.predict(input_processed)

        st.markdown(
            f"<h2 style='text-align: center; font-size: 2em;'>Renda anual prevista: R$ {prediction[0]:,.2f}</h2>", unsafe_allow_html=True)

        st.subheader("Gráficos Relevantes:")

        # Distribuição de Renda
        st.write("Distribuição de Renda")
        fig, ax = plt.subplots()
        sns.histplot(prediction, kde=True, ax=ax)
        st.pyplot(fig)

        # Idade vs. Renda
        st.write("Idade vs. Renda")
        fig, ax = plt.subplots()
        sns.scatterplot(x=input_df['idade'], y=prediction, ax=ax)
        ax.set_xlabel('Idade')
        ax.set_ylabel('Renda Prevista')
        st.pyplot(fig)

        # Tempo de Emprego vs. Renda
        st.write("Tempo de Emprego vs. Renda")
        fig, ax = plt.subplots()
        sns.scatterplot(x=input_df['tempo_emprego'], y=prediction, ax=ax)
        ax.set_xlabel('Tempo de Emprego')
        ax.set_ylabel('Renda Prevista')
        st.pyplot(fig)

    except Exception as e:
        st.write(f"Erro ao fazer a previsão: {str(e)}")

st.markdown("""
    <style>
        .css-1offfwp {
            font-size: 1.2em !important;
            height: 3em !important;
        }
    </style>
""", unsafe_allow_html=True)
