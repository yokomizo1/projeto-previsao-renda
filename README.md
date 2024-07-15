# Projeto de Previsão de Renda

Este projeto tem como objetivo prever a renda de indivíduos com base em diversos fatores socioeconômicos usando diferentes modelos de machine learning.

## Estrutura do Projeto

1. **Análise Exploratório de Dados**:
   - Distribuições univariadas.
   - Análises bivariadas e de correlação.

2. **Preparação dos Dados**:
   - Tratamento de valores ausentes.
   - Transformações e codificação.
   - Divisão em treino e teste.

3. **Modelagem e Avaliação**:
   - Regressão Linear.
   - Random Forest.
   - XGBoost.
   - Comparação de métricas de desempenho.

4. **Aplicativo Streamlit**:
   - Interface para entrada de parâmetros.
   - Previsão de renda baseada nos parâmetros inseridos.

## Como Usar

1. Clone este repositório.
   ```bash
   git clone <URL_DO_REPOSITORIO>

2. Navegue até o diretório do projeto.
    cd nome_do_projeto

3. Instale as dependências.
    pip install -r requirements.txt

4. Execute o aplicativo Streamlit.
    streamlit run app.py

## Estrutura dos Arquivos

1. data/previsao_de_renda.csv: Conjunto de dados utilizado no projeto.
2. models/best_rf_model.pkl: Modelo Random Forest treinado.
3. models/preprocessor.pkl: Objeto de pré-processamento.
4. reports/relatorio_exploratorio.html: Relatório exploratório em HTML.
5. previsao_renda.ipynb: Notebook Jupyter com a análise e modelagem.
6. app.py: Código do aplicativo Streamlit.
7. README.md: Documentação do projeto.

## Licença

Este projeto está licenciado sob a licença MIT.