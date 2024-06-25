import os
import pandas as pd

# Diretorio principal e pastas
diretorio = '../LSTM'
pastas = ['result_v2_3', 'result_v2_6', 'result_v2_12', 'result_v2_24', 'result_v2_36']
arquivo = 'lstm_results.xlsx'

# Carregar, remover duplicatas e salvar os arquivos
for pasta in pastas:
    caminho_arquivo = os.path.join(diretorio, pasta, arquivo)
    # Carregar o arquivo Excel
    df = pd.read_excel(caminho_arquivo)
    # Remover duplicatas
    df = df.drop_duplicates()
    # Salvar o arquivo novamente
    df.to_excel(caminho_arquivo, index=False)