# Desgaste dental # Di -> 358 (1.51%)
# Dente impactado # I, M3i -> 5 (0.02%) e 310 (1.3%)
import os
import json
import pandas as pd
import shutil

diretorio_origem = 'json/'
diretorio_destino = 'dentes_impactados_e_desgastados/'

diretorio_origem_imagem = 'imagens/'
diretorio_destino_imagem = 'implante-img/'

arquivos = os.listdir(diretorio_origem)

dados = []
consulta_arquivo = []

for arquivo in arquivos:
    if arquivo.endswith('.json'):
        with open(os.path.join(diretorio_origem, arquivo), 'r') as f:
            conteudo = json.load(f)
            
            for annotation in conteudo['annotations']:
                for block in annotation['blocks']:
                    estados = block['states']
                    
                    dados.extend(estados)
                    consulta_arquivo.extend([{'Nome_Arquivo': arquivo, 'State': estado} for estado in estados])

df = pd.DataFrame(dados, columns=['States'])

print("dl.acm.org", "ieee", "pubmed", "web of science")

# Calcula a porcentagem de cada state
porcentagem = df['States'].value_counts(normalize=True) * 100

# Calcula a quantidade em número de cada state
quantidade = df['States'].value_counts()

print("Porcentagem de cada state:")
print(porcentagem)
print("\nQuantidade em número de cada state:")
print(quantidade)

df_file = pd.DataFrame(consulta_arquivo)
# Filtra os nomes dos arquivos que possuem 'Di', 'I' ou 'M3i' como state
nomes_arquivos_filtrados = df_file[df_file['State'].isin(['Im'])]['Nome_Arquivo'].unique()

print("Nomes dos arquivos que possuem 'Di', 'I' ou 'M3i' como state:")
print(nomes_arquivos_filtrados)

# 287 arquivos até o momento
for nome_arquivo in nomes_arquivos_filtrados:
    shutil.copy(os.path.join(diretorio_origem, nome_arquivo), os.path.join(diretorio_destino, nome_arquivo))

for nome_arquivo in nomes_arquivos_filtrados:
    nome_arquivo = nome_arquivo.replace(".json", ".jpg")
    shutil.copy(os.path.join(diretorio_origem_imagem, nome_arquivo), os.path.join(diretorio_destino_imagem, nome_arquivo))