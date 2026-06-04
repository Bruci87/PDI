import os
from degradacao import gerar_dataset_experimental

def main():
    # Defina o caminho para a pasta onde estão as imagens originais do AGAR
    # (Ajuste o nome da pasta se a sua for diferente)
    diretorio_originais = "./imagens" 
    
    # Defina a pasta de destino onde o seu dataset completo de experimentos será gerado
    diretorio_destino = "./imagens_experimentais"
    
    # Validação simples para te ajudar caso esqueça de colocar as imagens na pasta
    if not os.path.exists(diretorio_originais):
        print(f"❌ Erro: A pasta de origem '{diretorio_originais}' não foi encontrada.")
        print("Por favor, crie a pasta e coloque algumas imagens do AGAR (.jpg e .json) dentro dela.")
        return
        
    print("🚀 Iniciando a geração das degradações artificiais...")
    print("Isso pode levar alguns segundos dependendo da quantidade de imagens...")
    
    # Chama a função que automatiza tudo (Semente padrão = 42)
    gerar_dataset_experimental(diretorio_originais, diretorio_destino, semente_base=42)
    
    # Verificação final para conferir se os arquivos nasceram na pasta nova
    if os.path.exists(diretorio_destino):
        total_arquivos = len(os.listdir(diretorio_destino))
        print(f"📊 Verificação: {total_arquivos} arquivos gerados em '{diretorio_destino}'")
        print("✨ Etapa de degradação concluída com sucesso e perfeitamente replicada!")

if __name__ == "__main__":
    main()