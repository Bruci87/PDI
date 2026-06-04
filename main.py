import cv2
import os
import json
import numpy as np

# =========================================================================
# 1. FUNÇÕES DE CARREGAMENTO E GERAÇÃO DE RUÍDO
# =========================================================================

def carregar_dados_originais(caminho_img, caminho_json):
    """
    Carrega a imagem original em RGB e o arquivo JSON correspondente.
    """
    img_bgr = cv2.imread(caminho_img, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem em: {caminho_img}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    with open(caminho_json, 'r', encoding='utf-8') as f:
        dados_gt = json.load(f)
        
    return img_rgb, dados_gt

def adicionar_ruido_gaussiano_rgb(imagem, desvio_padrao, semente=42):
    """
    Aplica ruído gaussiano aditivo n_c ~ N(0, sigma^2) 
    independentemente em cada um dos três canais (R, G, B).
    """
    rng = np.random.default_rng(semente)
    ruido = rng.normal(0, desvio_padrao, imagem.shape)
    
    img_degradada = imagem.astype(np.float64) + ruido
    img_degradada = np.clip(img_degradada, 0, 255).astype(np.uint8)
    return img_degradada

def adicionar_ruido_sal_e_pimenta_rgb(imagem, densidade, semente=42):
    """
    Seleciona aleatoriamente uma fração (p) dos pixels.
    Metade vira Sal [255, 255, 255] e metade vira Pimenta [0, 0, 0].
    """
    img_degradada = imagem.copy()
    altura, largura, canais = imagem.shape
    total_pixels = altura * largura
    
    rng = np.random.default_rng(semente)
    num_pixels_ruido = int(densidade * total_pixels)
    indices_corrompidos = rng.choice(total_pixels, size=num_pixels_ruido, replace=False)
    
    metade = num_pixels_ruido // 2
    indices_sal = indices_corrompidos[:metade]
    indices_pimenta = indices_corrompidos[metade:]
    
    linhas_sal, colunas_sal = np.unravel_index(indices_sal, (altura, largura))
    linhas_pimenta, colunas_pimenta = np.unravel_index(indices_pimenta, (altura, largura))
    
    img_degradada[linhas_sal, colunas_sal] = [255, 255, 255]
    img_degradada[linhas_pimenta, colunas_pimenta] = [0, 0, 0]
    
    return img_degradada

def gerar_dataset_experimental(diretorio_entrada, diretorio_saida, semente_base=42):
    """
    Varre o diretório de entrada, gera as 4 versões degradadas de cada imagem
    e duplica os respectivos arquivos JSON com os nomes corrigidos.
    """
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
        
    # Listar todos os arquivos de imagem no diretório de entrada
    arquivos = [f for f in os.listdir(diretorio_entrada) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not arquivos:
        print(f"Aviso: Nenhuma imagem encontrada na pasta '{diretorio_entrada}'.")
        return

    # Configurações solicitadas pelo enunciado
    config_degradacoes = {
        "gauss_baixo":  {"tipo": "gauss", "param": 5},
        "gauss_medio":  {"tipo": "gauss", "param": 15},
        "sp_baixo":     {"tipo": "sp",    "param": 0.01},
        "sp_medio":     {"tipo": "sp",    "param": 0.05}
    }
    
    print(f"Iniciando o processamento de {len(arquivos)} imagens originais...")
    print("-" * 50)

    for arquivo_img in arquivos:
        nome_base, extensao = os.path.splitext(arquivo_img)
        arquivo_json = f"{nome_base}.json"
        
        caminho_img = os.path.join(diretorio_entrada, arquivo_img)
        caminho_json = os.path.join(diretorio_entrada, arquivo_json)
        
        # Garante que o par Imagem + JSON existe antes de prosseguir
        if not os.path.exists(caminho_json):
            print(f"Aviso: Ground truth (JSON) não encontrado para {arquivo_img}. Pulando...")
            continue
            
        print(f"Processando: {arquivo_img}...")
        
        # Carrega os dados originais
        img_rgb, dados_gt = carregar_dados_originais(caminho_img, caminho_json)
        
        # Copia o original para o diretório de saída para centralizar o experimento
        cv2.imwrite(os.path.join(diretorio_saida, arquivo_img), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        with open(os.path.join(diretorio_saida, arquivo_json), 'w', encoding='utf-8') as f:
            json.dump(dados_gt, f, indent=4)
            
        # Gera as 4 variações artificiais
        for sufixo, config in config_degradacoes.items():
            if config["tipo"] == "gauss":
                img_proc = adicionar_ruido_gaussiano_rgb(img_rgb, config["param"], semente=semente_base)
            elif config["tipo"] == "sp":
                img_proc = adicionar_ruido_sal_e_pimenta_rgb(img_rgb, config["param"], semente=semente_base)
                
            # Mantém a extensão original da imagem salva
            novo_nome_img = f"{nome_base}_{sufixo}{extensao}"
            novo_nome_json = f"{nome_base}_{sufixo}.json"
            
            # Salva a imagem degradada (convertendo de volta para BGR antes de gravar)
            cv2.imwrite(os.path.join(diretorio_saida, novo_nome_img), cv2.cvtColor(img_proc, cv2.COLOR_RGB2BGR))
            
            # Duplica integralmente o JSON de referência para a nova imagem
            with open(os.path.join(diretorio_saida, novo_nome_json), 'w', encoding='utf-8') as f:
                json.dump(dados_gt, f, indent=4)
                
    print("-" * 50)
    print(f"Sucesso! Dataset experimental gerado e unificado em: '{diretorio_saida}'")


# =========================================================================
# 2. BLOCO PRINCIPAL DE EXECUÇÃO E CARREGAMENTO EM MEMÓRIA (ARRAY)
# =========================================================================
if __name__ == "__main__":
    # 2.1 Definir pastas de entrada e saída do experimento
    pasta_originais = "imagens"
    pasta_saida_degradadas = "imagens_experimentais"
    
    # 2.2 Gerar o dataset de degradadas e clonar os arquivos JSON
    gerar_dataset_experimental(pasta_originais, pasta_saida_degradadas)
    print("\n" + "="*50 + "\n")
    
    # 2.3 Carregar os resultados redimensionados para a memória RAM de forma segura
    largura_maxima = 800  
    imagens = []
    
    print(f"Carregando e otimizando imagens da pasta '{pasta_saida_degradadas}' para a RAM...")
    
    for f in sorted(os.listdir(pasta_saida_degradadas)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            caminho_imagem = os.path.join(pasta_saida_degradadas, f)
            img = cv2.imread(caminho_imagem)
            
            if img is not None:
                # Redimensiona proporcionalmente para economizar memória
                altura, largura = img.shape[:2]
                if largura > largura_maxima:
                    proporcao = largura_maxima / float(largura)
                    nova_altura = int(altura * proporcao)
                    img = cv2.resize(img, (largura_maxima, nova_altura), interpolation=cv2.INTER_AREA)
                
                # Adiciona no array em BGR de forma eficiente
                imagens.append(img)

    # Notificação de encerramento do processo unificado
    print("-" * 50)
    if imagens:
        print(f"As imagens foram geradas e otimizadas com sucesso!")
        print(f"Total de imagens carregadas no array para os próximos passos: {len(imagens)}")
    else:
        print("Aviso: Nenhuma imagem válida foi carregada no array.")