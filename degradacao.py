import os
import json
import cv2
import numpy as np

def carregar_dados_originais(caminho_img, caminho_json):
    """
    Carrega a imagem original mantendo a estrutura RGB de 8 bits
    e lê o arquivo de anotação correspondente (Ground Truth).
    """
    # OpenCV por padrão lê em BGR. Vamos converter para RGB para seguir a risca o enunciado
    img_bgr = cv2.imread(caminho_img, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem em: {caminho_img}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    with open(caminho_json, 'r') as f:
        dados_gt = json.load(f)
        
    return img_rgb, dados_gt

def adicionar_ruido_gaussiano_rgb(imagem, desvio_padrao, semente=42):
    """
    Aplica ruído gaussiano aditivo n_c ~ N(0, sigma^2) 
    independentemente em cada um dos três canais (R, G, B).
    """
    # Configura a semente aleatória local para garantir reprodutibilidade
    rng = np.random.default_rng(semente)
    
    # O ruído deve ter o mesmo formato tridimensional da imagem (Altura, Largura, 3 Canais)
    ruido = rng.normal(0, desvio_padrao, imagem.shape)
    
    # Somamos o ruído à imagem convertida para float para evitar estouro de representação (uint8)
    img_degradada = imagem.astype(np.float64) + ruido
    
    # Clipamos os valores para garantir o intervalo [0, 255] e retornamos para uint8
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
    
    # Quantidade total de pixels que serão corrompidos
    num_pixels_ruido = int(densidade * total_pixels)
    
    # Sorteia os índices lineares dos pixels que sofrerão a degradação
    indices_corrompidos = rng.choice(total_pixels, size=num_pixels_ruido, replace=False)
    
    # Divide metade para Sal e metade para Pimenta
    metade = num_pixels_ruido // 2
    indices_sal = indices_corrompidos[:metade]
    indices_pimenta = indices_corrompidos[metade:]
    
    # Convertendo os índices lineares de volta para coordenadas 2D (linha, coluna)
    linhas_sal, colunas_sal = np.unravel_index(indices_sal, (altura, largura))
    linhas_pimenta, colunas_pimenta = np.unravel_index(indices_pimenta, (altura, largura))
    
    # Aplica os vetores RGB de forma idêntica em todas as três camadas do pixel selecionado
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
        
    # Listar todos os arquivos de imagem no diretório
    arquivos = [f for f in os.listdir(diretorio_entrada) if f.endswith(('.jpg', '.png'))]
    
    # Configurações solicitadas pelo enunciado
    config_degradacoes = {
        "gauss_baixo":  {"tipo": "gauss", "param": 5},
        "gauss_medio":  {"tipo": "gauss", "param": 15},
        "sp_baixo":     {"tipo": "sp",    "param": 0.01},
        "sp_medio":     {"tipo": "sp",    "param": 0.05}
    }
    
    for arquivo_img in arquivos:
        nome_base = os.path.splitext(arquivo_img)[0]
        arquivo_json = f"{nome_base}.json"
        
        caminho_img = os.path.join(diretorio_entrada, arquivo_img)
        caminho_json = os.path.join(diretorio_entrada, arquivo_json)
        
        # Garante que o par Imagem + JSON existe antes de prosseguir
        if not os.path.exists(caminho_json):
            print(f"Aviso: Ground truth não encontrado para {arquivo_img}. Pulando...")
            continue
            
        # Carrega os dados originais
        img_rgb, dados_gt = carregar_dados_originais(caminho_img, caminho_json)
        
        # Copia o original para o diretório de saída para centralizar o experimento
        cv2.imwrite(os.path.join(diretorio_saida, arquivo_img), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        with open(os.path.join(diretorio_saida, arquivo_json), 'w') as f:
            json.dump(dados_gt, f, indent=4)
            
        # Gera as 4 variações artificiais
        for sufixo, config in config_degradacoes.items():
            if config["tipo"] == "gauss":
                img_proc = adicionar_ruido_gaussiano_rgb(img_rgb, config["param"], semente=semente_base)
            elif config["tipo"] == "sp":
                img_proc = adicionar_ruido_sal_e_pimenta_rgb(img_rgb, config["param"], semente=semente_base)
                
            # Define os novos nomes com os sufixos exigidos
            novo_nome_img = f"{nome_base}_{sufixo}.jpg"
            novo_nome_json = f"{nome_base}_{sufixo}.json"
            
            # Salva a imagem (convertendo de volta para BGR antes de gravar em disco)
            cv2.imwrite(os.path.join(diretorio_saida, novo_nome_img), cv2.cvtColor(img_proc, cv2.COLOR_RGB2BGR))
            
            # Duplica integralmente o JSON de referência
            with open(os.path.join(diretorio_saida, novo_nome_json), 'w') as f:
                json.dump(dados_gt, f, indent=4)
                
    print(f"Sucesso! Dataset experimental gerado e unificado em: '{diretorio_saida}'")