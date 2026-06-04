import cv2
import numpy as np

def pipeline_1_hsv_clahe(img_rgb):
    """
    Pipeline 1: Foco em Processamento de Histograma (Luminosidade) + Modelo de Cor.
    Muito eficiente para variações globais de iluminação e ruído Gaussiano.
    
    Categorias do Item 4 atendidas:
      - Processamento de Histogramas (CLAHE)
      - Filtros de Suavização (Gaussiano)
    """
    # 1. Conversão de Espaço de Cor: RGB para HSV
    # Separa a informação cromática (H, S) da luminosidade/brilho (V).
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)
    
    # 2. Processamento de Histograma: CLAHE no canal V
    # Equaliza localmente o contraste sem distorcer as cores reais do ágar.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_equalizado = clahe.apply(v)
    
    # Reconstrói a imagem HSV com a luminosidade corrigida
    img_hsv_corrigida = cv2.merge([h, s, v_equalizado])
    
    # 3. Filtro de Suavização: Gaussiano
    # Reduz o impacto do ruído aditivo simulado (Gaussiano Baixo/Médio).
    img_hsv_suave = cv2.GaussianBlur(img_hsv_corrigida, (5, 5), 0)
    
    # 4. Segmentação por Limiarização Multicanal (Modelo de Cor)
    # Subimos o limite inferior de Saturação (S=65) e Brilho (V=90).
    # Isso faz o fundo homogêneo e opaco do leite cair para PRETO (0) naturalmente.
    limite_inferior = np.array([0, 65, 90])
    limite_superior = np.array([180, 255, 255])
    mascara_binaria = cv2.inRange(img_hsv_suave, limite_inferior, limite_superior)
    
    # 5. Operação Morfológica: Abertura (Opening) Pesada com Kernel Elíptico
    # Elemento estruturante 9x9 varre a imagem eliminando qualquer ruído, 
    # reflexo de borda ou poeira menor do que o tamanho mínimo de uma colônia real.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mascara_final = cv2.morphologyEx(mascara_binaria, cv2.MORPH_OPEN, kernel)
    
    return mascara_final


def pipeline_2_restauracao_colorimetria(img_rgb):
    """
    Pipeline 2: Foco em Filtros de Restauração + Segmentação por Histograma Invertido.
    Desenvolvido cirurgicamente para aniquilar o ruído Sal e Pimenta.
    
    Categorias do Item 4 atendidas:
      - Filtros de Restauração (Mediana)
      - Análise Estatística de Componentes (Filtragem por Área)
    """
    # 1. Filtro de Restauração: Mediana no RGB
    # Substitui cada pixel pela mediana da vizinhança, extirpando o ruído Sal e Pimenta.
    img_restaurada = cv2.medianBlur(img_rgb, 5)
    
    # 2. Conversão para Escala de Cinza e Limiarização Global Dinâmica (Otsu Invertido)
    # O THRESH_BINARY_INV garante que o fundo volumoso (leite) vire PRETO (0) 
    # e as colônias fiquem BRANCAS (255).
    gray = cv2.cvtColor(img_restaurada, cv2.COLOR_RGB2GRAY)
    _, mascara_binaria = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Filtragem Estrita por Área (Análise de Componentes Conectados)
    # Como não temos uma máscara circular, filtramos os objetos pelo seu tamanho em pixels.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara_binaria, connectivity=8)
    mascara_filtrada = np.zeros_like(mascara_binaria)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # FILTRO DINÂMICO DE TAMANHO:
        # Rejeita sujeiras microscópicas residuais (área < 150 pixels)
        # Rejeita o anel gigante de acrílico da borda da placa de Petri (área > 20000 pixels)
        if 150 < area < 20000:
            mascara_filtrada[labels == i] = 255
            
    # 4. Refinamento Morfológico Conjugado (Fechamento seguido de Abertura)
    # O fechamento une partes separadas de colônias e a abertura arredonda os contornos.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_final = cv2.morphologyEx(mascara_filtrada, cv2.MORPH_CLOSE, kernel)
    mascara_final = cv2.morphologyEx(mascara_final, cv2.MORPH_OPEN, kernel)
    
    return mascara_final