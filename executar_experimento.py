import os
import glob
import time
import json
import cv2
import numpy as np
from roi_count import RoiCounter
from pipelines import pipeline_1_hsv_clahe, pipeline_2_restauracao_colorimetria

def agrupar_arquivos(diretorio_dados):
    """Separa as imagens em seus 5 grupos experimentais correspondentes."""
    grupos = {
        "original": [],
        "gauss_baixo": [],
        "gauss_medio": [],
        "sp_baixo": [],
        "sp_medio": []
    }
    
    # Busca todas as imagens na pasta de experimentos
    imagens = glob.glob(os.path.join(diretorio_dados, "*.jpg")) + glob.glob(os.path.join(diretorio_dados, "*.png"))
    
    for img_path in imagens:
        nome_arquivo = os.path.basename(img_path)
        if "gauss_baixo" in nome_arquivo:
            grupos["gauss_baixo"].append(img_path)
        elif "gauss_medio" in nome_arquivo:
            grupos["gauss_medio"].append(img_path)
        elif "sp_baixo" in nome_arquivo:
            grupos["sp_baixo"].append(img_path)
        elif "sp_medio" in nome_arquivo:
            grupos["sp_medio"].append(img_path)
        else:
            # Arquivos originais (sem sufixo de ruído)
            grupos["original"].append(img_path)
            
    return grupos

def calcular_metricas(tp, fp, fn, gt_rois, pred_rois):
    """Calcula matematicamente Precision, Recall, F1 e MAE."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mae = abs(gt_rois - pred_rois)
    return precision, recall, f1, mae

def benchmark_tempo(pipeline_func, imagens_teste, n_repeticoes=3):
    """
    Mede o tempo total médio do pipeline (filtragem + segmentação + avaliação)
    em 5 imagens fixas, repetindo 3 vezes e excluindo a leitura de disco (I/O).
    """
    tempos_totais = []
    
    for img_path in imagens_teste:
        json_path = os.path.splitext(img_path)[0] + ".json"
        if not os.path.exists(json_path):
            continue
            
        # Carrega os dados na memória ANTES de iniciar a cronometragem (Exclui I/O)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        counter = RoiCounter(json_path)
        
        tempos_imagem = []
        for _ in range(n_repeticoes):
            t_inicio = time.perf_counter()
            
            # Executa o pipeline completo (Filtragem + Segmentação)
            img_binaria = pipeline_func(img_rgb)
            # Executa a Avaliação
            _ = counter.evaluate(img_binaria)
            
            t_fim = time.perf_counter()
            tempos_imagem.append(t_fim - t_inicio)
            
        # Armazena a média das 3 repetições para esta imagem
        tempos_totais.append(np.mean(tempos_imagem))
        
    # Retorna o tempo médio geral gasto para processar uma imagem neste grupo
    return np.mean(tempos_totais) if tempos_totais else 0.0

def rodar_experimento_grupo(lista_imagens, pipeline_func, nome_pipeline):
    """Executa o pipeline e avalia todas as imagens de um grupo específico."""
    resultados = []
    
    for img_path in lista_imagens:
        json_path = os.path.splitext(img_path)[0] + ".json"
        if not os.path.exists(json_path):
            continue
            
        # Carrega a imagem RGB
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Instancia o avaliador oficial do professor
        counter = RoiCounter(json_path)
        
        # 1. Processa a imagem usando o nosso pipeline
        img_binaria = pipeline_func(img_rgb)
        
        # 2. Avalia usando o script roi_count.py
        res = counter.evaluate(img_binaria)
        
        # Extrai os dados retornados pelo script do professor
        tp = res['TP']
        fp = res['FP']
        fn = res['FN']
        pred_rois = res['predicted_rois']
        gt_rois = res['ground_truth_rois']
        
        # 3. Calcula as métricas matemáticas exigidas
        prec, rec, f1, mae = calcular_metricas(tp, fp, fn, gt_rois, pred_rois)
        
        resultados.append([tp, fp, fn, prec, rec, f1, mae])
        
    return np.array(resultados)

def exibir_tabela_consolidada(nome_grupo, dados_p1, dados_p2, tempo_p1, tempo_p2):
    """Formata e imprime os resultados estatísticos conforme exigido no relatório."""
    print(f"\n================ Grupo: {nome_grupo.upper()} ================")
    print(f"{'Métrica':<12} | {'Pipeline 1 (Média ± DP)':<25} | {'Pipeline 2 (Média ± DP)':<25}")
    print("-" * 70)
    
    metricas_nomes = ["TP", "FP", "FN", "Precisão", "Recall", "F1-Score", "MAE"]
    
    for i, nome in enumerate(metricas_nomes):
        media_p1, dp_p1 = np.mean(dados_p1[:, i]), np.std(dados_p1[:, i])
        media_p2, dp_p2 = np.mean(dados_p2[:, i]), np.std(dados_p2[:, i])
        
        # Formata com mais casas decimais para as métricas float
        fmt = ".4f" if i >= 3 else ".2f"
        
        print(f"{nome:<12} | {media_p1:{fmt}} ± {dp_p1:{fmt}} | {media_p2:{fmt}} ± {dp_p2:{fmt}}")
        
    print("-" * 70)
    print(f"{'Tempo Médio':<12} | {tempo_p1 * 1000:.2f} ms / imagem            | {tempo_p2 * 1000:.2f} ms / imagem")
    print("=" * 70)

def main():
    diretorio_dados = "./imagens_experimentais"
    
    if not os.path.exists(diretorio_dados) or len(os.listdir(diretorio_dados)) == 0:
        print(f"❌ Erro: A pasta '{diretorio_dados}' está vazia ou não existe.")
        print("Execute primeiro o script 'gerar_testes.py' para criar as imagens de teste.")
        return
        
    print("📦 Separando imagens por grupos experimentais...")
    grupos = agrupar_arquivos(diretorio_dados)
    
    # Sorteia e fixa 5 imagens aleatórias do grupo original para o teste estrito de tempo
    # (Usando semente fixa para manter a consistência metodológica)
    rng = np.random.default_rng(42)
    total_originais = len(grupos["original"])
    
    if total_originais < 5:
        print("⚠️ Aviso: Você possui menos de 5 imagens originais. O benchmark usará todas as disponíveis.")
        indices_tempo = np.arange(total_originais)
    else:
        indices_tempo = rng.choice(total_originais, size=5, replace=False)
        
    for nome_grupo, lista_imagens in grupos.items():
        if len(lista_imagens) == 0:
            print(f"Aviso: O grupo {nome_grupo} não contém imagens. Pulando...")
            continue
            
        print(f"🧪 Processando o grupo [{nome_grupo}] com ambos os pipelines...")
        
        # Executa as avaliações de qualidade das imagens do grupo
        dados_p1 = rodar_experimento_grupo(lista_imagens, pipeline_1_hsv_clahe, "Pipeline 1")
        dados_p2 = rodar_experimento_grupo(lista_imagens, pipeline_2_restauracao_colorimetria, "Pipeline 2")
        
        # Mapeia quais imagens deste grupo específico correspondem às 5 imagens fixas de teste de tempo
        imagens_tempo_grupo = []
        for idx in indices_tempo:
            nome_base_original = os.path.splitext(os.path.basename(grupos["original"][idx]))[0]
            # Localiza o arquivo equivalente do grupo atual (ex: adicionando _gauss_baixo se for o caso)
            for img_path in lista_imagens:
                if nome_base_original in os.path.basename(img_path):
                    imagens_tempo_grupo.append(img_path)
                    break
                    
        # Roda o benchmark de tempo do grupo de forma limpa e isolada
        tempo_p1 = benchmark_tempo(pipeline_1_hsv_clahe, imagens_tempo_grupo)
        tempo_p2 = benchmark_tempo(pipeline_2_restauracao_colorimetria, imagens_tempo_grupo)
        
        # Imprime na tela a tabela consolidada com Média ± Desvio-Padrão
        exibir_tabela_consolidada(nome_grupo, dados_p1, dados_p2, tempo_p1, tempo_p2)

if __name__ == "__main__":
    main()