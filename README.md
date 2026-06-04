[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/8INDgbLT)
# [TADS]+[PDI] Projeto Filtragem Espacial + Segmentação Microbiológica

Projeto da disciplina de Processamento Digital de Imagens do curso de Análise e Desenvolvimento de Sistemas da UFRN aplicando transformações de intensidade, processamento de histogramas, filtragem espacial, restauração simples e segmentação para quantificação automática de colônias microbianas em placas de Petri no contexto da análise microbiológica de leite.

## Descrição

Deve-se desenvolver e analisar pipelines de pré-processamento de imagens visando melhorar a segmentação e a contagem automática de colônias microbianas. Como objetivos específicos, deve-se:

-   Carregar o conjunto de imagens indicado na seção Dados;
-   Implementar rotinas de degradação artificial para simulação de condições adversas de captura;
-   Implementar técnicas de processamento de histogramas;
-   Implementar filtros espaciais de suavização;
-   Implementar filtros espaciais de aguçamento;
-   Implementar técnicas simples de restauração de imagens;
-   Implementar um pipeline de segmentação e contagem automática de colônias;
-   Determinar quais pipelines de pré-processamento produzem melhor desempenho de segmentação;
-   Determinar quais pipelines produzem melhor contagem microbiológica;
-   Analisar experimentalmente o comportamento médio dos pipelines sob diferentes tipos de degradação.

Espera-se que, ao final do trabalho, os alunos sejam capazes de modelar pipelines de processamento digital de imagens e analisar empiricamente o impacto das técnicas de filtragem espacial na qualidade da segmentação e da quantificação final.

## Dados

Deve-se utilizar todas as imagens presentes no diretório imagens.

Essas imagens são um subconjunto dos dados da base AGAR.

AGAR Dataset --- https://agar.neurosys.com/


### Arquivos de anotação (ground truth)

Cada imagem do conjunto AGAR Dataset possui um arquivo de anotação associado no formato JSON, contendo as informações de referência (ground truth) da imagem.

Exemplo de informações disponíveis:

- `colonies_number`: quantidade de regiões de colônias anotadas presentes na imagem;
- `labels`: lista contendo as anotações individuais de cada colônia, incluindo:
  - identificador (`id`);
  - posição aproximada (`x`, `y`);
  - dimensão aproximada (`width`, `height`);
- `background`: informação descritiva do fundo da imagem;
- `class`: espécie microbiológica anotada (não utilizada neste projeto).

Para este trabalho, devem ser considerados apenas:

- quantidade de regiões anotadas no ground truth (`colonies_number`);
- posição aproximada das colônias (`labels`).

A informação de espécie (`class`) deve ser desconsiderada.

Ao gerar versões degradadas artificialmente de uma imagem, seu arquivo JSON associado deve ser replicado integralmente, preservando as mesmas informações de referência, uma vez que as degradações aplicadas neste projeto não alteram a quantidade, posição ou dimensão das colônias, apenas a qualidade visual da imagem.

Exemplo:

```text
000123.jpg
000123.json
```

Após aplicar ruído gaussiano médio:

```text
000123_gauss_medio.jpg
000123_gauss_medio.json
```

Todos devem herdar o mesmo ground truth da imagem original correspondente.

Essa replicação é necessária para:

* manter a rastreabilidade entre imagem e anotação;
* automatizar a leitura do ground truth durante os experimentos;
* permitir comparação direta entre imagens originais e degradadas;
* calcular automaticamente os erros de contagem para cada pipeline avaliado.



## Degradações artificiais

Para cada imagem selecionada do AGAR Dataset, deverão ser geradas versões degradadas artificialmente conforme a tabela abaixo.

As degradações devem ser aplicadas individualmente, isto é, uma degradação por vez sobre a imagem original (sem composição entre degradações nesta etapa).

As imagens do conjunto devem ser consideradas como imagens coloridas RGB de 8 bits, com cada canal no intervalo:

```math
[0,255]
```

Toda degradação deve ser aplicada aos três canais de cor, preservando a estrutura RGB da imagem.

| Tipo                        | Parâmetro              | Baixo | Médio |
| --------------------------- | ---------------------- | ----: | ----: |
| Ruído Gaussiano aditivo | desvio padrão ($\sigma$) |     5 |    15 |
| Sal e Pimenta           | densidade (p)          |  0,01 |  0,05 |

onde:

* ruídos aditivos devem ser aplicados pixel a pixel;
* valores fora do intervalo devem ser saturados para 0 ou 255;
* densidade (p) corresponde à fração de pixels contaminados;
* cada degradação deve ser gerada de forma aleatória, registrando a semente utilizada para garantir reprodutibilidade do experimento.

### Formulação das degradações

Considere:

```math
I(x,y)=[R(x,y),G(x,y),B(x,y)]
```

como o vetor RGB do pixel.


---

#### 1) Ruído Gaussiano aditivo

Aplicar independentemente em cada canal:

```math
I'(x,y)=I(x,y)+[n_R,n_G,n_B], \quad n_c \sim \mathcal{N}(0,\sigma^2)
```

com:

* baixo: ($\sigma=5$)
* médio: ($\sigma=15$)

---

#### 2) Ruído Sal e Pimenta

Selecionar aleatoriamente uma fração (p) dos pixels:

* metade recebe:

```math
[255,255,255]
```

* metade recebe:

```math
[0,0,0]
```

Densidade:

* baixo: 1%
* médio: 5%


---


## Função de Quantificação


Assuma que a etapa de segmentação já foi realizada pelos alunos, gerando uma imagem binária em que:

- background = preto = 0
- colônias segmentadas = branco = 255
- existe um arquivo JSON associado contendo:
  - `colonies_number`
  - `labels` com:
    - `x`
    - `y`
    - `width`

Neste projeto:

- cada anotação do JSON define uma ROI quadrada;
- o lado do quadrado será dado por `width`;
- a ROI terá canto superior esquerdo em `(x,y)`;
- para cada ROI, considera-se acerto se existir ao menos 1 pixel branco (255) da segmentação dentro dessa ROI.

Utilize o código fornecido no arquivo roi_count.py que, dada a imagem segmentada e o json de ground truth, retorna:
- TP: quantidade de componentes conectados segmentados que intersectam ao menos uma ROI real;
- FP: quantidade de componentes conectados segmentados que não intersectam nenhuma ROI real;
- FN: quantidade de regiões reais que não foram segmentadas;
- predicted_rois: quantidade de regiões preditas;
- ground_truth_rois: quantidade de regiões anotadas no ground truth;
- visualization: imagem com a visualização dos resultados.


Exemplo de uso:

```python
import cv2
from roi_count import RoiCounter

counter = RoiCounter("imagem.json")
img = cv2.imread("segmentada.png", cv2.IMREAD_GRAYSCALE)

result = counter.evaluate(img)

```


## Metodologia

Sugere-se os seguintes passos:

1. Desenvolva um módulo para carregamento das imagens e anotações
   - carregar as imagens do subconjunto AGAR Dataset;
   - carregar o arquivo `.json` associado a cada imagem;
2. Desenvolva um módulo de segmentação considerando somente as imagens sem ruído
   - aplicar técnicas de segmentação baseadas em:
     - modelos de cor;
     - limiarização simples;
     - limiarização multicanal;
     - segmentação por colorimetria;
   - produzir uma imagem binária:
     - fundo = 0;
     - colônia = 255.
   - A saída do pipeline deve obrigatoriamente ser uma imagem binária monocanal uint8 contendo apenas valores 0 e 255.
   - O desenvolvimento deve ser feito em Python, utilizando NumPy e OpenCV.
3. Desenvolva um módulo de avaliação quantitativa
   - utilizar o script roi_count.py;
   - gerar TP, FP, FN;
   - produzir visualização da avaliação;
   - consolidar métricas.
4. Defina 2 pipelines distintos
   - Cada pipeline deve combinar técnicas de pré-processamento + segmentação.
   - Exemplos diversos:
     - gaussiano => segmentação
     - bilateral => segmentação
     - gaussiano => Laplaciano => segmentação  
     - mediana => unsharp => segmentação  
    - No conjunto dos dois pipelines, devem aparecer técnicas de pelo menos duas das seguintes categorias:
     - Filtros de restauração
     - Processamento de Histogramas
     - Detecção de bordas
5. Desenvolva um módulo para geração de degradações artificiais
   - implementar as degradações especificadas na seção anterior;
   - aplicar individualmente cada degradação em dois níveis (baixo e médio) sobre o mesmo conjunto de imagens;
   - utilizar sementes controladas para permitir reprodutibilidade;
   - registrar tipo de degradação, intensidade e semente.
6. Aplique todos os pipelines ao conjunto experimental
   * O conjunto experimental será composto por 5 grupos: 1 grupo sem degradação + 4 grupos degradados.
   * executar todos os pipelines sobre todas as imagens sem degradação;
   * executar todos os pipelines sobre todas as imagens para cada grupo degradado;
   * consolidar os resultados em tabelas (média e desvio-padrão por grupo: sem degradação e para cada degradação);
    * TP
    * FP
    * FN
    * precisão
    * recall
    * F1-score
    * MAE: erro absoluto entre a quantidade de regiões anotadas no ground truth (ground_truth_rois) e quantidade estimada pelo pipeline (predicted_rois).
    * tempo de processamento.
7. Análise e discussão dos resultados
   * qual pipeline obteve menor erro de contagem;
   * das regiões que o algoritmo marcou como colônia, quantas realmente pertencem a colônias;
   * das colônias que realmente existem, quantas o algoritmo conseguiu encontrar;
   * qual impacto de cada degradação;
   * custo computacional versus desempenho;
   * pipeline recomendado para quantificação microbiológica em leite.


Para o item 4, deve-se considerar as técnicas de pré-processamento abaixo, implementando somente as técnicas necessárias para o pipeline escolhido, usando a implementação do Opencv quando possível:
   - Transformações de Intensidade
   - Filtros de suavização
     - média;
     - gaussiano;
     - mediana;
     - bilateral.
   - Filtros de aguçamento
     - Laplaciano;
     - unsharp masking;
   - Detecção de bordas
     - Sobel;
     - Canny.
   - Processamento de Histogramas
     - Equalização
     - Especificação
     - CLAHE
     - Processamento Local de Histograma usando Estatísticas da Imagem
   - Filtros de Restauração
     - média geométrica;
     - média harmônica;
     - média contra-harmônica;
     - mínimo e máximo;
     - Filtro Adaptativo de Redução de Ruído Local
     - Filtro Adaptativo de Mediana

### Métricas

---

* Precisão (Precision)  
Mede, entre as regiões detectadas, quantas realmente correspondem a colônias:

```math
Precision=\frac{TP}{TP+FP}
```

---

* Recall (Sensibilidade)  
Mede, entre as colônias reais, quantas foram encontradas:

```math
Recall=\frac{TP}{TP+FN}
```

---

* F1-score  
Média harmônica entre precisão e recall:

```math
F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall}
```

---

* MAE (Mean Absolute Error)  
Erro absoluto entre a quantidade real e a quantidade estimada:

```math
MAE=\left|ground\_truth\_rois-predicted\_rois\right|
```

onde:

```math
ground\_truth\_rois = \text{quantidade de regiões de colônias anotadas no ground truth}
```

```math
predicted\_rois = \text{quantidade de componentes conectados segmentados}
```

---

* Tempo de processamento  
Média de 3 repetições do tempo total necessário para executar o pipeline completo (filtragem + segmentação + avaliação) em 5 imagens selecionadas aleatoriamente uma única vez de cada grupo e mantidas fixas durante todo o experimentos, sem incluir o tempo de carregamento das imagens do disco.

---

## Entrega

A entrega será realizada via github classroom.

Devem ser entregues:

1.  Implementação dos módulos solicitados;
2.  Relatório com análise comparativa e discussão dos resultados;
    - Descrição dos pipelines, incluindo imagens exemplificando cada etapa;
    - Tabelas ilustrando os resultados;
    - Gráficos de boxplot comparando as métricas de Precision, Recall, F1, MAE e tempo entre os grupos (sem degradação e para cada grupo degradado).
    - Conclusão técnica sobre o pipeline mais indicado.


## Avaliação

A avaliação considerará os itens solicitados na seção de entrega, no entanto, a atribuição da nota é condicionada à comprovação da autoria por meio de perguntas durante a apresentação, em que poderão ser solicitadas explicações, justificativas e modificações no código apresentado.
