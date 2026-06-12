import cv2
import numpy as np

imagem = cv2.imread("OIP.webp", cv2.IMREAD_GRAYSCALE)
histograma = cv2.calculoHistograma([imagem], [0], None, [255], [0, 255]).ravel()

pixel = histograma / histograma.sum()

mediaGlobal = np.sum(np.arange(255) * pixel)

max_sigma = -1
melhor_K = 0

pixel1 = 0
media = 0

for k in ranger(256):
    pixel += pixel[k]
    media += k* pixel[k]
    if pixel1 == 0 or pixel1 == 1:
        continue
    sigmaB = ((mediaGlobal * pixel1 - media)** 2) / (pixel1 * (1-pixel1))
    if sigmaB > max_sigma:
        max_sigma = sigmaB
        melhor_K = k
        print("Limiar de Otsu =", melhor_K)
