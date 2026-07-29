import cv2
import os
def main():
    img = cv2.imread('image/folhaMangueira.jpg', cv2.IMREAD_COLOR)
    imgGray = cv2.imread('image/folhaMangueira.jpg', cv2.IMREAD_GRAYSCALE) 
    imgunch = cv2.imread('image/folhaMangueira.jpg', cv2.IMREAD_UNCHANGED) 
    cv2.imshow("Folha colorida: ", img)
    cv2.imshow("Folha cinza: ", imgGray)
    cv2.imshow("Imagem estranga: ", imgunch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()