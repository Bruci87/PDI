import cv2
def main():
    img = cv2.imread('image/folhaMangueira.jpg', cv2.IMREAD_COLOR)
    heigth, width, chanell = img.shape
    sizeImg = img.size
    type_date = img.dtype
    print(f"Dimensoes: {width}x{heigth} pixels")
    print(f"Canais: {chanell}")
    print(f"Tamannho total da matriz (elementos): {sizeImg}")
    print(f"Tipos de dados pizels: {type_date}")
    cv2.imshow("Algo: ", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()    