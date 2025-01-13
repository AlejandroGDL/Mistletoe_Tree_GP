import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from math import *
from random import *


# ES: Función para realizar el filtro gaussiano en imagen
# EN: Function to apply Gaussian filter on image
def Filter_Gaussian(image):
    Img = np.nan_to_num(image, nan=0.0)
    return cv2.GaussianBlur(Img, (15, 15), 0)

# ES: Función para obtener la derivada en X y Y utilizando Sobel
# EN: Function to get the derivative in X and Y using Sobel
def DX(image):
    Img = np.nan_to_num(image, nan=0.0)
    sobelx = cv2.Sobel(Img, cv2.CV_64F, 1, 0, ksize=5)
    return sobelx

def DY(image):
    Img = np.nan_to_num(image, nan=0.0)
    sobely = cv2.Sobel(Img, cv2.CV_64F, 0, 1, ksize=5)
    return sobely

# ES: Función para normalizar imágenes multiespectrales de manera lineal
# EN: Function to linearly normalize multispectral images
def Normalize_Image(image):
    Img = np.nan_to_num(image, nan=0.0)
    norm_image = cv2.normalize(Img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image

# ES: Función para sumar dos imagenes
# EN: Funtion to sum two images
def Img_Sum(Image1, Image2):
    Validate_Size(Image1, Image2)

    #Validar valores NAN
    Img1 = np.nan_to_num(Image1, nan=0.0)
    Img2 = np.nan_to_num(Image2, nan=0.0)

    return cv2.add(Img1, Img2, dtype=cv2.CV_64F)

# ES: Función para restar dos imagenes
# EN: Funtion to subtract two images
def Img_Sub(Image1, Image2):
    Validate_Size(Image1, Image2)

    Img1 = np.nan_to_num(Image1, nan=0.0)
    Img2 = np.nan_to_num(Image2, nan=0.0)
    return cv2.subtract(Img1, Img2, dtype=cv2.CV_64F)

# ES: Función para multiplicar dos imagenes
# EN: Funtion to multiply two images
def Img_Multi(Image1, Image2):
    Validate_Size(Image1, Image2)

    Img1 = np.nan_to_num(Image1, nan=0.0)
    Img2 = np.nan_to_num(Image2, nan=0.0)
    return cv2.multiply(Img1, Img2, dtype=cv2.CV_64F)

# ES: Función para dividir dos imagenes
# EN: Funtion to divide two images
def Img_Div(Image1, Image2):
    Validate_Size(Image1, Image2)
    Img1 = np.nan_to_num(Image1, nan=0.0)
    Img2 = np.nan_to_num(Image2, nan=0.0)

    # Remplaza los ceros en la Img para evitar dividir 
    # Replace zeros in Img2 to avoid division by zero
    Img2[Img2 == 0] = Img1[Img2 == 0]
    
    
    return cv2.divide(Img1, Img2, dtype=cv2.CV_64F)

# ES: Función para observar las bandas de la imagen
# EN: Funtion to see all image bands
def Multiband_Convertation(Image):
    # Abrir la imagen
    # Open Image
    with rasterio.open(Image) as src:
        numero_bandas = src.count
        #print(numero_bandas)
        
        # Leer todas las bandas
        # Read all bands
        bandas = [src.read(i) for i in range(1, numero_bandas + 1)]

    # Validar y sustituir valores no numéricos por 0
    # Validate and sustitute not a number by 0
    bandas = [np.nan_to_num(banda, nan=0.0) for banda in bandas]
    
    return bandas

# ES: Función para validar que dos imágenes sean del mismo tamaño
# EN: Function to validate that two images are the same size
def Validate_Size(Img1, Img2):
    if Img1.shape != Img2.shape:
        raise ValueError("The images do not have the same size.")



def ADD(ValuesList):
    sumtotal = 0
    if (ValuesList[0] is None) or (ValuesList[1] is None):
        return 0
    for val in ValuesList:
        sumtotal = sumtotal + val
    return sumtotal

def SUB(ValuesList):
    if (ValuesList[0] is None) or (ValuesList[1] is None):
        return 0
    return ValuesList[0] - ValuesList[1]

def MUL(ValuesList):
    if (ValuesList[0] is None) or (ValuesList[1] is None):
        return 0
    return ValuesList[0] * ValuesList[1]

# Protected division
def DIV(ValuesList):
    # Leon Dozal - CentroGeo - 15/10/2019
    if (ValuesList[0] is None) or (ValuesList[1] is None):
        return np.zeros(ValuesList[0].shape)
    if (type(ValuesList[0]) == 'int') or (type(ValuesList[1]) == 'int'):
        return np.zeros(ValuesList[0].shape)

    resp = np.divide(ValuesList[0], ValuesList[1])
    resp[resp == inf] = 1.0  # Replace infinite values with ones
    resp[resp == -inf] = 1.0  # Replace negative infinite values with ones
    resp[resp == nan] = 0.0  # Replace NaNs with zeros
    return resp

# Spatial Saliency Functions

# Leon Dozal - CentroGeo - 27/10/2019
# Derivative x of image
def Dx(image):
    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
    sobelx64f = cv2.Sobel(image[0], cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    return np.uint8(abs_sobel64f)

    #kernel_identity = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    #output = cv2.filter2D(image[0], -1, kernel_identity)
    #return output
    #cv2.imshow('Identity filter', output)

# Leon Dozal - CentroGeo - 28/10/2019
# Derivative y of image
def Dy(image):
    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
    sobely64f = cv2.Sobel(image[0], cv2.CV_64F, 0, 1, ksize=5)
    abs_sobel64f = np.absolute(sobely64f)
    return np.uint8(abs_sobel64f)

    #kernel_identity = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    #output = cv2.filter2D(image[0], -1, kernel_identity)
    #return output
    #cv2.imshow('Identity filter', output)

def Dxy(image):
    return Dx(Dy(image[0]))

def Gauss_1(image):
    return cv2.GaussianBlur(image[0], (0, 0), 1)

def Gauss_2(image):
    return cv2.GaussianBlur(image[0], (0, 0), 2)

def ABS(image):
    return np.absolute(image[0])

def HALF(image):
    return 0.5 * np.array(image[0])

def SQRT(image):
    resp = np.sqrt(np.absolute(image[0]))
    resp[resp == inf] = 1.0  # Replace infinite values with ones
    resp[resp == -inf] = 1.0  # Replace negative infinite values with ones
    resp[resp == nan] = 0.0  # Replace NaNs with zeros
    return np.array(resp, np.uint8)

def SQR(image):
    return np.square(image[0])

def GABOR0(image):
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    return cv2.filter2D(image[0], cv2.CV_8UC3, g_kernel)

def GABOR45(image):
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    return cv2.filter2D(image[0], cv2.CV_8UC3, g_kernel)

def GABOR90(image):
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 2, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    return cv2.filter2D(image[0], cv2.CV_8UC3, g_kernel)

def GABOR135(image):
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, (np.pi*3) / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    return cv2.filter2D(image[0], cv2.CV_8UC3, g_kernel)

# Leon Dozal - CentroGeo - 16/11/2019
# Protected natural logarithm
def LOG(image):
    if (image[0] is None) or (isinstance(image[0], int)):
        return 0
    return np.log(np.absolute(image[0]), out=np.zeros_like(np.absolute(image[0])), where=(np.absolute(image[0]) != 0))

def EXP(image):
    if image[0] is None:
        return 0
    img = np.exp(image[0])
    return img

def RANDINT(ValuesList):  # return a random integer between ranges a,b
    if ValuesList[1] < ValuesList[0]:
        return randint(ValuesList[1], ValuesList[0])
    return randint(ValuesList[0], ValuesList[1])

def RANDOM(ValuesList):
    return random()


def main():
    Imagen1_path = '/Users/andro/Documents/Repositorios/CentroGeo-Practicas/Fun/DJI_1350.JPG'
    Imagen2_path = '/Users/andro/Documents/Repositorios/CentroGeo-Practicas/Fun/DJI_1360.JPG'
    #Validate_Size(Imagen1_path,Imagen2_path)

    # Imagen1 = Multiband_Convertation(Imagen1_path)
    # Imagen2 = Multiband_Convertation(Imagen2_path)
    # print(Imagen1[0])


    # PRUEBAS
    Imagen1 = [np.array([[np.nan, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), np.array([[5, 6], [7, 8]], dtype=np.float32), np.array([[9, 10], [11, 12]], dtype=np.float32)]
    Imagen2 = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]], dtype=np.float32), np.array([[5, 6], [7, 8]], dtype=np.float32), np.array([[9, 10], [11, 12]], dtype=np.float32)]
    print(f'Arreglo Original: \n {Imagen1[0]}')
    


    # Filtro Gaussiano
    Filtro = Filter_Gaussian(Imagen1[0],3)
    cv2.imshow('Filtro Gaussiano', Filtro)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Derivadas
    Dx = DX(Imagen1[0])
    print(f'Dervidad X: \n {Dx}')
    cv2.imshow('Derivada X', Dx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Dy = DY(Imagen1[0])
    print(f'Dervidad Y: \n {Dy}')
    cv2.imshow('Derivada Y', Dy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Normalización
    Img_Norm = Normalize_Image(Imagen1[0])
    print(f'Normalización: \n {Img_Norm}')
    cv2.imshow('Normalización', Img_Norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Operaciones
    Suma = Img_Sum(Imagen1[0], Imagen2[0])
    print(f'Suma: \n {Suma}')
    cv2.imshow('Suma', Suma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Resta = Img_Sub(Imagen1[0], Imagen2[0])
    print(f'Resta: \n {Resta}')
    cv2.imshow('Resta', Resta)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Multiplicación = Img_Multi(Imagen1[0], Imagen2[0])
    print(f'Multiplicaión: \n {Multiplicación}')
    cv2.imshow('Multiplicación', Multiplicación)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Divición = Img_Div(Imagen1[0], Imagen2[0])
    print(f'Divición: \n {Divición}')
    cv2.imshow('Divición', Divición)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    
 

if __name__ == "__main__":
    main()

