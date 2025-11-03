import pickle
import os
import Funciones as F
import numpy as np
import cv2
from sklearn import preprocessing as pre

# ES: Definición de los nodos, Funciones y Terminales
# EN: Define Nodes, Functions and terminals 
class Node:
    pass

class FunctionNode(Node):
    def __init__(self, function, arity):
        self.function = function
        self.arity = arity
        self.children = []

    def evaluate(self):
        results = [child.evaluate() for child in self.children]
        return self.function(*results)

    def __str__(self):
        return f"F.{self.function.__name__}({', '.join(map(str, self.children))})"

class LeafNode(Node):
    def __init__(self, value):
        self.value = value

    def evaluate(self):
        return self.value

    def __str__(self):
        return str(self.value)

def print_tree_visual(tree, indent="", last=True):
    print(indent, end="")
    if last:
        print("└── ", end="")
        indent += "    "
    else:
        print("├── ", end="")
        indent += "│   "
    if isinstance(tree, LeafNode):
        print(f"({tree.value})")
    else:
        print(f"({tree.function.__name__})")
        for i, child in enumerate(tree.children):
            print_tree_visual(child, indent, i == len(tree.children) - 1)

# ES: Cargar la población
# EN: Load poblation
with open('Historial Resultados/20-Mar-2025/gp_individuals_generation_100.pkl', 'rb') as f:
    population = pickle.load(f)

# ES: Imprimir toda la población
# EN: Print all individuals
# for individual in population:
#     print(individual)

# ES: Cargar el individuo
# EN: Load individual
individual = population[0]

print("====================")
print("Individuo a evaluar:")
print(individual)

# ES: Imprimir el árbol visualmente
# EN: Print the tree visually
print("Árbol visual:")
print_tree_visual(individual)

tree_str = str(individual)
Number_Folder = 5

# ES: Obtiene el par de imagenes
# EN: Get image pairs
def get_image_pairs(directory):
    files = sorted([f for f in os.listdir(directory) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')])
    return [(files[i], files[i + 1]) for i in range(0, len(files), 2)]

Fitness = []
Precision = []
Recall = []
Overall_Accuracy = []
Harmonic_Mean = []
Weighted_Kappa = []

image_directory = f"/Users/andro/Documents/Repositorios/Mistletoe_Tree_GP/Imagenes_Entrenamiento/{Number_Folder}"
image_pairs = get_image_pairs(image_directory)

for img1, img2 in image_pairs:

    Img = F.Multiband_Convertation(os.path.join(image_directory, img1))
    Img_mask = F.Multiband_Convertation(os.path.join(image_directory, img2))

    # ES: Imagenes en diferentes bandas
    # EN: Images in different bands
    R = Img[0] # Red
    G = Img[1] # Green
    B = Img[2] # Blue
    I = np.stack(Img[:3], axis=-1)
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)  # Gray
    REG = Img[3] # Red edge spectral
    NIR = Img[4] # Near-infrared

    # ES: Normalizada cada banda entre 0 y 1
    # EN: Normlize each band between 0 and 1
    R = pre.MinMaxScaler().fit_transform(R)
    G = pre.MinMaxScaler().fit_transform(G)
    B = pre.MinMaxScaler().fit_transform(B)
    I = pre.MinMaxScaler().fit_transform(I)
    REG = pre.MinMaxScaler().fit_transform(REG)
    NIR = pre.MinMaxScaler().fit_transform(NIR)

    # ES: Ejecutar el tree_str como expresión de Python
    # EN: Execute the tree_str as a Python expression
    Res = eval(tree_str, {"F": F, "I": I, "R": R, "G": G, "B": B, "REG": REG, "NIR": NIR})
        
    Binary_Res = np.where(Res > 0, 1, 0)

    # ES: Calcula los componentes de la matriz de confusión
    # EN: Calculate confusion matrix components
    TP = np.sum(np.logical_and(Binary_Res == 1, Img_mask[0] == 1))
    TN = np.sum(np.logical_and(Binary_Res == 0, Img_mask[0] == 0))
    FP = np.sum(np.logical_and(Binary_Res == 1, Img_mask[0] == 0))
    FN = np.sum(np.logical_and(Binary_Res == 0, Img_mask[0] == 1))

    weight_matrix = np.array([[0, 2], [1, 0]])

    # Crear la matriz de confusión
    confusion_matrix = np.array([[TP, FP], [FN, TN]])

    # ===== Calcular el Coeficiente Kappa ponderado =====
    
    # Calcular la matriz de confusión observada
    O = confusion_matrix
    # Calcular la matriz de confusión esperada bajo independencia
    total = np.sum(O)
    E = np.outer(np.sum(O, axis=1), np.sum(O, axis=0)) / total
    # Calcular el Coeficiente Kappa ponderado
    weighted_kappa = 1 - np.sum(weight_matrix * O) / np.sum(weight_matrix * E)
    Weighted_Kappa.append(weighted_kappa)



    # ES: Pintar los True Positives (TP) en la imagen original RGB
    # EN: Paint True Positives (TP) on the original RGB image
    tp_overlay = np.zeros_like(Img[0], dtype=np.uint8)
    tp_overlay[np.logical_and(Binary_Res == 1, Img_mask[0] == 1)] = 255  # TP - White

    # Convert the original RGB image to uint8 for overlay
    original_rgb = np.stack((R, G, B), axis=-1) * 255
    original_rgb = original_rgb.astype(np.uint8)

    # Add the TP overlay to the original RGB image
    tp_colored_overlay = cv2.merge((tp_overlay, np.zeros_like(tp_overlay), np.zeros_like(tp_overlay)))
    result_image = cv2.addWeighted(original_rgb, 0.7, tp_colored_overlay, 0.3, 0)

    # Guardar la imagen con los TP pintados en formato JPG o PNG
    # EN: Save the image with TP painted in JPG or PNG format
    output_path = os.path.join(f"/Users/andro/Documents/Repositorios/Mistletoe_Tree_GP/Imagenes_Resultados/{Number_Folder}", f"TP_overlay_{os.path.splitext(img1)[0]}.png")
    cv2.imwrite(output_path, result_image)



    # ES: Calcular precisión, exhaustividad y F-medida
    # EN: Calculate precision, recall, and F-measure
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # ES: Calcular la precisión general
    # EN: Calculate overall accuracy
    overall_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # ES: Calcular la media armónica
    # EN: Calculate harmonic mean
    harmonic_mean = 3 / (1 / f_measure + 1 / precision + 1 / recall) if (f_measure > 0 and precision > 0 and recall > 0) else 0


    #print(f"Precision: {precision} Recall: {recall} F-measure: {f_measure}")

    Fitness.append(f_measure)
    Precision.append(precision)
    Recall.append(recall)
    Overall_Accuracy.append(overall_accuracy)
    Harmonic_Mean.append(harmonic_mean)

# ES: Promedio Fitness, Precisión, Recall, Overall Acuracy y Harmonic Mean
# EN: Average Fitness, Precision, Recall, Overall acuracy and Harmonic Mean
Avg_Fitness = np.mean(Fitness)
Avg_Precision = np.mean(Precision)
Avg_Recall = np.mean(Recall)
Avg_Overall_Accuracy = np.mean(Overall_Accuracy)
Avg_Harmonic_Mean = np.mean(Harmonic_Mean)
Avg_Weighted_Kappa = np.mean(Weighted_Kappa)

print(f"Fitness: {Avg_Fitness}")
print(f"Precision: {Avg_Precision}")
print(f"Recall: {Avg_Recall}")
print(f"Overall Accuracy: {Avg_Overall_Accuracy}")
print(f"Harmonic Mean: {Avg_Harmonic_Mean}")
print(f"Weighted Kappa: {Avg_Weighted_Kappa}")



#return Avg_Fitness, Avg_Precision, Avg_Recall, Avg_Overall_Accuracy, Avg_Harmonic_Mean, Avg_Weighted_Kappa

