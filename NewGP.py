import random
import Funciones as F
import os

import pickle
import csv
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from sklearn import preprocessing as pre


# ES: Definición de los nodos
# EN: Node define 
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

# ES: Generación de un árbol aleatorio
# EN: Generate a random tree
def generate_random_tree(max_depth):
    if max_depth == 0:
        return LeafNode(random.choice(Term_set))
    else:
        function, arity = random.choice(Func_set)
        node = FunctionNode(function, arity)
        for _ in range(arity):
            node.children.append(generate_random_tree(max_depth - 1))
        return node

# ES: Obtener la profundidad del árbol
# EN: Get the depth of the tree
def get_tree_depth(node):
    if isinstance(node, LeafNode):
        return 1
    elif isinstance(node, FunctionNode):
        return 1 + max(get_tree_depth(child) for child in node.children)
    else:
        raise TypeError("Unknown node type")

# ES: Imprimir arbol (Texto)
# EN: Print tree (String)
def print_individual(tree):
    print(tree)

# ES: Imprimir arbol (Visual)
# EN: Print tree (Visual)
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

# ES: Función para evaluar un individuo
# EN: Function to evaluate an individual
def evaluate_individual(tree):
    tree_str = str(tree)
    print("====================")
    print("Individuo a evaluar:")
    print(tree_str)

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

    # ES: Carpeta imagenes entrenamiento
    # EN: Folder Image Training
    for Number_Folder in range(1, 5):

        image_directory = f"/Users/andro/Documents/Repositorios/Mistletoe_Tree_GP/ImagenesEntrenamiento/{Number_Folder}"
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
                
            # ===== PRUEBAS =====
            # ES: Nota, cuando se utilizan las pruebas eliminar el [0] de las líneas 275 - 278 y 289 - 292
            # EN: Note, when using tests remove the [0] of the lines 275 - 278 and 289 - 292
            # Res = np.array([[255, 0], [0, 255]])
            # Img_mask = np.array([[1, 0], [0, 1]])

            # ES: Mediante la mascara se consigue la matriz correcta
            # EN: Using the mask to get the correct matrix
            # Res = cv2.normalize(Res, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
            # _, thresh1 = cv2.threshold(Res, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            Binary_Res = np.where(Res > 0, 1, 0)
            #_, Binary_Res = cv2.threshold(BinaryMatrix, 0, 255, 0)
            
            # ES: Multiplica cada valor de la mascara por 255
            # EN: Each value has been multiply by 255
            #Img_mask = [value * 255 for value in Img_mask]

            # ES: Calcula los componentes de la matriz de confusión
            # EN: Calculate confusion matrix components
            TP = np.sum(np.logical_and(Binary_Res == 1, Img_mask[0] == 1))
            TN = np.sum(np.logical_and(Binary_Res == 0, Img_mask[0] == 0))
            FP = np.sum(np.logical_and(Binary_Res == 1, Img_mask[0] == 0))
            FN = np.sum(np.logical_and(Binary_Res == 0, Img_mask[0] == 1))

            weight_matrix = np.array([[0, 2], [1, 0]])

            # Crear la matriz de confusión
            confusion_matrix = np.array([[TP, FP], [FN, TN]])

            # Calcular el Coeficiente Kappa ponderado
            weighted_kappa = calculate_weighted_kappa(confusion_matrix, weight_matrix)
            Weighted_Kappa.append(weighted_kappa)

            # print(f"True Positives: {TP}")
            # print(f"True Negatives: {TN}")
            # print(f"False Positives: {FP}")
            # print(f"False Negatives: {FN}")
            # print(f"Sumatoria: {np.sum([TP, TN, FP, FN])}"  )

            # ES: Crea una imagen para visualizar la matriz de confución
            # EN: Create an image to visualize the confusion matrix
            # confusion_image = np.zeros_like(Binary_Res, dtype=np.uint8)
            # confusion_image[np.logical_and(Binary_Res == 1, Img_mask[0] == 1)] = 255  # TP - White
            # confusion_image[np.logical_and(Binary_Res == 0, Img_mask[0] == 0)] = 128  # TN - Gray
            # confusion_image[np.logical_and(Binary_Res == 1, Img_mask[0] == 0)] = 64  # FP - Light Gray
            # confusion_image[np.logical_and(Binary_Res == 0, Img_mask[0] == 1)] = 192  # FN - Dark Gray

            # ES: Combina las imagenes y las muestra
            # EN: Combine the images and show it
            # combined_image = np.hstack((Res, confusion_image))
            # cv2.imshow('Resultado y Matriz de Confusión', combined_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

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
    print(f"Weighted Kappa: {Avg_Weighted_Kappa}")

    return Avg_Fitness, Avg_Precision, Avg_Recall, Avg_Overall_Accuracy, Avg_Harmonic_Mean, Avg_Weighted_Kappa

# ES: Selección de padres (Ruleta)
# EN: Parent selection (Roulette)
def select_parent(population, fitness_values):
    total_fitness = sum(fitness_values)
    # ES: Si todos tienen fitness 0, selección aleatoria
    # EN: If all individual has fitness 0, random selection
    if total_fitness == 0:
        return random.choice(population)
    
    probabilities = [f / total_fitness for f in fitness_values]
    selected = random.choices(population, weights=probabilities, k=1)
    return selected[0]

# ES: Seleccion de par de individuos
# EN: Selection of pair of individuals
def select_distinct_parents(population, fitness_values):
    parent1 = select_parent(population, fitness_values)
    parent2 = select_parent(population, fitness_values)
    
    while parent1 == parent2:
        parent2 = select_parent(population, fitness_values)
    
    return parent1, parent2

# ES: Obtener todos los nodos y contarlos
# EN: Get all nodes and count
def get_all_nodes(node, nodes=[]):
    nodes.append(node)
    if isinstance(node, FunctionNode):
        for child in node.children:
            get_all_nodes(child, nodes)
    return nodes

# ES: Cruce
# EN: CrossOver
def crossover(parent1, parent2):
    # Clonar los padres para no modificar los originales
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)
    
    # Obtener todos los nodos de cada padre
    nodes1 = get_all_nodes(offspring1, [])
    nodes2 = get_all_nodes(offspring2, [])
    
    # Seleccionar puntos de cruce aleatorios
    crossover_point1 = random.choice(nodes1)
    crossover_point2 = random.choice(nodes2)
    
    # Verificar que el cruce sea válido
    if isinstance(crossover_point1, LeafNode) or isinstance(crossover_point2, LeafNode):
        return offspring1, offspring2  # Si se seleccionó una hoja, no se hace crossover
    
    # Intercambiar los subárboles
    crossover_point1.function, crossover_point2.function = crossover_point2.function, crossover_point1.function
    crossover_point1.children, crossover_point2.children = crossover_point2.children, crossover_point1.children
    
    return offspring1, offspring2

# ES: Mutación
# EN: Mutation
def mutate(tree, mutation_rate):
    # Función auxiliar para recorrer y modificar el árbol
    def mutate_node(node):
        if random.random() < mutation_rate:
            if isinstance(node, FunctionNode):
                # Opción 1: Cambiar la función (50% de probabilidad)
                if random.random() < 0.5:
                    new_func, new_arity = random.choice(Func_set)
                    node.function = new_func
                    # Ajustar hijos según la nueva aridad
                    if new_arity > len(node.children):
                        for _ in range(new_arity - len(node.children)):
                            node.children.append(generate_random_tree(max_depth=1))
                    else:
                        node.children = node.children[:new_arity]
                # Opción 2: Mutar un hijo (50% de probabilidad)
                else:
                    if node.children:
                        child_to_mutate = random.choice(node.children)
                        mutate(child_to_mutate, mutation_rate)  # <--- ¡Corrección aquí!
            elif isinstance(node, LeafNode):
                node.value = random.choice(Term_set)

    # Clonar el árbol para no modificar el original
    mutated_tree = copy.deepcopy(tree)
    # Recorrer el árbol clonado
    def traverse(node):
        mutate_node(node)
        if isinstance(node, FunctionNode):
            for child in node.children:
                traverse(child)
    traverse(mutated_tree)
    return mutated_tree

# ES: Calcular Kappa
# EN: Calculate Kappa
def calculate_weighted_kappa(confusion_matrix, weight_matrix):
    # Calcular la matriz de confusión observada
    O = confusion_matrix
    
    # Calcular la matriz de confusión esperada bajo independencia
    total = np.sum(O)
    E = np.outer(np.sum(O, axis=1), np.sum(O, axis=0)) / total
    
    # Calcular el Coeficiente Kappa ponderado
    kappa_w = 1 - np.sum(weight_matrix * O) / np.sum(weight_matrix * E)
    
    return kappa_w

# ES: Ciclo principal
# EN: Main Loop
def genetic_algorithm(population_size, generations, max_depth=5, crossover_rate=0.9, mutation_rate=0.1):
    # ES: Validar que las probabilidades sumen 1.0
    # EN: Validate that the probabilities sum to 1.0
    if not np.isclose(crossover_rate + mutation_rate, 1.0):
        raise ValueError("La suma de crossover_rate y mutation_rate debe ser 1.0")

    # ES: Variables para el mejor individuo global
    # EN: Variables for the global best individual

    global_max_fitness = -1
    global_best_individual = None

    Generation_Fitness = []
    fitness_history = []
    Max_Fitness_History = []

    # ES: Definir el tamaño de la élite
    # EN: Define elite size
    Elitism_Size = 1

    # ES: Crear la carpeta 'resultados'
    # EN: Create 'resultados' folder
    if not os.path.exists("resultados"):
        os.makedirs("resultados")

    # ES: Crear población
    # EN: Create population
    population = [generate_random_tree(max_depth) for _ in range(1 + population_size)]

    for generation in range( 1 + generations):
        print(f"\Generación {generation}:")

        # ES: Variables inciales de la generación
        # EN: Initial variables of the generation
        MaxFitness = 0
        best_individual_number = 0
        best_individual = 0

        # ES: Crea un archivo CSV de cada generación y guarda cada individuo
        # EN: Create a CSV archive of each generation and save each individual
        with open(f'resultados/gp_results_generation_{generation}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Individual', 'Fitness', 'Precision', 'Recall', 'Overall_Accuracy', 'Harmonic_Mean', 'Weighted_Kappa', 'Tree'])

            Avg_Fitness_Population = []

            # ES: Evaluación
            # EN: Evaluation
            for ind in population:
                Individual_Fitness, Individual_Precision, Individual_Recall, Individual_Overall_Accuracy, Individual_Harmonic_Mean, Individual_Weighted_Kappa = evaluate_individual(ind)

                Generation_Fitness.append(Individual_Weighted_Kappa)
                Avg_Fitness_Population.append(Individual_Weighted_Kappa)

                writer.writerow([population.index(ind), Individual_Fitness, Individual_Precision, Individual_Recall, Individual_Overall_Accuracy, Individual_Harmonic_Mean, Individual_Weighted_Kappa, ind])

                # ES: Encontrar el individuo con mayor fitness
                # EN: Find the individual with the highest fitness
                if Individual_Weighted_Kappa > MaxFitness:
                    #MaxFitness = Individual_Fitness
                    MaxFitness = Individual_Weighted_Kappa
                    best_individual_number = population.index(ind)
                    best_individual = ind
            
            # ES: Actualizar el mejor individuo global
            # EN: Update the global best individual
            if MaxFitness > global_max_fitness:
                global_max_fitness = MaxFitness
                global_best_individual = best_individual

            print(f"Mejor Individuo de la Generación {generation}:")
            print_individual(best_individual)
            print(f"Fitness: {MaxFitness}")

            # ES: Guardar el fitness máximo de esta generación
            # EN: Save the max fitness of this generation
            Max_Fitness_History.append(global_max_fitness)

            # ES: Elitismo
            # EN: Elitism
            elites = sorted(zip(population, Avg_Fitness_Population), key=lambda x: x[1], reverse=True)[:Elitism_Size]
            elites = [individual for individual, fitness in elites]

            # ES: Crear la nueva población con los élites
            # EN: Create the new population with the elites
            new_population = elites.copy()

            # ES: Generar una nueva población
            # EN: Generate a new population
            while len(new_population) < population_size:
                if random.random() < crossover_rate:
                    parent1, parent2 = select_distinct_parents(population, Avg_Fitness_Population)
                    child1, child2 = crossover(parent1, parent2)

                    print("============ CrossOver: ============")
                    print("Padre 1:")
                    print_tree_visual(parent1)
                    print("Padre 2:")
                    print_tree_visual(parent2)

                    print("Hijo 1 cruzado:")
                    print_tree_visual(child1)

                    print("Hijo 2 cruzado:")
                    print_tree_visual(child2)

                    if get_tree_depth(child1) <= max_depth:
                        new_population.append(child1)
                    if get_tree_depth(child2) <= max_depth:
                        new_population.append(child2)
                else:
                    parent = select_parent(population, Avg_Fitness_Population)
                    child = mutate(parent, mutation_rate=1)

                    print("============ Mutate: ============")
                    print("Individuo a mutar:")
                    print_tree_visual(parent)
                    print("Individuo mutado:")
                    print_tree_visual(child)

                    if get_tree_depth(child) <= max_depth:
                        new_population.append(child)
            population = new_population

        # ES: Promedio de fitness por generación
        # EN: Average fitness per generation
        Avg_Generation_Fitness = sum(Generation_Fitness) / len(Generation_Fitness)

        # ES: Guardar el promedio y maximo fitness 
        # EN: Save average and max fitness
        with open(f'resultados/gp_results_generation_{generation}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Average', Avg_Generation_Fitness, '','','','',''])
            writer.writerow(['Max', MaxFitness,'',best_individual_number,'','',''])

        # ES: Guardar los individuos de la generación en un archivo .pkl
        # EN: Save the individuals of the generation in a .pkl file
        with open(f'individuos/gp_individuals_generation_{generation}.pkl', 'wb') as f:
            pickle.dump(population, f)

        fitness_history.append(Avg_Generation_Fitness)

    # ES: Crear folder de resultados
    # EN: Save results folder
    results_folder = f'resultados'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # ES: Graficar el fitness promedio
    # EN: Plotting average fitness
    plt.plot(fitness_history, label='Average Fitness')
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Average Fitness por Generación')
    plt.legend()
    plt.grid(True)

    # ES: Guardar el gráfico de fitness promedio
    # EN: Save average fitness plot
    plt.savefig(f'resultados/average_fitness_plot.png')
    #plt.show()
    plt.clf()

    # ES: Graficar el fitness máximo
    # EN: Plotting max fitness
    plt.plot(Max_Fitness_History, label='Max Fitness', linestyle='--', color='red')
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Max Fitness por Generación')
    plt.legend()
    plt.grid(True)

    # ES: Guardar el gráfico de fitness máximo
    # EN: Save max fitness plot
    plt.savefig(f'resultados/max_fitness_plot.png')
    #plt.show()
    plt.clf()

    # ES: Retornar el mejor individuo global
    # EN: Return the global best individual
    print("\nMejor Individuo Global:")
    print_individual(global_best_individual)
    print(f"Fitness: {global_max_fitness}")
    
    return global_best_individual

# ES: Funciones y sus terminales
# EN: Functions and terninals
#Func_set = [(F.Img_Sum, 2), (F.Img_Sub, 2), (F.Img_Multi, 2), (F.Img_Div, 2), (F.DX,1), (F.DY,1), (F.Filter_Gaussian,1), (F.Log_Image,1), (F.Exp_Image,1), (F.Half_Image,1), (F.Sqrt_Image,1), (F.Gabor0_Image,1), (F.Gabor45_Image,1) , (F.Gabor90_Image,1)   ]
#Func_set = [(F.Img_Sum, 2), (F.Img_Sub, 2), (F.Img_Multi, 2), (F.Img_Div, 2), (F.DX,1), (F.DY,1), (F.Filter_Gaussian,1), (F.Log_Image,1), (F.Exp_Image,1), (F.Half_Image,1), (F.Sqrt_Image,1)]
Func_set = [(F.Img_Sum, 2), (F.Img_Sub, 2), (F.Img_Multi, 2), (F.Img_Div, 2), (F.DX,1), (F.DY,1), (F.Filter_Gaussian1,1), (F.Filter_Gaussian2,1), (F.Filter_Gaussian3,1), (F.Filter_Gaussian4,1), (F.Filter_Gaussian5,1), (F.Log_Image,1), (F.Exp_Image,1), (F.Half_Image,1), (F.Sqrt_Image,1)]
Term_set = ["I", "R", "G", "B", "REG", "NIR"]

# ES: Ejecutar el algoritmo genético
# EN: Run GA
best_tree = genetic_algorithm(population_size=200, generations=100, max_depth=15, crossover_rate=0.7, mutation_rate=0.3)
#best_tree = genetic_algorithm(population_size=5, generations=5, max_depth=5, crossover_rate=0.7, mutation_rate=0.3)
print("\nFinal Best Tree:")
print_individual(best_tree)
print("\nFinal Visual Representation:")
print_tree_visual(best_tree)

# 1.
# ✅ Agregar selección por torneo.

# 2.
# ✅ Guardar en cada generación en un CSV.
# ✅ Cambiar las probabilidades de mutación.

# 3.
# ✅ Agregar Overall Acuracy.
# ✅ Entrenar el individuo con las 4 carpetas de entrenamiento.
# ✅ Retornar: Recall, Fitness, Recall, Overall acuracy.
# ✅ Validar crossover y mutation.
# ✅ Agregar más funciones.

# 4. 
# ✅ Ejecutar el código 10 veces y probar el mejor individuo generado de cada ejecución.

# 5. 
# ✅ Guardar cada individuo para poder ser ejecutado en otro archivo.
