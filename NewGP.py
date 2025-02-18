import random
import Funciones as F
import os

import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

Number_Folder = 1

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

# ES: Funciones y sus terminales
# EN: Functions and terninals
Func_set = [(F.Img_Sum, 2), (F.Img_Sub, 2), (F.Img_Multi, 2), (F.Img_Div, 2), (F.DX,1), (F.DY,1),(F.Filter_Gaussian,1)]
Term_set = ["I", "R", "G", "B"]

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
    print("====================")


    # ES: Obtiene el par de imagenes
    # EN: Get image pairs
    def get_image_pairs(directory):
        files = sorted([f for f in os.listdir(directory) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')])
        return [(files[i], files[i + 1]) for i in range(0, len(files), 2)]

    image_directory = f"/Users/andro/Documents/Repositorios/Mistletoe_Tree_GP/ImagenesEntrenamiento/{Number_Folder}"
    image_pairs = get_image_pairs(image_directory)

    Fitness = []
    Precision = []
    Recall = []

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

        # ES: Ejecutar el tree_str como expresión de Python
        # EN: Execute the tree_str as a Python expression
        Res = eval(tree_str, {"F": F, "I": I, "R": R, "G": G, "B": B})
            
        # ===== PRUEBAS =====
        # ES: Nota, cuando se utilizan las pruebas eliminar el [0] de las líneas 275 - 278 y 289 - 292
        # EN: Note, when using tests remove the [0] of the lines 275 - 278 and 289 - 292
        # Res = np.array([[255, 0], [0, 255]])
        # Img_mask = np.array([[1, 0], [0, 1]])

        # ES: Mediante la mascara se consigue la matriz correcta
        # EN: Using the mask to get the correct matrix
        Res = cv2.normalize(Res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh1 = cv2.threshold(Res, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # ES: Multiplica cada valor de la mascara por 255
        # EN: Each value has been multiply by 255
        Img_mask = [value * 255 for value in Img_mask]

        # ES: Calcula los componentes de la matriz de confusión
        # EN: Calculate confusion matrix components
        TP = np.sum(np.logical_and(thresh1 == 255, Img_mask[0] == 255))
        TN = np.sum(np.logical_and(thresh1 == 0, Img_mask[0] == 0))
        FP = np.sum(np.logical_and(thresh1 == 255, Img_mask[0] == 0))
        FN = np.sum(np.logical_and(thresh1 == 0, Img_mask[0] == 255))

        # print(f"True Positives: {TP}")
        # print(f"True Negatives: {TN}")
        # print(f"False Positives: {FP}")
        # print(f"False Negatives: {FN}")
        # print(f"Sumatoria: {np.sum([TP, TN, FP, FN])}"  )

        # ES: Crea una imagen para visualizar la matriz de confución
        # EN: Create an image to visualize the confusion matrix
        confusion_image = np.zeros_like(thresh1, dtype=np.uint8)
        confusion_image[np.logical_and(thresh1 == 255, Img_mask[0] == 255)] = 255  # TP - White
        confusion_image[np.logical_and(thresh1 == 0, Img_mask[0] == 0)] = 128  # TN - Gray
        confusion_image[np.logical_and(thresh1 == 255, Img_mask[0] == 0)] = 64  # FP - Light Gray
        confusion_image[np.logical_and(thresh1 == 0, Img_mask[0] == 255)] = 192  # FN - Dark Gray

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

        #print(f"Precision: {precision} Recall: {recall} F-measure: {f_measure}")

        Fitness.append(f_measure)
        Precision.append(precision)
        Recall.append(recall)

    # ES: Promedio Fitness, Precisión y Recall
    # EN: Average Fitness, Precision and recall
    Avg_Fitness = np.mean(Fitness)
    Avg_Precision = np.mean(Precision)
    Avg_Recall = np.mean(Recall)

        
    return Avg_Fitness

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

# ES: Cruce
# EN: CrossOver
def crossover(parent1, parent2):
    if isinstance(parent1, LeafNode) or isinstance(parent2, LeafNode):
        return parent1 if random.random() < 0.5 else parent2
    else:
        child = FunctionNode(parent1.function, parent1.arity)
        for i in range(parent1.arity):
            if random.random() < 0.5:
                child.children.append(crossover(parent1.children[i], parent2.children[i]))
            else:
                child.children.append(parent1.children[i] if random.random() < 0.5 else parent2.children[i])
        return child

# ES: Mutación
# EN: Mutation
def mutate(tree, mutation_rate=0.1):
    if random.random() < mutation_rate:
        return generate_random_tree(2)
    if isinstance(tree, FunctionNode):
        for i in range(len(tree.children)):
            tree.children[i] = mutate(tree.children[i], mutation_rate)
    return tree

# ES: Ciclo principal
# EN: Main Loop
def genetic_algorithm(population_size, generations, max_depth=5):

    generation_fitness = []

    # ES: Crear la carpeta 'resultados'
    # EN: Create 'resultados' folder
    if not os.path.exists("resultados"):
        os.makedirs("resultados")

    # ES: Crear población
    # EN: Create population
    population = [generate_random_tree(max_depth) for _ in range(population_size)]

    fitness_history = []

    for generation in range(generations):
        print(f"\Generación {generation}:")

        # ES: Variables inciales
        # EN: Initial variables
        MaxFitness = 0
        best_individual_number = 0
        best_individual = 0

        # print("Mejor Individuo (Texto):")
        # print_individual(population[0])

        # print("Mejor Individuo:")
        # print_tree_visual(population[0])

        # ES: Crea un archivo CSV de cada generación y guarda cada individuo
        # EN: Create a CSV archive of each generation and save each individual
        with open(f'resultados/{Number_Folder}/gp_results_generation_{generation}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Individual', 'Fitness', 'Tree'])

            Avg_Fitness_population = []

            # ES: Evaluación
            # EN: Evaluation
            for ind in population:
                Avg_Fitness = evaluate_individual(ind)

                generation_fitness.append(Avg_Fitness)
                Avg_Fitness_population.append(Avg_Fitness)

                writer.writerow([population.index(ind), Avg_Fitness, ind])

                # ES: Encontrar el individuo con mayor fitness
                # EN: Find the individual with the highest fitness
                MaxFitness = max(generation_fitness)
                best_individual_number = generation_fitness.index(MaxFitness)
                best_individual = population[best_individual_number]

                print(f"Mejor Individuo de la Generación {generation}:")
                print_individual(best_individual)
                print(f"Fitness: {MaxFitness}")
            
            # ES: Elitismo
            # EN: Elitism
            new_population = [best_individual]

            #
            # ES: Promedio de fitness de la población
            # EN: Average fitness of the population
            avg_fitness_population_mean = sum(Avg_Fitness_population) / len(Avg_Fitness_population)

            # ES: Generar una nueva población
            # EN: Generate a new population
            while len(new_population) < population_size:
                parent1 = select_parent(population, avg_fitness_population_mean)
                parent2 = select_parent(population, avg_fitness_population_mean)
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            population = new_population

        # ES: Promedio de fitness por generación
        # EN: Average fitness per generation
        Avg_Generation_Fitness = sum(generation_fitness) / len(generation_fitness)

        # ES: Guardar el promedio y maximo fitness 
        # EN: Save average and max fitness
        with open(f'resultados/{Number_Folder}/gp_results_generation_{generation}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Average', Avg_Generation_Fitness, ''])
            writer.writerow(['Max', MaxFitness, best_individual_number])

        fitness_history.append(Avg_Generation_Fitness)

    # ES: Crear folder de resultados
    # EN: Save results folder
    results_folder = f'resultados/{Number_Folder}'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # ES: Graficar el fitness promedio
    # EN: Plotting Average of fitness
    plt.plot(fitness_history, label='Average Fitness')
    plt.axhline(y=MaxFitness, color='r', linestyle='--', label='Max Fitness')
    plt.xlabel('Generación')
    plt.ylabel('Fitness Promedio')
    plt.title('Fitness por Generación')
    plt.legend()
    plt.grid(True)

    # ES: Guardar Historial de fitness en una imagen
    # EN: Save fitness history in an image
    plt.savefig(f'{results_folder}/fitness_plot.png')
    plt.show()
    plt.clf()    

    return population[0]

# ES: Ejecutar el algoritmo genético
# EN: Run GA
best_tree = genetic_algorithm(population_size=20, generations=5, max_depth=2)
print("\nFinal Best Tree:")
print_individual(best_tree)
print("\nFinal Visual Representation:")
print_tree_visual(best_tree)

# 1.
# Agregar selección por torneo

# 2.
# Guardar en cada generación en un CSV
# 