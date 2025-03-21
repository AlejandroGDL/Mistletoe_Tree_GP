### Genetic Programming System
### By Paras Chopra
### http://www.paraschopra.com
### email: paras1987 [at] gmail [dot] com
### email: paraschopra [at] paraschopra [dot] com

## Note: You may use this Genetic Programming module in any way you wish
## But please do not forget to give me the credit which I deserve


# Added by Leon - CentroGeo 08-2019 For increment recursion limit in SetNodeId
import sys
from math import *
from pickle import *
from random import random, randint, uniform
#from Saliency import pySaliencyMapDefs
from Saliency import pySaliencyMap as SM
from Saliency import main_webcam as sal
# Global
from typing import Any, Union

import cv2
import numpy as np

import Funciones
import os
import rasterio
from rasterio.plot import show

import csv
import matplotlib.pyplot as plt

#sys.setrecursionlimit(3000) #ORIGINAL
#sys.setrecursionlimit(10000) 
sys.setrecursionlimit(10000000)
#sys.setrecursionlimit(1000000000)


class Variables:
    def __init__(self, VarList=[]):
        self.VarDict = {}
        for variable in VarList:
            self.VarDict[variable] = 0  # initially a lot each variable a zero value

    def GetVal(self, var):

        if type("String") == type(var):
            # Leon - CentroGeo 08-2019 - Modification for python 3.7
            # if self.VarDict.has_key(var):
            if var in self.VarDict:
                return self.VarDict[var]
            else:
                return 0
        else:
            return var

    def SetVal(self, var, val):
        # Leon - CentroGeo 08-2019 - Modification for python 3.7
        # if self.VarDict.has_key(var):
        if var in self.VarDict:
            self.VarDict[var] = val

    def __len__(self):
        return len(self.VarDict)

    def keys(self):
        return self.VarDict.keys()

# NodeTypes
# LF:Leaf e.g constant or a variable i.e it comprises of CS and VR
# CS:Constant e.g 1,2,3 etc. ,
# VR: variables A,B,C ,etc. and
# FN: Function eg. ADD, SUB, etc.

class Node:
    NodeTypes = {"FN": 0, "LF": 1, "CS": 2, "VR": 3}
 
    def __init__(self, Value=None, Nodes=[], Type="FN", FuncName=None, Variables=None):
        if Value == "random":
            self.Value = random()
        else:
            self.Value = Value
        self.Nodes = Nodes
        self.NodeValues = []
        self.Type = self.NodeTypes[Type]
        self.FuncName = FuncName
        self.Variables = Variables
        self.Size = 1
        self.Depth = 1
        self.NodeId = 1

    def GetNode(self, nodeno):
        self.SetNodeId()
        RetVal = self.GetNodeTemp(nodeno)
        return RetVal

    def SetNodeId(self, curnumber=1):
        self.NodeId = curnumber
        curnumber += 1
        if self.Nodes:
            for node in self.Nodes:
                curnumber = node.SetNodeId(curnumber)
        return curnumber

    def GetNodeTemp(self, nodeno):
        if nodeno == self.NodeId:
            return self
        if self.Nodes:
            for i in range(0, len(self.Nodes)):
                if self.Nodes[i].GetNodeTemp(nodeno) != None:
                    return self.Nodes[i].GetNodeTemp(nodeno)
        return None

    # def GetNodeTemp(self, nodeno):
    #     stack = [self]
    #     while stack:
    #         node = stack.pop()
    #         if node.NodeId == nodeno:
    #             return node
    #         if node.Nodes:
    #             stack.extend(node.Nodes)
    #     return None

    def SetNode(self, nodeno, CopyNode):
        if nodeno == self.NodeId:
            return CopyNode
        if self.Nodes:
            for i in range(0, len(self.Nodes)):
                self.Nodes[i] = self.Nodes[i].SetNode(nodeno, CopyNode)
        return self
    
    def RecalSize(self):
        self.Size = 1
        if self.Nodes:
            for Unit in self.Nodes:
                self.Size += Unit.RecalSize()
        return self.Size

    def ReInit(self):
        self.SetNodeId()
        self.ReCalculate()

    def ReCalculate(self):
        self.Size = 1
        self.Depth = 1
        largest_depth = 1
        if self.Nodes:
            for Unit in self.Nodes:
                Unit.ReCalculate()
                if Unit.Depth > largest_depth:
                    largest_depth = Unit.Depth
                self.Size += Unit.Size
            self.Depth += largest_depth
    
    # Crear función exclusiva para calular la profundidad
    def Depth(self):
        if not self.Nodes:
            return 1
        else:
            return 1 + max(node.Depth() for node in self.Nodes)
        
    # Leon Dozal - CentroGeo - 17/10/2019
    # def Eval(self, R, G, B):
    def Eval(self, Terms):
        self.NodeValues[:] = []
        if self.Type == self.NodeTypes["LF"]:
            return Terms.get(self.Value)
        elif self.Type == self.NodeTypes["CS"]:
            return self.Value
        else:
            for Unit in self.Nodes:
                self.NodeValues.append(Unit.Eval(Terms))
            return self.FuncName(self.NodeValues)

    def PrintTree(self):
        self.DrawTree(1)

    def DrawTree(self, level):
        kIndentText = "|  "
        IndentText = ""
        for n in range(1, level):
            IndentText = IndentText + kIndentText
        self.NodeValues[:] = []
        if self.Type == self.NodeTypes["LF"]:
            print(IndentText, "+--[", self.Value, "]", self.NodeId)

        elif self.Type == self.NodeTypes["CS"]:
            print(IndentText, "+--[", str(self.Value), "]", self.NodeId)

        else:
            print(IndentText, "+--", self.FuncName.__name__, self.NodeId)
            for i in range(0, len(self.Nodes)):
                self.Nodes[i].DrawTree(level + 1)

class Program:
    NodeTypes = {"FN": 0, "LF": 1, "CS": 2, "VR": 3}

    def RandomTree(self, FuncDict, Terminals, depth):
        if self.MaxDepth == 1:
            NodeUse = self.NodeTypes["LF"]
        elif depth == 1:
            NodeUse = self.NodeTypes["FN"] 
        elif depth == self.MaxDepth:
            NodeUse = self.NodeTypes["LF"]
        else:
            NodeUse = randint(0, 1)
        if NodeUse == self.NodeTypes["FN"]:
            childFuncList = []
            FuncToUse: Union[int, Any] = randint(0, len(list(FuncDict)) - 1)

            # Leon - CentroGeo 08-2019 - Modification for python 3.7
            for i in range(0, list(FuncDict.values())[FuncToUse]):
                child = self.RandomTree(FuncDict, Terminals, depth + 1)
                if not child:
                    print("Error: Child is nonetype")
                    break
                childFuncList.append(child)
            # Leon - CentroGeo 08-2019 - Modification for python 3.7
            return Node(None, childFuncList, "FN", list(FuncDict)[FuncToUse], Terminals)
        else:
            # there is 50/50 chance that leaf would be variable or constant
            #if randint(0, 1) == 0:
                # leaf would be constant
                #TermToUse = randint(0, len(self.TerminalList) - 1)
                #return Node(list(self.TerminalList)[TermToUse], None, "CS", None, self.Variables)
                # return Node(self.TerminalList[TermToUse], None, "CS", None, self.Variables)
            #else:
            # leaf would be a variable
            VarToUse = randint(0, len(Terminals) - 1)
            # Leon - CentroGeo 08-2019 - Modification for python 3.7
            return Node(list(Terminals)[VarToUse], None, "LF", None, Terminals)

    # Append tree
    def __init__(self, Func_set, Term_set, MaxDepth=10):
        self.MaxDepth = MaxDepth
        self.Fitness = 0
        # ADD Ramped Init initialization method #########################
        self.Tree = self.RandomTree(Func_set, Term_set, 1)
        self.Tree.ReInit()

    def EvalTree(self, Number_Folder):
        print("\n ===== Next Individual =====")
        self.PrintTree()

        tree_str = self.TreetoStr(self.Tree)
        #print(tree_str)

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

            Img = Funciones.Multiband_Convertation(os.path.join(image_directory, img1))
            Img_mask = Funciones.Multiband_Convertation(os.path.join(image_directory, img2))

            # ES: Imagenes en diferentes bandas
            # EN: Images in different bands
            R = Img[0] # Red
            G = Img[1] # Green
            B = Img[2] # Blue
            I = np.stack(Img[:3], axis=-1)
            I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)  # Gray

            # ES: Ejecutar el tree_str como expresión de Python
            # EN: Execute the tree_str as a Python expression
            Res = eval(tree_str, {"Funciones": Funciones, "I": I, "R": R, "G": G, "B": B})
            
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

            print(f"True Positives: {TP}")
            print(f"True Negatives: {TN}")
            print(f"False Positives: {FP}")
            print(f"False Negatives: {FN}")
            print(f"Sumatoria: {np.sum([TP, TN, FP, FN])}"  )

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

            print(f"Precision: {precision} Recall: {recall} F-measure: {f_measure}")

            #
            Fitness.append(f_measure)
            Precision.append(precision)
            Recall.append(recall)

        #Promedio Fitness
        Avg_Fitness = np.mean(Fitness)
        Avg_Precision = np.mean(Precision)
        Avg_Recall = np.mean(Recall)

        return Avg_Fitness,Avg_Precision,Avg_Recall

    def PrintTree(self):
        self.Tree.PrintTree()

    def Depth(self):
        return self.Tree.Depth

    def Size(self):
        return self.Tree.Size

    def AssignFitness(self, Fitness, Precision, Recall):
        self.Fitness = Fitness
        self.Precision = Precision
        self.Recall = Recall

    def GetNode(self, nodeno):
        return self.Tree.GetNode(nodeno)

    def SetNode(self, NodeNo, CopyNode):
        self.Tree.SetNode(NodeNo, CopyNode)

    # ES: Evaluación arbol preorder
    # EN: Preoder tree evaluation
    def TreetoStr(self, Tree):
        if Tree is None:
            return ""
        if Tree.Type == self.NodeTypes["LF"]:
            return str(Tree.Value)
        elif Tree.Type == self.NodeTypes["CS"]:
            return str(Tree.Value)

        else:
            result = "Funciones." + Tree.FuncName.__name__ + " ("
            for i, child in enumerate(Tree.Nodes):
                if i > 0:
                    result += " , "
                result += self.TreetoStr(child)
            result += " )"
            return result

    def Count_TreeNodes(self):
        return self.Tree.RecalSize()

    def CalcSize(self):
        return self.Tree.ReCalculate()

    def RetCopy(self):
        return self

class Programs:

    def __init__(self, Func_set, Term_set, MaxDepth=1, Population_size=10, MaxGen=50, ReqFitness=99,
                 CrossRate=1, MutRate=0, BirthRate=0.2, HighFitness=100, Verbose=False):
        self.Individuo = []
        self.MaxGen = MaxGen
        self.Population = Population_size
        self.ReqFitness = ReqFitness
        self.CrossRate = CrossRate
        self.MutRate = MutRate
        self.MaxFitness = 0
        self.MaxFitnessProg = None
        self.BirthRate = BirthRate
        self.HighFitness = HighFitness
        self.MaxDepth = MaxDepth
        self.Verbose = Verbose
        #Initial Poblation
        for i in range(0, Population_size):
            self.Individuo.append(Program(Func_set, Term_set, MaxDepth))

    @property
    def MainLoop(self):
        # ES: 
        # EN: 
        for Number_Folder in range(1, 5):
            fitness_history = []
            best_individual_number = -1

            # ES: Se crean la generaciones
            # EN: Create a generation
            for i in range(0, 1 + self.MaxGen):
                print("Generation no:", i)
                generation_fitness = []
                self.MaxFitness = 0  
                Avg_Generation_Fitness = 0

                # Reinciar población cada generación
                # self.Individuo = []
                # for j in range(0, self.Population):
                #     self.Individuo.append(Program(Func_set, Term_set, self.MaxDepth))

                # ES: Crea un archivo CSV de cada generación y guarda cada individuo
                # EN: Create a CSV archive of each generation and save each individual
                with open(f'results/{Number_Folder}/gp_results_generation_{i}.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Individual', 'Fitess', 'Precision', 'Recall', 'Tree'])

                    # ES: Evalua una población
                    # EN: Evaluate a poblation
                    for j in range(0, self.Population):
                        Avg_Fitness, Avg_Precision, Avg_Recall = FitnessFunction(self.Individuo[j], Number_Folder)
                        self.Individuo[j].AssignFitness(Avg_Fitness, Avg_Precision, Avg_Recall)
                        generation_fitness.append(Avg_Fitness)
                        tree_str = self.Individuo[j].TreetoStr(self.Individuo[j].Tree)
                        writer.writerow([j, Avg_Fitness, Avg_Precision, Avg_Recall, tree_str])

                        if Avg_Fitness > self.MaxFitness:
                            self.MaxFitness = Avg_Fitness
                            self.MaxFitnessProg = self.Individuo[j]
                            best_individual_number = j
                        
                    self.Elitism()
                    self.Reproduction()

                # ES: Promedio de fitness por generación
                # EN: Average fitness per generation
                Avg_Generation_Fitness = sum(generation_fitness) / len(generation_fitness)

                # ES: Guardar el promedio y maximo fitness 
                # EN: Save average and max fitness
                with open(f'results/{Number_Folder}/gp_results_generation_{i}.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Average', Avg_Generation_Fitness, '', '', ''])
                    writer.writerow(['Max', self.MaxFitness, '', '', best_individual_number])

                fitness_history.append(Avg_Generation_Fitness)

            # ES: Crear folder de resultados
            # EN: Save results folder
            results_folder = f'results/{Number_Folder}'
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            # ES: Graficar el fitness promedio
            # EN: Plotting Average of fitness
            plt.plot(fitness_history, label='Average Fitness')
            plt.axhline(y=self.MaxFitness, color='r', linestyle='--', label='Max Fitness')
            plt.xlabel('Generación')
            plt.ylabel('Fitness Promedio')
            plt.title('Fitness por Generación')
            plt.legend()
            plt.grid(True)

            # ES: Guardar Historial de fitness en una imagen
            # EN: Save fitness history in an image
            plt.savefig(f'{results_folder}/fitness_plot.png')
            plt.clf()
            #plt.show()

            ### If you want confirmation to continue after each generation uncomment the following
            # ans=raw_input("Do you wanna quit? (1==Yes,0==No)")
            # print ans,":",type(ans)
            # if ans=="1":
            # break

        # self.MaxFitness = 0
        # i = 0
        # for Unit in self.Individuo:
        #     if Unit.Fitness > self.MaxFitness:
        #         best = Unit
        #         self.MaxFitness = best.Fitness
        #         best_number = i
        #     i += 1
        # print("The end of all the generations.")
        # print("The best solution found is Program number: ", str(best_number))
        # best.PrintTree()
        # print("The fitness value is:", FitnessFunction(best))
        return 

    def CrossOver(self):
        parent1 = roulette_selection(self.Individuo)
        parent2 = roulette_selection(self.Individuo)

        print("PADRES seleccionados para cruce:")
        print("Padre 1:")
        parent1.PrintTree()
        print("Padre 2:")
        parent2.PrintTree()

        crossover_point1 = randint(2, parent1.Count_TreeNodes())
        crossover_point2 = randint(2, parent2.Count_TreeNodes())

        print(f"Punto de cruce en el padre 1: {crossover_point1}")
        print(f"Punto de cruce en el padre 2: {crossover_point2}")

        subtree1 = parent1.GetNode(crossover_point1)
        subtree2 = parent2.GetNode(crossover_point2)

        child1 = parent1.RetCopy()
        child2 = parent2.RetCopy()


        child1.SetNode(crossover_point1, subtree2)

        child2.SetNode(crossover_point2, subtree1)

        child1.Tree.ReInit()
        child2.Tree.ReInit()

        if child1.Depth() > self.MaxDepth or child2.Depth() > self.MaxDepth:
            print("Uno de los hijos excede la profundidad máxima, se descartan.")
            return None, None

        print("HIJOS generados por cruce:")
        print("Hijo 1:")
        child1.PrintTree()
        print("Hijo 2:")
        child2.PrintTree()

        return child1, child2
        
    def Mutation(self):
        print("Individuo sin mutar:")
        individual_to_mutate = roulette_selection(self.Individuo)
        #individual_to_mutate.PrintTree()

        mutation_point = randint(1, individual_to_mutate.Count_TreeNodes())
        print(f"Punto de mutación: {mutation_point}")

        Func_set = {Funciones.Img_Sum: 2, Funciones.Img_Sub: 2, Funciones.Img_Multi: 2, Funciones.Img_Div: 2, Funciones.DX: 1, Funciones.DY: 1, Funciones.Filter_Gaussian: 1}
        Term_set = ["I", "R", "G", "B"]
        new_subtree = Program(Func_set, Term_set, self.MaxDepth).Tree

        individual_to_mutate.SetNode(mutation_point, new_subtree)
        individual_to_mutate.ReInit()

        IndivualDepth = individual_to_mutate.Depth()

        # ES: Validar que los hijos no tengan más nodos de los deseados
        # EN: Validate that the children do not exceed the maximum desired depth
        if IndivualDepth > self.MaxDepth:
            print("El individuo mutado excede la profundidad máxima, se descarta.")
            return None
        else:
            print("Individuo MUTADO:")
            individual_to_mutate.PrintTree()
            return individual_to_mutate

    def Reproduction(self):
        new_generation = []

        # ES: Generar una nueva generación 
        # EN: Generate a new generation
        while len(new_generation) < self.Population:
            if random() < self.MutRate:
                mutated_individual = self.Mutation()
                if mutated_individual:
                    new_generation.append(mutated_individual)
            else:
                offspring1, offspring2 = self.CrossOver()
                if offspring1:
                    new_generation.append(offspring1)
                if offspring2:
                    new_generation.append(offspring2)

        # ES: Eliminar al individuo con menos fitness
        # EN: Delete the individual with less fitness
        new_generation.remove(min(new_generation, key=lambda prog: prog.Fitness))

        # ES: Asegurarse de que el mejor individuo pase a la siguiente generación sin modificaciones
        # EN: Ensure the best individual passes to the next generation without modification
        best_individual = max(self.Individuo, key=lambda prog: prog.Fitness)
        new_generation.append(best_individual)

        if len(new_generation) > self.Population:
           new_generation = new_generation[:self.Population]

        self.Individuo = new_generation[:self.Population]

        return 

    def Elitism(self):
        # ES: Encuentra al mejor individuo de la población en curso
        # EN: Find the best individual in the current population
        best_individual = max(self.Individuo, key=lambda prog: prog.Fitness)
        
        # ES: Asegurase que el mejor individuo pase a la siguiente generación sin modificaciones
        # EN: Ensure the best individual passes to the next generation without modification
        self.Individuo.append(best_individual)

    def RemoveIndividual(self):
        # ES: Encuentra al individuo con el fitness más bajo
        # EN: Find the individual with the lowest fitness
        Low_Individual = min(self, key=lambda prog: prog.Fitness)
        # ES: Elimina al individuo 
        # EN: Remove the individual from the population
        self.remove(Low_Individual)

    def RetCopy(self):
        return self


# You just need to modify this function to generate trees of your own choice
def FitnessFunction(Prog, Number_Folder):
    Prog.PrintTree()
    Avg_Fitness,Avg_Precision,Avg_Recall = Prog.EvalTree(Number_Folder)

    return Avg_Fitness,Avg_Precision,Avg_Recall

# ES: Función ruleta para obtener un individuo
# EN: Roulette funtion to obtain an individual
def roulette_selection(population):
    max_fitness = sum([prog.Fitness for prog in population])
    pick = uniform(0, max_fitness)
    current = 0
    for prog in population:
        current += prog.Fitness
        if current > pick:
            return prog

def symbolic_regression(x):
    return (x * x + x + 1)

### Problem Description
# We will try to evolve a tree for Symbolic Regression of a Quadratic Polynomial
# That is the fitness function x^2+x+1 in the range of -1 to 1


if __name__ == "__main__":
    Func_set = {Funciones.Img_Sum: 2, Funciones.Img_Sub:2, Funciones.Img_Multi: 2, Funciones.Img_Div: 2, Funciones.DX: 1, Funciones.DY: 1, Funciones.Filter_Gaussian: 1}
    Term_set = ["I", "R", "G", "B"]
    pr = Programs(Func_set, Term_set, MaxDepth=15, Population_size=10, MaxGen=20)
    pr.MainLoop
    # wait = raw_input("Press any key to terminate....")


#### Sample Usage example
# prg = Programs({COS:1,RANDOM:0}, [1,2],["A","B"])

#### Syntax of the program
# pr=Programs(FuncDict,TerminalList,Variable,MaxDepth=10,Population=100,MaxGen=100,ReqFitness=99,CrossRate=0.9,MutRate=0.1,BirthRate=0.2,HighFitness=100)
# pr.MainLoop()

# pr=Prograns( {function1: no_of_arguments_of_function,...} , [list of leafs or constants], [list of variable names] )

### Description of arguments

# FuncDictis the dictionary of actual function names and the number of arguments it takes 
# TerminalList is the list of terminal constants possible in the tree e.g. [1,2,5,6] or range(5,11) or [1,2,"random"[]
# "random" in the Terminal List produces a number between 0 and 1 and e.g. 0.257522 or 0.444621
# Variable is a list of possible variables in the tree e.g. ["X","Y"] or ["A","B"]
# It is the responsibility of fitness function to supply values to the variables by using syntax:
# Prog.Variables.SetVal(Variable_Name,Variable_Value) e.g Prog.Variables.SetVal("X",10)
# MaxDepth is the maximum depth allowed for the initial trees
# Population is population in each generation. It starts from 0 to 99 i.e If u want 100 individuals then pass 99 as parameter
# MaxGen is the maximum number of generations until the evolution is aborted
# ReqFitness is the fitness level above which if any program is possesing fitness the program is terminated
# In the default case it is 99, i.e if any program has fitness greater than 99, the evolution is aborted and the candidate is termed as best
# CrossRate is the crossover rate, its default value is 0.9 i.e. the crossover is bound to happen 90% of time
# MutRate is the rate of mutation
# BirthRate is the number of new individuals produced per unit of population
# Its default value is 0.2 i.e if the population is 100 then 20 children will be produced per crossover operation
# HighFitness is the highest fitness attainable by the candidate, in default case it is 100
# MaxGen is maximum number of generations
# BirthRate is no of offsprings per 100 population e.g. if BirthRate is 2 and population of current population is 100 then in the next generation only 2 offsprings will be produced

#### Sample Usage example
# prg = Programs({COS:1,RANDOM:0}, [1,2],["A","B"])

#### To define functions of your own
# The functions used in the trees are real world python functions
# So of you want to add a new function such as power(a,b) i.e to calculate a^b
# use the following synatx

# def POWER(ValuesList):
#     ans = ValuesList[0]
#     if ValuesList[1] < 0:
#         return 0
#     for i in range(0, ValuesList[1]):
#         ans = ans * ValuesList[0]
#     return ans

# The fuctions which you will define will always contain only one argument which is ValuesList
# ValuesList is the list of values passed to the function
# In the present case of a^b ValuesList will contain values of a and b
# So ValuesList[0] will represent the first value i.e a
# and ValuesList[1] will represent the second value i.e b

# If your function takes three values then you will also use ValuesList[2]
# If your function does not takes any values such as RANDOM() then the list will be empty

# But observe that only one value can be returned from the function


### Note
# You may also use this module to create instancesof many GPs running simultaneously
# Or use it to run GP elsewhere in your program

