import pickle


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

# ES: Cargar la población
# EN: Load poblation
with open('individuos/gp_individuals_generation_100.pkl', 'rb') as f:
    population = pickle.load(f)

# ES: Imprimir toda la población
# EN: Print all individuals
# for individual in population:
#     print(individual)

# ES: Cargar el individuo
# EN: Load individual
individual = population[0]

print(individual)


# 