import numpy as np

class Node(object):
    
    def __init__(self, name, coordenadas, coordenadas2,puntos):
        self.name = name
        self.neighbors = []
        self.coordenadas = coordenadas
        self.coordenadas2 = coordenadas2
        self.x1x2y1y2 = puntos
        
    def __repr__(self):
        return self.name
    
def find_cliques(potential_clique=[], remaining_nodes=[], skip_nodes=[], depth=0,cliques_list=[]):
    if len(remaining_nodes) == 0 and len(skip_nodes) == 0:
        cliques_list.append(potential_clique)
        return 1
    found_cliques = 0
    for node in remaining_nodes:
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in node.neighbors]
        new_skip_list = [n for n in skip_nodes if n in node.neighbors]
        found_cliques += find_cliques(new_potential_clique, new_remaining_nodes, new_skip_list, depth + 1,cliques_list)
        remaining_nodes.remove(node)
        skip_nodes.append(node)
    return found_cliques

def centroid1(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    _len = len(points)
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    return [centroid_x, centroid_y]

def Optimizer_V2(cliques_set, limit):

    dict_items = {}
    cliques_list = []
    sorted_cliques = sorted(cliques_set, key=len, reverse=False)
    while len(sorted_cliques) != 0:
        clique_extracted = sorted_cliques.pop()
        cliques_list.append(clique_extracted)
        copy_cliques = sorted_cliques

        for element in clique_extracted:
            dict_items[element] = dict_items.get(element, 0) + 1
            if dict_items[element] >= limit:
                for cliques_selected in range(len(copy_cliques)):
                    copy_cliques[cliques_selected].discard(element)
                
            copy_cliques = [x for x in copy_cliques if x != set()]
            discard_cliques = []
            
            for cliques_selected_1 in range(len(copy_cliques)):
                exit_subset = False
                index = 0
                while index < len(copy_cliques) and exit_subset == False:
                   if cliques_selected_1 != index:
                       a = copy_cliques[cliques_selected_1]
                       b = copy_cliques[index]
                       if b.issubset(a):
                          exit_subset = True
                   index = index + 1
                if exit_subset == False:
                      discard_cliques.append(copy_cliques[cliques_selected_1])
            sorted_cliques = discard_cliques

    return cliques_list



def bron_kerbosch(adj_matrix : np.ndarray, pivot : bool = False) -> list:
    maximal_cliques = []

    def N(v):
        return {i for i, weight in enumerate(adj_matrix[v]) if weight}

    def _bron_kerbosch(R, P, X):
        if not P and not X:
            maximal_cliques.append(R)
        else:
            if pivot:
                u = max(P | X, key=lambda i: len(N(i)))
                _P = P.copy() - N(u)
            else:
                _P = P.copy()

            for v in _P:
                _bron_kerbosch(R | {v}, P & N(v), X & N(v))
                P.remove(v)
                X.add(v)

    R, P, X = set(), set(range(len(adj_matrix))), set()
    _bron_kerbosch(R, P, X)

    return maximal_cliques

