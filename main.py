import numpy as np
from scipy.linalg import lu, qr, expm, fractional_matrix_power, eigsh
import networkx as nx
import matplotlib.pyplot as plt


def communicability_between_vertices(G, v, w, beta=1):
    """
    Calcula la comunicabilidad entre dos nodos v y w en un grafo G.
    Ecuación 2.1 del paper de Communicability Cosine Distance de Ernesto Estrada
    
    Parámetros:
    - G: Un grafo de NetworkX.
    - v: El índice del primer nodo.
    - w: El índice del segundo nodo.
    - beta: Un parámetro empírico, establecido en 1 por defecto.
    
    Retorna:
    - La comunicabilidad entre los nodos v y w.
    """
    # Convierte el grafo en una matriz de adyacencia
    A = nx.to_numpy_array(G)
    
    # Calcula la exponencial de beta*A
    exp_beta_A = expm(beta * A)
    
    # Retorna el elemento (v, w) de la matriz exponencial
    return exp_beta_A[v, w]


def communicability_between_vertices_spectral(G, v, w, beta=1):
    """
    Calcula la comunicabilidad entre dos nodos v y w en un grafo G.
    Ecuación 2.2 del paper de Communicability Cosine Distance de Ernesto Estrada
    
    Parámetros:
    - G: Un grafo de NetworkX.
    - v: El índice del primer nodo.
    - w: El índice del segundo nodo.
    - beta: Un parámetro empírico, establecido en 1 por defecto.
    
    Retorna:
    - La comunicabilidad entre los nodos v y w.
    """
    
    # Convierte el grafo en una matriz de adyacencia
    A = nx.to_numpy_array(G)

    # Calcula los valores y vectores propios de A
    """eigenvalues, eigenvectors = np.linalg.eig(A)""" # es muy importante obtener vectores ortogonales y de norma 1
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Inicializa la matriz exponencial como una matriz de ceros del mismo tamaño que A
    exp_beta_A = np.zeros_like(A, dtype=np.complexfloating)
    
    # Suma sobre todos los valores propios y sus correspondientes vectores propios
    for j in range(len(eigenvalues)):
        lambda_j = eigenvalues[j]
        psi_j = eigenvectors[:, j].reshape(-1, 1)
        exp_beta_A += np.exp(beta * lambda_j) * (psi_j @ psi_j.T.conj())

    # Retorna el valor real de la comunicabilidad entre los nodos v y w
    return exp_beta_A[v, w].real


def communicability_between_vertices_approx(G, v, w, beta=1, k=None):
    """
    Calcula la comunicabilidad entre dos nodos v y w en un grafo G.
    Ecuación 6.7 del paper de Communicability Cosine Distance de Ernesto Estrada

    Parámetros:
    - G: Un grafo de NetworkX.
    - v: El índice del primer nodo.
    - w: El índice del segundo nodo.
    - beta: Un parámetro empírico, establecido en 1 por defecto.
    - k: Número de mayores valores propios y vectores propios a considerar.
         Si es None, se utilizan todos.

    Retorna:
    - La comunicabilidad entre los nodos v y w.
    """

    # Convierte el grafo en una matriz de adyacencia
    A = nx.to_numpy_array(G)

    # Si no se especifica k, usar todos los valores propios
    if k is None:
        k = len(eigenvalues)

    # Calcula los valores y vectores propios de A
    eigenvalues, eigenvectors = eigsh(A, k=k, which='LM')

    # Inicializa la matriz exponencial como una matriz de ceros del mismo tamaño que A
    exp_beta_A = np.zeros_like(A, dtype=np.complexfloating)

    # Suma sobre los k mayores valores propios y sus correspondientes vectores propios
    for j in range(k):
        lambda_j = eigenvalues[j]
        psi_j = eigenvectors[:, j].reshape(-1, 1)
        exp_beta_A += np.exp(beta * lambda_j) * (psi_j @ psi_j.T.conj())

    # Retorna el valor real de la comunicabilidad entre los nodos v y w
    return exp_beta_A[v, w].real


def calculate_proximity_measures(G, v, w, beta=1):
    """
    Calcula las medidas de proximidad ξvw y ζvw entre dos nodos v y w en un grafo G.
    Ecuaciones 2.3 y 2.4 del paper de Estrada
    
    Parámetros:
    - G: Un grafo de NetworkX.
    - v: El índice del primer nodo.
    - w: El índice del segundo nodo.
    - beta: Un parámetro empírico, establecido en 1 por defecto.
    
    Retorna:
    - ξvw: Medida de proximidad que cuenta para la proximidad entre los nodos v y w.
    - ζvw: Otra medida de proximidad basada en la raíz cuadrada de la relación entre Gvw y el producto de Gvv y Gww.
    """
    # Calcula Gvv, Gww, y Gvw usando la función de comunicabilidad entre vértices
    Gvv = communicability_between_vertices(G, v, v, beta)
    Gww = communicability_between_vertices(G, w, w, beta)
    Gvw = communicability_between_vertices(G, v, w, beta)
       
    # Calcula ξvw según la definición dada
    xi_vw = Gvv + Gww - 2*Gvw
    
    # Calcula ζvw según la definición dada
    zeta_vw = (Gvw / np.sqrt(Gvv * Gww))
    
    return xi_vw, zeta_vw


def calculate_xi_zeta(G, v, w):

    """
    Calcula las medidas de proximidad ξvw y ζvw entre dos nodos v y w en un grafo G.
    Ecuaiones 2.7 y 2.8 del paper de Estrada
    
    Parámetros:
    - G: Un grafo de NetworkX.
    - v: El índice del primer nodo.
    - w: El índice del segundo nodo.
    - beta: Un parámetro empírico, establecido en 1 por defecto.
    
    Retorna:
    - ξvw
    - ζvw
    """

    # Convierte el grafo en una matriz de adyacencia
    A = nx.to_numpy_array(G)

    """U = np.linalg.eig(A)[1].T"""  # U^T, la transpuesta de la matriz de vectores propios
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    X = expm(np.diag(eigenvalues) / 2) @ eigenvectors.T
    X_T = X.T
    xv, xw = X_T[v, :], X_T[w, :]

    # Calcula los vectores de posición xv y xw
    """xv = np.exp(-0.5) * U[:, v]  # Vector de posición de v
    xw = np.exp(-0.5) * U[:, w]  # Vector de posición de w"""

    xi_vw = np.linalg.norm(xv - xw)**2

    # Calcula zeta_vw
    zeta_vw = np.dot(xv, xw) / (np.linalg.norm(xv) * np.linalg.norm(xw))

    return xi_vw, zeta_vw


def communicability_cosine_distance_CCD(G, v, w):

    """
    Calcula la distancia del coseno de comunicabilidad cuadrada (CCD) entre dos nodos v y w en un grafo G.
    
    Parámetros:
    - G: Un grafo de NetworkX.
    - v: El índice del primer nodo.
    - w: El índice del segundo nodo.
    
    Retorna:
    - Dvw: La distancia CCD entre los nodos v y w.
    """
    # Convierte el grafo en una matriz de adyacencia
    A = nx.to_numpy_array(G)
    
    # Calcula la exponencial de la matriz de adyacencia
    expA = expm(A)
    
    # Extrae los valores de comunicabilidad específicos
    G_vw = expA[v, w]
    G_vv = expA[v, v]
    G_ww = expA[w, w]
    
    # Calcula el coseno del ángulo de comunicabilidad
    cos_theta_vw = G_vw / np.sqrt(G_vv * G_ww)
    
    # Calcula la distancia coseno de comunicabilidad (CCD)
    D_vw = 2 - 2 * cos_theta_vw
    
    return D_vw


def calculate_CCD(G, v, w):
    """
    Calcula la distancia del coseno de comunicabilidad cuadrada (CCD) entre dos nodos v y w en un grafo G.
    
    Parámetros:
    - G: Un grafo de NetworkX.
    - v: El índice del primer nodo.
    - w: El índice del segundo nodo.
    
    Retorna:
    - Dvw: La distancia CCD entre los nodos v y w.
    """
    # Calcula el coseno del ángulo de comunicabilidad entre v y w
    # cos_theta_vw = calculate_Gvw_xi_zeta(G, v, w)[2]
    cos_theta_vw = calculate_proximity_measures(G, v, w)[1]
    
    # Calcula Dvw según la definición dada
    Dvw = 2 - 2 * cos_theta_vw
    
    return Dvw

def calculate_CCD_matrix(G):
    """
    Calcula la matriz CCD para un grafo G.
    
    Parámetros:
    - G: Un grafo de NetworkX.
    
    Retorna:
    - D: La matriz CCD.
    """
    # Obtiene el número de nodos en el grafo
    n = len(G.nodes())
    # Inicializa la matriz D con ceros
    D = np.zeros((n, n))
    
    # Itera sobre cada par de nodos para calcular Dvw
    for v in range(n):
        for w in range(v, n):  # Empieza desde v para asegurar la simetría
            D_vw = calculate_CCD(G, v, w)
            D[v, w] = D_vw
            D[w, v] = D_vw  # Asegura la simetría
    
    return D

def calculate_CCC(G, v):
    """
    Calcula la centralidad de cercanía basada en la distancia del coseno (CCC) para el vértice v en el grafo G.
    
    Parámetros:
    - G: Un grafo de NetworkX.
    - v: El índice del vértice para el cual calcular la CCC.
    
    Retorna:
    - Cv: La CCC del vértice v.
    """
    # Calcula la matriz CCD para el grafo G
    D = calculate_CCD_matrix(G)
    # print(D)
    
    # Obtiene el número de nodos en el grafo
    n = len(G.nodes())
    
    # Calcula la suma de las distancias CCD desde v a todos los otros vértices
    sum_Dvw = 0
    for w in range(n):
        sum_Dvw += D[v, w]
    
    # Calcula la CCC para el vértice v
    Cv = 1 / sum_Dvw
    
    return Cv


def create_indexed_graph(edges):
    """
    Crea un grafo de NetworkX con índices de nodos explícitos antes de añadir las aristas.
    
    Parámetros:
    - edges: Una lista de tuplas representando las aristas (v, w).
    
    Retorna:
    - G: Un grafo de NetworkX con nodos indexados.
    - node_indices: Un diccionario que asigna índices a los nodos.
    """
    G = nx.Graph()
    nodes = set()
    
    # Recopila todos los nodos únicos a partir de las aristas
    for v, w in edges:
        nodes.add(v)
        nodes.add(w)
    
    # Crea índices para los nodos
    node_indices = {node: idx for idx, node in enumerate(nodes)}
    
    # Añade nodos al grafo
    G.add_nodes_from(node_indices.values())
    
    # Añade aristas al grafo usando los índices
    indexed_edges = [(node_indices[v], node_indices[w]) for v, w in edges]
    G.add_edges_from(indexed_edges)
    
    return G, node_indices


if __name__ == "__main__":

       
    # # Definimos las artistas del grafo
    # Figura 3
    # edges = [(1,2),
    #          (2,3),
    #          (2,5),
    #          (2,6),
    #          (3,4),
    #          (3,6),
    #          (4,5),
    #          (4,6),
    #          (5,7)]
    
    # Fig 7.a
    # Definimos las artistas del grafo
    Fig_7a_edges = [(1,3),(1,8),(2,3),(2,4),(2,5),(3,7),(3,8),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8),(6,7),(6,8)]

    # Fig 7.b
    # Definimos las artistas del grafo
    Fig_7b_edges = [(1,4),(1,5),(1,7),(1,8),(2,3),(2,6),(2,7),(2,8),(3,4),(3,5),(3,6),(4,6),(4,7),(5,6),(5,8),(6,7),(6,8),(7,8)]

    # Fig 7.c
    # Definimos las artistas del grafo
    Fig_7c_edges = [(1,7),(1,8),(2,3),(2,4),(2,5),(3,6),(3,7),(3,8),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8),(6,8)]

    # Fig 7.d
    # Definimos las artistas del grafo
    Fig_7d_edges = [(1,7),(1,8),(2,5),(2,6),(2,7),(2,8),(3,5),(3,6),(3,7),(3,8),(4,5),(4,6),(4,7),(4,8),(5,8),(6,8)]

    # Fig 7.e
    # Definimos las artistas del grafo
    Fig_7e_edges = [(1,3),(1,6),(1,7),(1,8),(2,3),(2,6),(2,7),(2,8),(3,4),(3,5),(4,5),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8),(6,7),(6,8),(7,8)]

    # Fig 7.f
    # Definimos las artistas del grafo
    Fig_7f_edges = [(1,8),(2,8),(3,8),(4,8),(5,7),(6,7),(6,8)]

    # Fig 7.g
    # Definimos las artistas del grafo
    Fig_7g_edges = [(1,6),(2,3),(2,7),(3,8),(4,5),(4,8),(5,7),(6,7),(6,8)]

    # Fig 7.h
    # Definimos las artistas del grafo
    Fig_7h_edges = [(1,3),(1,4),(1,5),(2,6),(2,7),(2,8),(3,4),(3,5),(3,8),(4,6),(4,7),(5,6),(5,7),(6,8),(7,8)]

    edges = []
    edges.append(Fig_7a_edges)
    edges.append(Fig_7b_edges)
    edges.append(Fig_7c_edges)
    edges.append(Fig_7d_edges)
    edges.append(Fig_7e_edges)
    edges.append(Fig_7f_edges)
    edges.append(Fig_7g_edges)
    edges.append(Fig_7h_edges)


    # Lista para almacenar los resultados
    results = []
    etiq = ['a','b','c','d','e','f','g','h']
    j = 0
    # Iterar sobre cada grafo representado en edges
    for graph_edges in edges:
        # Creamos el grafo indexado con las aristas
        G, node_indices = create_indexed_graph(graph_edges)

        # Calculamos la CCC para cada vértice en el grafo
        for i in range(len(G.nodes())):
            results.append(("G_"+etiq[j], i+1, calculate_CCC(G, i)))
        j += 1

    # Imprimir los resultados en forma de tabla
    current_graph = ""
    for graph_edges, vertex, ccc in results:
        if graph_edges != current_graph:
            print(f"\nGraph {graph_edges}:")
            print("Vertex\tCCC")
            current_graph = graph_edges
        print(f"{vertex}\t{ccc}")