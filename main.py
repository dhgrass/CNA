import numpy as np
from scipy.linalg import lu, qr, expm, fractional_matrix_power
import networkx as nx
import matplotlib.pyplot as plt


def communicability_between_vertices(G, v, w, beta=1):
    """
    Calcula la comunicabilidad entre dos nodos v y w en un grafo G.
    Ecuación 2.1 del paper de Estrada
    
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
    Ecuación 2.2 del paper de Estrada
    
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
    
    # Definimos las artistas del grafo
    edges = [(1,2),
             (2,3),
             (2,5),
             (2,6),
             (3,4),
             (3,6),
             (4,5),
             (4,6),
             (5,7)]
    
    # Creamos el grafo indexado con las aristas anteriores
    G, node_indices = create_indexed_graph(edges)

    listProximity = []
    # Itera sobre cada par de vértices para calcular la proximidad
    for v in G.nodes():
        vint = int(v)
        for w in G.nodes():
            wint = int(w)
            if vint < wint:  # Only consider the upper triangular part
                pair = (f"{vint} - {wint}")
                if pair not in [x[0] for x in listProximity]:
                    listProximity.append((pair, calculate_proximity_measures(G, vint, wint)[1]))
    listProximity.sort(key=lambda x: x[1], reverse=False)
    
    print("Proximidad entre nodos")
    for i in listProximity:
        print(i)

    print("Communicability Cosine Centrality")
    for i in range(len(G.nodes())):
        print(i+1, calculate_CCC(G, i))

    # # Calcula e imprime la CCC para cada vértice en el grafo
    # for v in G.nodes():
    #     ccc = calculate_CCC(G, int(v)-1)
    #     print(f"CCC para el vértice {v}: {ccc}")


    # # Dibuja el grafo
    # nx.draw(G, with_labels=True)
    # plt.show()

    




