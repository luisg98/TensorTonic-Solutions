import numpy as np

def k_means_assignment(points, centroids):
    """
    Atribui cada ponto ao centróide mais próximo.
    """

    # Converter listas para arrays do NumPy
    points = np.array(points)
    centroids = np.array(centroids)

    # Diferença entre cada ponto e cada centróide
    differences = points[:, np.newaxis, :] - centroids

    # Distância euclidiana entre cada ponto e cada centróide
    distances = np.linalg.norm(differences, axis=2)

    # Índice do centróide mais próximo de cada ponto
    assignments = np.argmin(distances, axis=1)

    # Converter para lista Python
    return assignments.tolist()