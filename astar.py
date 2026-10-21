"""A* para un grafo no dirigido ponderado con posiciones (x,y).
Los pesos deben ser >= distancia euclídea para garantizar que h es admisible.
"""
from __future__ import annotations
from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot, inf
from time import perf_counter
import networkx as nx


@dataclass
class Resultado:
    camino: list[int]
    coste: float
    tiempo_ms: float
    expandidos: list[int]
    decisiones: list[dict]


def heuristica(grafo: nx.Graph, nodo: int, meta: int) -> float:
    """Distancia en línea recta: cota inferior del coste real restante."""
    x1, y1 = grafo.nodes[nodo]['x'], grafo.nodes[nodo]['y']
    x2, y2 = grafo.nodes[meta]['x'], grafo.nodes[meta]['y']
    return hypot(x2 - x1, y2 - y1)


def reconstruir(padres: dict, meta: int) -> list[int]:
    camino = [meta]
    while camino[-1] in padres:
        camino.append(padres[camino[-1]])
    return camino[::-1]


def a_estrella(grafo: nx.Graph, inicio: int, meta: int) -> Resultado:
    """A* con cola de prioridad; registra cada decisión útil para explicar el algoritmo."""
    if inicio not in grafo or meta not in grafo:
        raise ValueError('El origen y el destino deben existir en el grafo.')
    t0 = perf_counter()
    g = {inicio: 0.0}  # Mejor coste conocido desde el origen
    padres = {}         # Para reconstruir el camino
    cola = []           # Tuplas (f, desempate, nodo, g_al_insertar)
    contador = 0
    heappush(cola, (heuristica(grafo, inicio, meta), contador, inicio, 0.0))
    expandidos, decisiones = [], []

    while cola:
        f_actual, _, actual, g_insertado = heappop(cola)
        # Ignora entradas obsoletas tras mejorar un camino a ese nodo
        if g_insertado > g.get(actual, inf) + 1e-10:
            continue
        h_actual = heuristica(grafo, actual, meta)
        expandidos.append(actual)
        decisiones.append({'paso': len(decisiones)+1, 'accion': 'EXPANDIR',
                           'nodo': actual, 'desde': '', 'g': round(g[actual], 4),
                           'h': round(h_actual, 4), 'f': round(g[actual]+h_actual, 4),
                           'explicacion': 'Menor f=g+h de la frontera'})
        if actual == meta:
            return Resultado(reconstruir(padres, meta), g[meta],
                             (perf_counter()-t0)*1000, expandidos, decisiones)

        for vecino, atributos in sorted(grafo[actual].items()):
            peso = float(atributos['weight'])
            if peso < 0:
                raise ValueError('A* requiere pesos no negativos.')
            tentativa = g[actual] + peso
            if tentativa + 1e-10 < g.get(vecino, inf):
                g[vecino] = tentativa
                padres[vecino] = actual
                h_vecino = heuristica(grafo, vecino, meta)
                contador += 1
                heappush(cola, (tentativa+h_vecino, contador, vecino, tentativa))
                decisiones.append({'paso': len(decisiones)+1, 'accion': 'MEJORAR',
                                   'nodo': vecino, 'desde': actual,
                                   'g': round(tentativa, 4), 'h': round(h_vecino, 4),
                                   'f': round(tentativa+h_vecino, 4),
                                   'explicacion': 'Nuevo coste g menor: actualizar padre y prioridad'})
            else:
                decisiones.append({'paso': len(decisiones)+1, 'accion': 'DESCARTAR',
                                   'nodo': vecino, 'desde': actual,
                                   'g': round(tentativa, 4),
                                   'h': round(heuristica(grafo, vecino, meta), 4),
                                   'f': round(tentativa+heuristica(grafo, vecino, meta), 4),
                                   'explicacion': 'No mejora el mejor coste conocido'})
    return Resultado([], inf, (perf_counter()-t0)*1000, expandidos, decisiones)


def generar_grafo(n: int = 50, semilla: int = 42, k: int = 3) -> nx.Graph:
    """Grafo conexo espacial reproducible: árbol vecino cercano + k vecinos.
    Pesos = distancia euclídea; así la heurística es admisible y consistente.
    """
    import random
    if not 2 <= n <= 300:
        raise ValueError('Se permiten de 2 a 300 vértices.')
    rng = random.Random(semilla)
    grafo = nx.Graph()
    for i in range(n):
        grafo.add_node(i, x=rng.uniform(0, 100), y=rng.uniform(0, 100))
    def distancia(a, b):
        return heuristica(grafo, a, b)
    # Aseguramos conectividad enlazando cada vértice al más cercano ya incorporado.
    for i in range(1, n):
        j = min(range(i), key=lambda v: distancia(i, v))
        grafo.add_edge(i, j, weight=distancia(i, j))
    # Añadimos alternativas de recorrido conectando con los vecinos más cercanos.
    for i in range(n):
        vecinos = sorted((j for j in range(n) if j != i), key=lambda j: distancia(i, j))[:k]
        for j in vecinos:
            grafo.add_edge(i, j, weight=distancia(i, j))
    return grafo


def grafo_desde_tablas(nodos, aristas) -> nx.Graph:
    """Crea grafo a partir de tablas editadas, y valida pesos/heurística."""
    grafo = nx.Graph()
    for _, r in nodos.iterrows():
        v = int(r['vertice'])
        if v in grafo:
            raise ValueError(f'El vértice {v} está duplicado.')
        grafo.add_node(v, x=float(r['x']), y=float(r['y']))
    for _, r in aristas.iterrows():
        u, v, w = int(r['origen']), int(r['destino']), float(r['peso'])
        if u not in grafo or v not in grafo:
            raise ValueError(f'Arista {u}-{v}: hay vértices inexistentes.')
        if u == v:
            raise ValueError('No se permiten lazos (origen=destino).')
        if w + 1e-8 < heuristica(grafo, u, v):
            raise ValueError(f'Arista {u}-{v}: peso {w:.3f} inferior a distancia euclídea '
                             f'{heuristica(grafo, u, v):.3f}. Aumenta el peso para garantizar optimalidad.')
        grafo.add_edge(u, v, weight=w)
    return grafo
