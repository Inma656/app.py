"""Ejecutar sin web: python ejemplo.py. Guarda una imagen de la solución."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from astar import a_estrella, generar_grafo

G = generar_grafo(n=50, semilla=42, k=3)
inicio, meta = 0, 49
res = a_estrella(G, inicio, meta)
print(f'Vertices: {G.number_of_nodes()}, aristas: {G.number_of_edges()}')
print('Camino:', ' -> '.join(map(str, res.camino)) if res.camino else 'sin ruta')
print(f'Coste: {res.coste:.4f}, tiempo A*: {res.tiempo_ms:.4f} ms')
print(f'Vertices expandidos: {len(res.expandidos)}')
print('Decisiones:')
for d in res.decisiones:
    print(f"{d['paso']:3} | {d['accion']:9} | nodo={d['nodo']:2} | "
          f"g={d['g']:.2f}, h={d['h']:.2f}, f={d['f']:.2f}")
optimo = nx.shortest_path_length(G, inicio, meta, weight='weight')
print('Validado con Dijkstra:', abs(res.coste-optimo) < 1e-8)
pos = {v: (d['x'], d['y']) for v, d in G.nodes(data=True)}
fig, ax = plt.subplots(figsize=(12, 8))
nx.draw(G, pos, ax=ax, with_labels=True, font_size=7, node_size=160,
        node_color='#93c5fd', edge_color='#cbd5e1')
nx.draw_networkx_edges(G, pos, edgelist=list(zip(res.camino, res.camino[1:])),
                       ax=ax, width=3, edge_color='#dc2626')
nx.draw_networkx_nodes(G, pos, nodelist=[inicio], ax=ax, node_size=310, node_color='#16a34a')
nx.draw_networkx_nodes(G, pos, nodelist=[meta], ax=ax, node_size=310, node_color='#f59e0b')
ax.set_title('A*: grafo de 50 vertices (rojo = camino minimo)')
ax.axis('off')
fig.tight_layout()
fig.savefig('solucion_astar.png', dpi=160)
print('Imagen guardada en solucion_astar.png')
