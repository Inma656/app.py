"""Web interactiva para estudiar A* con Streamlit.
Ejecutar: streamlit run app.py
"""
import json
from math import isfinite
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
from astar import a_estrella, generar_grafo, grafo_desde_tablas

st.set_page_config(page_title='A* | Grafo interactivo', layout='wide')
st.title('Algoritmo A* — grafo interactivo')
st.markdown('**Objetivo:** encontrar el camino de coste mínimo entre dos vértices usando '
            '**f(n) = g(n) + h(n)**. Edita las tablas, cambia el número de vértices '
            'y observa las decisiones y el tiempo de ejecución.')

with st.sidebar:
    st.header('Generador')
    n = st.slider('Número de vértices', 2, 150, 50)
    semilla = st.number_input('Semilla aleatoria', min_value=0, max_value=999999, value=42)
    vecinos = st.slider('Vecinos adicionales por vértice', 1, 8, 3)
    if st.button('Generar / restablecer grafo', type='primary') or 'nodos' not in st.session_state:
        base = generar_grafo(n, int(semilla), vecinos)
        st.session_state.nodos = pd.DataFrame([{'vertice': v, 'x': d['x'], 'y': d['y']}
                                                for v, d in base.nodes(data=True)])
        st.session_state.aristas = pd.DataFrame([{'origen': u, 'destino': v, 'peso': d['weight']}
                                                  for u, v, d in base.edges(data=True)])

st.subheader('1. Modifica los vértices y las aristas')
st.caption('Puedes añadir/borrar filas y editar coordenadas o pesos. '
           'Para preservar la garantía de A*, cada peso debe ser al menos la distancia euclídea entre sus extremos.')
c1, c2 = st.columns(2)
with c1:
    st.markdown('**Vértices (identificador y posición)**')
    nodos = st.data_editor(st.session_state.nodos, num_rows='dynamic', hide_index=True,
                           use_container_width=True, key='editor_nodos')
with c2:
    st.markdown('**Aristas no dirigidas (coste de recorrerlas)**')
    aristas = st.data_editor(st.session_state.aristas, num_rows='dynamic', hide_index=True,
                             use_container_width=True, key='editor_aristas')

try:
    grafo = grafo_desde_tablas(nodos, aristas)
    if grafo.number_of_nodes() == 0:
        st.warning('Añade al menos un vértice.')
        st.stop()
    ids = sorted(grafo.nodes)
    c3, c4 = st.columns(2)
    with c3:
        inicio = st.selectbox('Origen', ids, index=0)
    with c4:
        meta = st.selectbox('Destino', ids, index=len(ids)-1)

    resultado = a_estrella(grafo, inicio, meta)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric('Vértices', grafo.number_of_nodes())
    t2.metric('Aristas', grafo.number_of_edges())
    t3.metric('Tiempo A*', f'{resultado.tiempo_ms:.4f} ms')
    t4.metric('Nodos expandidos', len(resultado.expandidos))

    if resultado.camino:
        st.success(f'Camino encontrado: {" → ".join(map(str, resultado.camino))} | '
                   f'Coste total: {resultado.coste:.4f}')
        # Comparación independiente para validar que es el camino óptimo.
        optimo = nx.shortest_path_length(grafo, inicio, meta, weight='weight', method='dijkstra')
        st.write(f'**Verificación con Dijkstra:** coste óptimo = {optimo:.4f}. '
                 f'¿Coincide A*? **{"Sí" if abs(optimo-resultado.coste)<1e-7 else "No"}**')
    else:
        st.warning('No hay camino entre esos vértices. El grafo puede estar desconectado.')

    fig, ax = plt.subplots(figsize=(12, 8))
    posiciones = {v: (d['x'], d['y']) for v, d in grafo.nodes(data=True)}
    nx.draw_networkx_edges(grafo, posiciones, ax=ax, edge_color='#cbd5e1', width=1.0)
    nx.draw_networkx_nodes(grafo, posiciones, ax=ax, node_size=170, node_color='#93c5fd')
    nx.draw_networkx_labels(grafo, posiciones, ax=ax, font_size=7)
    if resultado.camino:
        ruta = list(zip(resultado.camino[:-1], resultado.camino[1:]))
        nx.draw_networkx_edges(grafo, posiciones, edgelist=ruta, ax=ax,
                               edge_color='#dc2626', width=3.3)
        nx.draw_networkx_nodes(grafo, posiciones, nodelist=resultado.camino, ax=ax,
                               node_color='#fca5a5', node_size=220)
    nx.draw_networkx_nodes(grafo, posiciones, nodelist=[inicio], ax=ax,
                           node_color='#16a34a', node_size=340)
    nx.draw_networkx_nodes(grafo, posiciones, nodelist=[meta], ax=ax,
                           node_color='#f59e0b', node_size=340)
    ax.set_title('Verde = origen | Naranja = destino | Rojo = camino A*')
    ax.set_axis_off()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader('2. Decisiones que toma A*')
    st.markdown('**g(n)**: coste desde el origen hasta n. **h(n)**: estimación '
                'euclídea desde n al destino. **f(n)=g(n)+h(n)**: prioridad. '
                'Se expande el nodo pendiente con menor f, se mejoran los vecinos '
                'si disminuye g y se termina al extraer el destino.')
    st.dataframe(pd.DataFrame(resultado.decisiones), hide_index=True, use_container_width=True)
    st.download_button('Descargar decisiones CSV',
                       pd.DataFrame(resultado.decisiones).to_csv(index=False).encode('utf-8'),
                       'decisiones_astar.csv', 'text/csv')
    grafo_json = {'nodos': nodos.to_dict(orient='records'),
                  'aristas': aristas.to_dict(orient='records')}
    st.download_button('Descargar grafo JSON', json.dumps(grafo_json, indent=2),
                       'grafo.json', 'application/json')
    st.caption('El tiempo mide solo la búsqueda A*, no el dibujo ni la interfaz; '
               'varía según el equipo. Para grafos desconectados no siempre existe ruta.')
except (ValueError, TypeError, KeyError, nx.NetworkXError) as error:
    st.error(f'Corrige la configuración del grafo: {error}')
