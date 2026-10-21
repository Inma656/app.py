# Algoritmo A* — grafo editable de 50 vértices

Proyecto en Python para la tarea de **Shortest Path Problem (SPP)** del Tema 1 de Algoritmos. Implementa A* desde cero (no llama a `networkx.astar_path`), muestra el camino mínimo en una gráfica y documenta las decisiones, la complejidad y el tiempo medido.

## Ejecución

1. Instala Python 3.10 o superior.
2. En una terminal, dentro de esta carpeta:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abrirá una web local en el navegador (normalmente `http://localhost:8501`). Para ejecutar el ejemplo sin web:

```bash
python ejemplo.py
```

Esto genera `solucion_astar.png` y muestra en consola los pasos y el tiempo.

## Publicar gratis como web y enlazar desde GitHub

1. Crea un repositorio público nuevo en [GitHub](https://github.com/new) y sube estos archivos (`app.py`, `astar.py`, `ejemplo.py`, `requirements.txt`, `README.md`).
2. Entra en [Streamlit Community Cloud](https://share.streamlit.io/) y accede con GitHub.
3. Pulsa **Create app**, elige el repositorio, rama `main` y archivo principal `app.py`, y despliega.
4. Comparte la URL resultante (por ejemplo, `https://nombre-de-app.streamlit.app/`; la URL real la asigna Streamlit). Esa web permitirá editar los vértices y aristas desde el navegador.

## Modelo y decisiones

- **Input:** grafo no dirigido `G=(V,E)`, pesos `w(u,v)` no negativos, origen `s`, destino `t`, posiciones cartesianas `(x,y)`.
- **Output:** camino `s→t` de coste mínimo, suma de pesos, tiempo medido, número de expansiones y tabla de decisiones.
- **g(n):** menor coste descubierto desde `s` hasta `n`.
- **h(n):** distancia euclídea entre `n` y `t`, calculada con sus coordenadas.
- **f(n)=g(n)+h(n):** prioridad de cada nodo en la cola de mínimos.
- **EXPANDIR:** se extrae el nodo con menor `f` pendiente.
- **MEJORAR:** si atravesar una arista reduce `g` de un vecino, se cambia su padre y prioridad.
- **DESCARTAR:** si no reduce `g`, no se modifica nada.
- **TERMINACIÓN:** se alcanza el destino cuando sale de la cola; si se vacía, no hay ruta.

El grafo aleatorio conecta cada vértice con otro vértice ya incorporado para garantizar conectividad, y después añade vecinos cercanos. Los costes son distancias euclídeas. La interfaz permite otros costes, pero exige que el peso de cada arista sea **al menos** su distancia euclídea. Esto garantiza que la heurística nunca sobreestime el coste restante y que sea consistente; por tanto A* devuelve un camino óptimo. Para validar, se compara el coste de A* con Dijkstra.

## Análisis de eficiencia

Con una cola de prioridad binaria, y suponiendo una heurística de coste constante, la implementación tiene cota temporal **O((|V|+|E|) log(|V|+|E|))** (inserciones repetidas y entradas obsoletas); memoria adicional **O(|V|+|E|)** en el peor caso contando la cola y el registro de decisiones. Para un grafo sencillo, `|E|=O(|V|²)`. El tiempo real se mide con `time.perf_counter()` y se presenta en milisegundos; depende del ordenador y de la instancia.

**Importante:** A* resuelve el problema de camino mínimo entre **dos** nodos; no es el problema TSP de visitar todos los nodos. Si la heurística fuera cero, A* se comportaría como Dijkstra. Los vértices numerados de 0 a 49 son identificadores, no un orden de visita.

## Experimento reproducible

Parámetros por defecto: `n=50`, `semilla=42`, `k=3`, `origen=0`, `destino=49`. Cambia `n` o la semilla en la barra lateral y pulsa «Generar / restablecer grafo» para crear otro ejemplo. Puedes editar, añadir y borrar filas en ambas tablas, siempre que las aristas referencien vértices existentes.
