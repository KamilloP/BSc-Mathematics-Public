import numpy as np
import directed_nonlinear_laplacian_module as nlin
import directed_chung_laplacian_module as chung
import networkx as nx
import plotly.graph_objects as go

import math

def show_and_save_graph_directed(E, temperatury_czasowe, name_of_scheme):
    """
    Teraz nie ma zapisu, bo zakomentowana linia na dole.
    Wizualizacja grafu 3D z kolorem zależnym od temperatury wierzchołka.
    
    Parametry:
    -----------
    E : numpy.array
        Macierz sąsiedztwa grafu skierowanego.
    temperatury_czasowe : numpy.array
        Temperatura w wierzchołkach w czasie. Rozmiar: (liczba_klatek, liczba_wierzchołków).
    name_of_scheme : str
        Nazwa schematu wykresu (np. "Implicit Euler scheme").
    """
    G = nx.from_numpy_array(E, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, dim=3, seed=42)
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    z_nodes = [pos[node][2] for node in G.nodes()]

    x_edges, y_edges, z_edges = [], [], []
    for edge in G.edges():
        x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])
        z_edges.extend([pos[edge[0]][2], pos[edge[1]][2], None])

    edge_trace = go.Scatter3d(
        x=x_edges, 
        y=y_edges, 
        z=z_edges, 
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='none'
    )

    frames = []
    for t, temps in enumerate(temperatury_czasowe):
        node_trace = go.Scatter3d(
            x=x_nodes, 
            y=y_nodes, 
            z=z_nodes, 
            mode='markers',  
            marker=dict(
                size=15,
                color=temps.reshape(-1), 
                colorscale='Viridis',
                colorbar=dict(title="Temp (°C)")
            ),
            text=[f"Node {node}: temp = {temp[0]:.2f} °C" for node, temp in enumerate(temps)],  
            hoverinfo='text'  
        )
        
        frame = go.Frame(data=[edge_trace, node_trace], name=f"Time {t}")
        frames.append(frame)

    layout = go.Layout(
        title=f"Visualization of vertices temperature in time ({name_of_scheme})",
        width=800,
        height=800,
        scene=dict(
            xaxis=dict(title="X Axis"),
            yaxis=dict(title="Y Axis"),
            zaxis=dict(title="Z Axis")
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "fromcurrent": True}])]
        )],
        sliders=[{
            'active': 0,  
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Time: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 500, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,  
            'x': 0.1,  
            'y': 0,  
            'steps': [
                {
                    'args': [[f"Time {t}"], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate", "transition": {"duration": 500}}],
                    'label': str(t),
                    'method': 'animate'
                } for t in range(len(temperatury_czasowe))
            ]
        }]
    )

    fig = go.Figure(data=[edge_trace, frames[0]['data'][1]], frames=frames, layout=layout)
    # fig.write_html(f"{name_of_scheme}_graph.html") # I do not want to save it right now.
    fig.show()

def analyze_graph(E, T, dt, scheme, name, x=None):
    if T < 0.:
        print("Bad T!")
        return
    V = E.shape[0]
    print(f"Liczba węzłów: {V}")
    print(f"Liczba krawędzi: {np.sum(E)}")
    if x is None:
        dplus = np.sum(E, axis=1)
        dminus = np.sum(E, axis=0)
        d = dplus+dminus
        x = np.sqrt(d).reshape(-1,1)
        print(x.shape)
    x_values = scheme(E, x, dt, T, 100)
    print("Last temperatures:", x_values[-1])
    show_and_save_graph_directed(E, x_values, name)

###################################################################################
############################ EXAMPLE GRAPHS AND SAVING ############################
###################################################################################

def save_graph(E, filepath):
    with open(filepath, 'w') as f:
        for i in range(len(E)):
            for j in range(len(E[i])):
                if math.isclose(E[i][j], 1.):  # Jeśli istnieje krawędź
                    f.write(f"{i} {j} 0\n")

def create_2_connected_cliques():
    """
    Creates directed graf with 2 cliques (no loops)
    and some edges between those 2 cliques, in both directions.
    Both cliques have 100 nodes. This graph is deterministic.
    Details are below.
    Returns: graph as adjacency matrix, type: numpy.array, float numbers, shape = (200,200). 
    """
    n = 200  # Liczba wierzchołków
    adj_matrix = np.zeros((n, n), dtype=np.float64)  # Inicjalizujemy macierz sąsiedztwa jako macierz zer

    # Dodajemy krawędzie wewnątrz klik
    for i in range(100):
        for j in range(100):
            if i != j:  # Nie dodajemy pętli
                adj_matrix[i, j] = 1.  # Krawędź od i do j

    for i in range(100, 200):
        for j in range(100, 200):
            if i != j:
                adj_matrix[i, j] = 1.  # Krawędź od i do j

    # Dodajemy krawędź między klastrami:
    # (0 -> 100)

    adj_matrix[0, 100] = 1.
    adj_matrix[199, 1] = 1.
    adj_matrix[50, 150] = 1.
    adj_matrix[50, 116] = 1.
    adj_matrix[60, 101] = 1.
    adj_matrix[180, 50] = 1.

    return adj_matrix

def create_random_almost_2_connected_cliques(V1, V2, V1_to_V2, V2_to_V1, p1, p2):
    """
    V1 - nr of vertices in first part of graph (first almost clique)
    V2 - nr of vertices in second part of graph (second almost clique)
    V1_to_V2 - max nr of edges going from V1 to V2
    V2_to_V1 - max nr of edges going from V2 to V1
    p1 - probability of edge between to vertices from V1
    p2 - probability of edge between to vertices from V2
    """
    E = np.zeros((V1+V2, V1+V2)).astype(np.float64)
    E1 = (np.random.uniform(0., 1., (V1, V1)) >= p1).astype(np.float64)
    E2 = (np.random.uniform(0., 1., (V2, V2)) >= p2).astype(np.float64)
    E[0:V1, 0:V1] = E1
    E[V2:V1+V2, V2:V1+V2] = E2
    for _ in range(V1_to_V2):
        i = np.random.randint(0, V1)
        o = np.random.randint(V2, V1+V2)
        E[i,o] = 1.
    for _ in range(V2_to_V1):
        o = np.random.randint(0, V1)
        i = np.random.randint(V2, V1+V2)
        E[i,o] = 1.
    return E

def largest_scc(E):
    """
    Znajduje największą silnie spójną składową w skierowanym grafie.
    Pod spodem używa algorytmu Tarjana. Inny algorytm, który mógłby być użyty to np. Kosaraju.
    
    :param E: Macierz sąsiedztwa reprezentująca graf
    :return: Lista wierzchołków należących do największej SCC
    """
    assert len(E.shape) == 2
    assert E.shape[0] == E.shape[1]
    n = E.shape[0]
    G = nx.DiGraph()
    for i in range(len(E)):
        for j in range(len(E[i])):
            if not math.isclose(E[i][j], 0):
                G.add_edge(i, j)
    
    sccs = list(nx.strongly_connected_components(G))
    largest = max(sccs, key=len)
    vertices = list(largest)
    # print(vertices)
    l = len(vertices)
    newE = np.zeros((l, l)).astype(np.float64)
    for i in range(l):
        for j in range(l):
            if math.isclose(E[vertices[i], vertices[j]], 1.):
                newE[i,j] = 1.
    return newE, vertices

    