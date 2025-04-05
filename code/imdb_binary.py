from torch_geometric.datasets import TUDataset
import numpy as np
from oct2py import Oct2Py
import plotly.graph_objects as go
import networkx as nx
from torch_geometric.utils import to_networkx

path = './datasets'

dataset = TUDataset(root=path, name='IMDB-BINARY')

# We can see that graphs in dataset have not enough nodes.
# I will check better candidate later: FIRSTMM_DB

# Information about dataset
print(f"Liczba grafów: {len(dataset)}")
print(f"Pierwszy graf:\n{dataset[0]}")

def show_and_save_graph(data, heat_values):

    G = to_networkx(data, to_undirected=True)

    # Step 3: Use a 3D layout for NetworkX to get 3D positions for nodes
    pos = nx.spring_layout(G, dim=3)  # 'spring_layout' often works well for 3D graphs

    # Extract node positions for plotting
    node_x = [pos[i][0] for i in G.nodes()]
    node_y = [pos[i][1] for i in G.nodes()]
    node_z = [pos[i][2] for i in G.nodes()]

    # Step 4: Define Plotly traces for nodes and edges

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=5,
            color=heat_values,  # Koloryzacja na podstawie wartości ciepła
            colorscale='YlOrRd',  # Skala kolorów: od żółtego (zimny) do czerwonego (gorący)
            colorbar=dict(title="Heat"),  # Dodanie paska kolorów
            opacity=0.8
        ),
        hoverinfo='text',
        text=[f"Node {i}: Heat={heat_values[i]:.2f}" for i in range(len(G.nodes))]  # Tekst hover
    )

    # Edge trace (bez zmian)
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=1),
        hoverinfo='none'
    )

    # Step 5: Plot
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        title="3D Heat Distribution in Graph"
    )
    fig.show()

def analyze_graph(data, T):
    print(f"Liczba węzłów: {data.num_nodes}")
    print(f"Liczba krawędzi: {data.num_edges}")
    print(f"Etykieta: {data.y}")
    # Graph info
    E = data.edge_index
    m = data.num_edges
    n = data.num_nodes
    print(E)
    # State-transition matrix
    P = np.zeros((n, n))
    for i in range(E.size(1)):
        x = E[0, i].item()
        y = E[1, i].item()
        P[x, y] = 1
    print(P[0,4])
    mu = np.sum(P, axis=1)
    P = P / mu.reshape(-1, 1)
    # print(P[0])
    # print(np.sum(P[0]) == 1.)
    L = -(np.identity(n) - P)
    u = np.zeros(n)
    u[0] = 1
    show_and_save_graph(data, u) #
    u = u.reshape(-1, 1)
    dt = 0.001
    # Schemat jawny Eulera
    # u(t+dt) = u(t) + dt Lu(t) <==>
    # u(t+dt) = (I + dt L) u(t)
    M = np.identity(n)+dt*L
    t = 0
    while t < T:
        t += dt
        u = M @ u
    print("Schemat jawny: u(" + str(T) + "s) = ", u.reshape(-1))
    print("Sumaryczne ciepło (schemat jawny):", np.sum(u))
    show_and_save_graph(data, u.reshape(-1))
    # Schemat zamknięty Eulera
    # u(t+dt) = u(t) + dt Lu(t+dt) <==>
    # (I - dtL) u(t+dt) = u(t)
    Mz = np.identity(n) - dt*L
    # I will use Octave:
    oc = Oct2Py()
    oc.push('M', Mz)
    uz = np.zeros(n)
    uz[0] = 1
    uz = uz.reshape(-1,1)
    oc.push('u', uz)
    oc.push('num_steps', T/dt)
    oc.eval("""
        [L, U, P] = lu(M);           % Rozkład LU macierzy M

        for step = 1:num_steps
            u = U \\ (L \\ (P * u)); % Rozwiąż M * u_nowe = u_stare
        end
    """)
    uz = oc.pull('u')
    print("Schemat zamknięty: u(" + str(T) + "5s) = ", uz.reshape(-1))
    print("Sumaryczne ciepło (schemat zamknięty):", np.sum(uz))
    show_and_save_graph(data, uz.reshape(-1))
    return

for i, data in enumerate(dataset):
    if data.num_nodes > 100:
        print(f"Graf {i}:")
        analyze_graph(data, 10.)
