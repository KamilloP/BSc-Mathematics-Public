from torch_geometric.datasets import TUDataset
import numpy as np
from oct2py import Oct2Py
import plotly.graph_objects as go
import networkx as nx
from torch_geometric.utils import to_networkx

def show_and_save_graph(data, heat_values_list, type):

    # BEGIN
    # import torch
    # from torch_geometric.utils import to_networkx
    # import plotly.graph_objects as go
    # import networkx as nx

    # # Sample data creation (for test)
    # from torch_geometric.data import Data

    # edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    # x = torch.tensor([[1.0], [2.5], [3.8]], dtype=torch.float)  # Example heat values for time t=0

    # data = Data(x=x, edge_index=edge_index)

    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, dim=3)  # 3D layout for graph

    # Node positions
    node_x = [pos[i][0] for i in G.nodes()]
    node_y = [pos[i][1] for i in G.nodes()]
    node_z = [pos[i][2] for i in G.nodes()]

    # Sample heat_values list for multiple time steps
    # Each sublist represents the heat values at a particular time step

    # Edge trace (static, as edges don't change over time)
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

    # Generate frames for each time step
    frames = []
    for t, heat_values in enumerate(heat_values_list):
        hv = np.array(heat_values)[0].flatten()
        # print("hv:", hv)
        # Node trace with time-dependent heat-based coloring
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=5,
                color=hv,  # Apply heat values for this time step
                colorscale='YlOrRd',
                cmin=np.min(hv),  # min temp over all time steps
                cmax=np.max(hv),  # max temp over all time steps
                colorbar=dict(title="Heat"),
                opacity=0.8
            ),
            hoverinfo='text',
            text=[f"Node {i}: Heat={hv[i]:.2f}" for i in range(len(G.nodes))]
        )
        frames.append(go.Frame(data=[edge_trace, node_trace], name=str(t)))

    # Create figure with initial data and animation settings
    fig = go.Figure(
        data=[edge_trace, frames[0].data[1]],  # Start with t=0
        layout=go.Layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            ),
            title="3D Heat Distribution in Graph Over Time in 0.1s (" + type +")",
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])
                        ])
            ]
        ),
        frames=frames  # Add all frames for animation
    )

    # Add slider for manual time control
    fig.update_layout(
        sliders=[{
            "steps": [{"args": [[str(t)], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
                    "label": f"t={t}", "method": "animate"} for t in range(len(heat_values_list))],
            "transition": {"duration": 300},
            "x": 0.1, "len": 0.9
        }]
    )

    fig.show()

    # # END

    # G = to_networkx(data, to_undirected=True)

    # # Step 3: Use a 3D layout for NetworkX to get 3D positions for nodes
    # pos = nx.spring_layout(G, dim=3)  # 'spring_layout' often works well for 3D graphs

    # # Extract node positions for plotting
    # node_x = [pos[i][0] for i in G.nodes()]
    # node_y = [pos[i][1] for i in G.nodes()]
    # node_z = [pos[i][2] for i in G.nodes()]

    # # Step 4: Define Plotly traces for nodes and edges

    # node_trace = go.Scatter3d(
    #     x=node_x, y=node_y, z=node_z,
    #     mode='markers',
    #     marker=dict(
    #         size=5,
    #         color=heat_values,  # Koloryzacja na podstawie wartości ciepła
    #         colorscale='YlOrRd',  # Skala kolorów: od żółtego (zimny) do czerwonego (gorący)
    #         colorbar=dict(title="Heat"),  # Dodanie paska kolorów
    #         opacity=0.8
    #     ),
    #     hoverinfo='text',
    #     text=[f"Node {i}: Heat={heat_values[i]:.2f}" for i in range(len(G.nodes))]  # Tekst hover
    # )

    # # Edge trace (bez zmian)
    # edge_x = []
    # edge_y = []
    # edge_z = []
    # for edge in G.edges():
    #     x0, y0, z0 = pos[edge[0]]
    #     x1, y1, z1 = pos[edge[1]]
    #     edge_x += [x0, x1, None]
    #     edge_y += [y0, y1, None]
    #     edge_z += [z0, z1, None]

    # edge_trace = go.Scatter3d(
    #     x=edge_x, y=edge_y, z=edge_z,
    #     mode='lines',
    #     line=dict(color='black', width=1),
    #     hoverinfo='none'
    # )

    # # Step 5: Plot
    # fig = go.Figure(data=[edge_trace, node_trace])
    # fig.update_layout(
    #     showlegend=False,
    #     scene=dict(
    #         xaxis=dict(showbackground=False),
    #         yaxis=dict(showbackground=False),
    #         zaxis=dict(showbackground=False)
    #     ),
    #     title="3D Heat Distribution in Graph"
    # )
    # fig.show()

def analyze_graph(data, T):
    if T < 0.:
        print("Bad T!")
        return
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
    # show_and_save_graph(data, u, 'open scheme') #
    u = u.reshape(-1, 1)
    dt = 0.001
    # Schemat jawny Eulera
    # u(t+dt) = u(t) + dt Lu(t) <==>
    # u(t+dt) = (I + dt L) u(t)
    M = np.identity(n)+dt*L
    t = dt
    jawny = [u.reshape(1, -1)]
    it = 0
    while t <= T:
        it += 1
        t += dt
        u = M @ u
        if it % 100 == 0:
            jawny.append(u.reshape(1, -1))
    print("Schemat jawny: u(" + str(T) + "s) = ", u.reshape(1, -1))
    # print(u)
    # print(mu)
    print("Sumaryczne ciepło (schemat jawny):", np.dot(u.reshape(-1), mu))
    print("Should be:", mu[0])
    show_and_save_graph(data, jawny, 'open scheme')

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
    oc.push('num_steps', int(T/dt))
    oc.eval("""
        [L, U, P] = lu(M);           % Rozkład LU macierzy M
        results = cell(floor(num_steps / 100)+1, 1);
        results{1} = u';
        result_index = 2;
        for step = 1:num_steps
            u = U \\ (L \\ (P * u)); % Rozwiąż M * u_nowe = u_stare
            if mod(step, 100) == 0
                results{result_index} = u';
                result_index = result_index+1;
            end
        end
    """)
    zamkniety = oc.pull('results')
    # print(zamkniety[-1])
    print("Schemat zamknięty: u(" + str(T) + "s) = ", zamkniety[-1][0].flatten())
    print("Sumaryczne ciepło (schemat zamknięty):", np.dot(zamkniety[-1][0].reshape(-1), mu))
    print("Should be:", mu[0])
    show_and_save_graph(data, zamkniety, 'closed scheme')
    return

# MAIN
path = './datasets'
database_name = input("Name of database:")
T = int(input("T:"))
minNrOfVertices = int(input("Minimal number of vertices:"))

# dataset = TUDataset(root=path, name='IMDB-BINARY')
dataset = TUDataset(root=path, name=database_name)

# We can see that graphs in dataset have not enough nodes.
# I will check better candidate later: FIRSTMM_DB

# Information about dataset
print(f"Liczba grafów: {len(dataset)}")
print(f"Pierwszy graf:\n{dataset[0]}")



for i, data in enumerate(dataset):
    if data.num_nodes >= minNrOfVertices:
        print(f"Graf {i}:")
        analyze_graph(data, T)

