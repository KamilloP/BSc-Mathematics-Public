import show_and_analyze_directed_module as sad
import numpy as np
import directed_nonlinear_laplacian_module as nlin
import directed_chung_laplacian_module as chung
import networkx as nx
import plotly.graph_objects as go
# email-Eu-core-temporal-Dept1.txt

############################################################
########################### MAIN ###########################
############################################################

option = int(input("Choose option:\n1. Debug\n2. Analyze nonlinear Laplacian\n3. Analyze Chung Laplacian\n4. Conductance estimation by nonlinear Yoshida Laplacian\n5. Create and save graph\n6. Optimal conductance (brute force)\n"))

if option != 5:
    path = './datasets'
    file_name = input("Name of file:")
    T = int(input("T:"))
    # debug = input("Debug?")
    dt = 0.001
    # minNrOfVertices = int(input("Minimal number of vertices:"))



    # dataset = TUDataset(root=path, name='IMDB-BINARY')
    # dataset = TUDataset(root=path, name=database_name)

    # We can see that graphs in dataset have not enough nodes.
    # I will check better candidate later: FIRSTMM_DB

    if option == 1:
        w = int(input("Choose where heat will be at first:"))
        minn, maxx = nlin.get_nr_min_max_vertex_until(path+"/"+file_name, time=14020050)
        x = np.zeros(maxx+1)
        x[w] = 1.
        E = nlin.get_graph(path+"/"+file_name)
        print("E:")
        print(E)
        Eout = chung.make_out_degree_positive(E)
        print("Eout:")
        print(Eout)
        P = Eout / np.sum(Eout, axis=1).reshape(-1, 1) 
        print("P:")
        print(P)
        val, vec = chung.perron_vector(P, debug=True)
        # Look at aperiodic.txt, it's eigenvalues and eigenvectors.
        # aperiodic_vec = np.array([0.47142211, 0.41545086, 0.36612498, 0.3226555, 0.28434708, 0.53493405])
        # aperiodic_val = 1.13472414
        # print("Aperiodic, vec.T @ E:")
        # print(aperiodic_vec.reshape(1, -1) @ E)
        # print("Aperiodic, vec.T * val:")
        # print(aperiodic_vec.reshape(1, -1)*aperiodic_val)
        print(f"Perron eigenvalue = {val}")
        print(f"Perron eigenvector = {vec}")
        print("Testing, vec @ P:", vec.reshape(1, -1) @ P, "; vec * val:", val * vec.reshape(1, -1))
        L = chung.chung_laplacian(P)
        print("L:")
        print(L)

        # print(f"E: {E}")
        # print(f"minn: {maxx}, maxx: {minn}")
        print(x)
        # BEGIN PROPER
        # analyze_graph(
        #     nlin.add_loops_for_isolated_vertices(nlin.get_graph(path+"/"+file_name)), 
        #     T, dt,
        #     nlin.euler_explicit,
        #     'Implicit Euler scheme for nonlinear normalized laplacian',
        #     x=x.reshape(-1, 1)
        # )
        # analyze_graph(
        #     chung.make_out_degree_positive(nlin.get_graph(path+"/"+file_name)),
        #     T, dt,
        #     chung.euler_explicit_chung,
        #     'Implicit Euler scheme for Chung laplacian',
        #     x=x.reshape(-1, 1)
        # )
        # analyze_graph(
        #     chung.make_out_degree_positive(nlin.get_graph(path+"/"+file_name)),
        #     T, dt,
        #     chung.euler_explicit_chung_combinatorial,
        #     'Implicit Euler scheme for combinatorial Chung laplacian',
        #     x=x.reshape(-1, 1)
        # )
        # END PROPER

        # analyze_graph(
        #     chung.make_out_degree_positive(nlin.get_graph(path+"/"+file_name)),
        #     T, dt,
        #     chung.euler_explicit_chung_combinatorial_scaled,
        #     'Implicit Euler scheme for combinatorial Chung laplacian (scaled)',
        #     x=x.reshape(-1, 1)
        # )
        # analyze_graph(
        #     chung.make_out_degree_positive(nlin.get_graph(path+"/"+file_name)),
        #     T, dt,
        #     chung.euler_explicit_chung_scaled,
        #     'Implicit Euler scheme for Chung laplacian (scaled)',
        #     x=x.reshape(-1, 1)
        # )
        print("Conductance!")
        E = nlin.add_loops_for_isolated_vertices(nlin.get_graph(path+"/"+file_name))
        x = x.reshape(-1, 1)
        x_values, S, conductanceS = nlin.finding_conductance_approximation(E, x, dt, T, 100)
        print("Last temperatures:", x_values[-1])
        print("S:", S)
        print("Conductance of S:", conductanceS)
        x_values.append(S)
        sad.show_and_save_graph_directed(E, x_values, "Conductance!")
        # kliki:
        two_cliques = sad.create_2_connected_cliques()
        x = np.zeros(200)
        x[w] = 1.
        x = x.reshape(-1, 1)
        x_values, S, conductanceS = nlin.finding_conductance_approximation(two_cliques, x, dt, T, 100)
        print("Last temperatures:", x_values[-1])
        print("S:", S)
        print("Conductance of S:", conductanceS)
        x_values.append(S)
        sad.show_and_save_graph_directed(two_cliques, x_values, "Conductance of 2 cliques!")
    elif option == 2:
        E = nlin.add_loops_for_isolated_vertices(nlin.get_graph(path+"/"+file_name))
        w = int(input("Where to set tempretature as 1?\n"))
        x = np.zeros(E.shape[0]).astype(np.float64)
        x[w] = 1.
        sad.analyze_graph(
            E, 
            T, dt,
            nlin.euler_explicit,
            f'Implicit Euler scheme for nonlinear normalized laplacian - start at node {w}',
            x=x.reshape(-1, 1)
        )
        x = np.random.uniform(0., 10., E.shape[0])
        sad.analyze_graph(
            E, 
            T, dt,
            nlin.euler_explicit,
            'Implicit Euler scheme for nonlinear normalized laplacian - random start',
            x=x.reshape(-1, 1)
        )
    elif option == 4:
        print("Conductance!")
        dt = float(input("dt:"))
        E = nlin.add_loops_for_isolated_vertices(nlin.get_graph(path+"/"+file_name))
        bestConductance = 1000000000000000. # Infinity should be here...
        S = np.zeros((E.shape[0], 1))
        bestS = S.copy()
        bestXvalues = None
        bestRayleigh = 0.
        for _ in range(10):
            x = np.random.uniform(-10., 10., E.shape[0])
            x = x.reshape(-1, 1)
            x_values, S, conductanceS = nlin.finding_conductance_approximation(E, x, dt, T, 100)
            rayleigh = nlin.Rayleigh_quotient(E, x_values[-1])
            print("Last temperatures:", x_values[-1].flatten())
            print("S:", S.reshape(-1))
            print("Conductance of S:", conductanceS)
            print("Rayleigh quotient:", rayleigh)
            print(conductanceS, "<=", 2*np.sqrt(rayleigh), ":", conductanceS <= 2*np.sqrt(rayleigh))
            x_values.append(S)
            if conductanceS < bestConductance:
                bestS = S.copy()
                bestConductance = conductanceS
                bestXvalues = x_values.copy()
                bestRayleigh = rayleigh
        sad.show_and_save_graph_directed(E, bestXvalues, f"Best conductance({bestConductance}) and R_G {bestRayleigh}!")
    else:
        # option == 6, Brute force conductance
        print("Brute force!")
        E = nlin.add_loops_for_isolated_vertices(nlin.get_graph(path+"/"+file_name))
        S, conductance = nlin.find_best_conductance_brute_force(E)
        sad.show_and_save_graph_directed(E, [S], f"Optimal conductance({conductance})!")
        print(S.flatten(), conductance)
elif option == 5:
    path = './datasets'
    file_name = input("Name of file:")
    file_path = path+"/"+file_name
    graph_type = int(input("Choose graph type:\n1. Two weakly connected cliques\n2. Two weakly connected random almost cliques\n3. Largest strongly connected component\n"))
    if graph_type == 1:
        sad.save_graph(sad.create_2_connected_cliques(), file_path)
    elif graph_type == 2:
        V1 = int(input("V1 number of vertices:"))
        V2 = int(input("V2 number of vertices:"))
        V1_to_V2 = int(input("Edges going from V1 to V2:"))
        V2_to_V1 = int(input("Edges going from V2 to V1:"))
        p1 = float(input("Probability of edge between two vertices from V1:"))
        p2 = float(input("Probability of edge between two vertices from V2:"))
        sad.save_graph(sad.create_random_almost_2_connected_cliques(V1, V2, V1_to_V2, V2_to_V1, p1, p2), file_path)
    elif graph_type == 3:
        file_name_graph = input("Name of file defining input graph:")
        graph_path = path+"/"+file_name_graph
        E = nlin.add_loops_for_isolated_vertices(nlin.get_graph(graph_path))
        newE, vertices = sad.largest_scc(E)
        print("Vertices in scc:", vertices)
        sad.save_graph(newE, file_path)
    else:
        print("Wrong graph option! Killing process!")


