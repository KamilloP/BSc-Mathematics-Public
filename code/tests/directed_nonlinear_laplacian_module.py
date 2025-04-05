import numpy as np

############################################################
################## DIFFERIENTIAL SCHEMES ###################
############################################################
def euler_explicit(Ed, x, dt, T, gap):
    x_values = [x]
    t = 0
    s = 0
    while t < T:
        x = euler_explicit_step(Ed, x, dt)
        t += dt
        s += 1
        if s%gap == 0:
            x_values.append(x)
    return x_values

def euler_explicit_step(Ed, x, dt):
    # Equation: dx = -L(x)dt
    # We use normalized laplacian
    L = normalized_laplacian_for_directed_graph(Ed, x)
    dx = -(L@x)*dt # was in an article, but values does not make sens then
    # dx = (L@x)*dt
    return x+dx

############################################################
############### LAPLACIAN FOR DIRECTED GRAPH ###############
############################################################
def normalized_laplacian_for_directed_graph(Ed, x):
    E = undirected_graph_for_nonlinear_laplacian(Ed, x)
    return normalized_laplacian_for_undirected_graph(E)


############################################################
############## LAPLACIAN FOR UNDIRECTED GRAPH ##############
############################################################
def normalized_laplacian_for_undirected_graph(A):
    assert A.shape[0] == A.shape[1]
    V = A.shape[0]
    d = np.sum(A, axis=1)
    sqd = np.sqrt(d).reshape(1,-1)
    invSqd = 1/sqd
    return np.eye(V)-invSqd.T*A*invSqd

############################################################
##################### UNDIRECTED GRAPH ##################### - tested
############################################################
# Ed - directed graph (such that d>0) as adges, x - current vector of vertices values
def undirected_graph_for_nonlinear_laplacian(Ed, x):
    assert Ed.shape[0] == Ed.shape[1]
    assert x.shape[0] == Ed.shape[0]
    assert x.shape[1] == 1
    assert x.shape[0] > 0 
    # Code
    V = x.shape[0]
    E = np.zeros((V, V))
    dplus = np.sum(Ed, axis=1)
    dminus = np.sum(Ed, axis=0)
    d = dplus+dminus
    assert len(d.shape) == 1
    # print("TEST: x and d shape:", x.shape, d.shape)
    if np.min(d) == 0:
        print(d)
    undirected_edge_index = np.where((x/np.sqrt(d.reshape(-1,1)))-(x.T/np.sqrt(d.reshape(1,-1))) >= 0)
    def add_undirected_edge(Ed, index):
        E = np.copy(Ed)
        B = np.zeros(E.shape)
        B[index] = 1.
        E *= B
        return E+E.T
    # print(undirected_edge_index)
    # print(add_undirected_edge(Ed, undirected_edge_index))
    E += add_undirected_edge(Ed, undirected_edge_index) # Here I added edges (u,v) if xu/sqrt(du) > xv/sqrt(dv)
    def add_loops_for_edge(Ed, opposite_index):
        E = np.copy(Ed)
        B = np.ones(E.shape)
        B[opposite_index] = 0
        E *= B
        diagonal = np.sum(E, axis=0)+np.sum(E, axis=1)
        return np.diag(diagonal)
    # print(add_loops_for_edge(Ed, undirected_edge_index))
    # diagonal = np.sum(E, axis=0)+np.sum(E, axis=1)
    # return np.diag(diagonal)
    E += add_loops_for_edge(Ed, undirected_edge_index) # Adding loops.
    return E

############################################################
###################### DIRECTED GRAPH ###################### - tested
############################################################
def get_nr_min_max_vertex_until(file_name, time=14020050):
    id_min = 1000000000
    id_max = -1
    with open(file_name, 'r') as file:
        for line in file:
            numbers = line.split()
            src = int(numbers[0])
            dst = int(numbers[1])
            id_min = min(id_min, src, dst)
            id_max = max(id_max, src, dst)
            ts = int(numbers[2])
            if ts > time:
                break
    return id_min, id_max

def get_graph(file_name, time=14020050):
    id_min, id_max = get_nr_min_max_vertex_until(file_name, time)
    V = id_max+1
    A = np.zeros((V, V))
    with open(file_name, 'r') as file:
        for line in file:
            numbers = line.split()
            src = int(numbers[0])
            dst = int(numbers[1])
            ts = int(numbers[2])
            if ts <= time:
                A[src, dst] = 1-A[src, dst]
            else:
                break
    print("Min:", id_min)
    print("Max:", id_max)
    return A

# Returns copy of graph, but with added loops for isolated vertices
def add_loops_for_isolated_vertices(E):
    assert E.shape[0] == E.shape[1]
    V = E.shape[0]
    dplus = np.sum(E, axis=1)
    dminus = np.sum(E, axis=0)
    d = dplus+dminus
    isolated = np.where(d == 0)
    Ec = np.copy(E)
    Ec[isolated, isolated] = 1
    return Ec