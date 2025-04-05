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
        if t > 10.0005 and t < 10.0015:
            print(f"Time: {t}")
            print("x:", x)
            print("L:")
            print(normalized_laplacian_for_directed_graph(Ed, x))
            print("Undirected graph:")
            print(undirected_graph_for_nonlinear_laplacian(Ed, x))

    return x_values

def euler_explicit_step(Ed, x, dt):
    # Equation: dx = -L(x)dt
    # We use normalized laplacian
    L = normalized_laplacian_for_directed_graph(Ed, x)
    # dx = -(L@x)*dt was in an article, but values does not make sens then
    dx = -(L@x)*dt
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
##################### UNDIRECTED GRAPH #####################
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
    E += add_undirected_edge(Ed, undirected_edge_index) # Here I added edges (u,v) if xu/sqrt(du) > xv/sqrt(dv)
    def add_loops_for_edge(Ed, opposite_index):
        E = np.copy(Ed)
        B = np.ones(E.shape)
        B[opposite_index] = 0
        E *= B
        diagonal = np.sum(E, axis=0)+np.sum(E, axis=1)
        return np.diag(diagonal)
    E += add_loops_for_edge(Ed, undirected_edge_index) # Adding loops.
    return E

############################################################
###################### DIRECTED GRAPH ######################
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

############################################################
###################### Projection GRAPH ######################
############################################################

def orthogonal_projection(v, mu):
    assert np.max(np.abs(mu)) > 0.000001 # Our vector mu is not 0
    return v - mu*(np.dot(mu.flatten(), v.flatten())/np.dot(mu.flatten(), mu.flatten()))

############################################################
###################### Utility ######################
############################################################
def mu_vector(E):
    dplus = np.sum(E, axis=1)
    dminus = np.sum(E, axis=0)
    d = dplus+dminus
    return np.sqrt(d).reshape(-1, 1)

def vol(S, E):
    dplus = np.sum(E, axis=1)
    dminus = np.sum(E, axis=0)
    d = (dplus+dminus).reshape(-1,1)
    return np.sum(S*d)

def supp(x, atol=1e-8):
    assert len(x.shape) == 2
    assert x.shape[1] == 1
    return (~np.isclose(x, 0, atol=atol)).astype(int)

def positive_part(x):
    return np.where(x > 0, x, 0)

def negative_part(x):
    return np.where(x < 0, x, 0)

def min_positive_coordinate(x):
    positive_values = x[x > 0]  # Filter out non-positive values
    return np.min(positive_values) if positive_values.size > 0 else 0  # Return None if no positive values

def max_negative_coordinate(x):
    return -min_positive_coordinate(-x)

# It may not stop!!!
def find_c(x_, E):
    n = E.shape[0]
    half_vol = vol(np.ones((n,1)), E) / 2.
    c = 0.
    y_ = x_ + c*np.ones_like(x_)
    while vol(supp(positive_part(y_)), E) > half_vol or vol(supp(negative_part(y_)), E) > half_vol:
        if vol(supp(positive_part(y_)), E) > half_vol:
            dc = min_positive_coordinate(y_)
            c -= dc
        else:
            dc = max_negative_coordinate(y_)
            c -= dc
        y_ = x_ + c*np.ones_like(x_)
    return c

def compute_auxiliary_quotient(x, E):
    assert len(x.shape) == 2
    assert x.shape[1] == 1
    dplus = np.sum(E, axis=1)
    dminus = np.sum(E, axis=0)
    d = dplus+dminus
    numerator = np.sum(positive_part(x-x.T)**2)
    denominator = np.dot(d, (x*x).flatten())
    return numerator / denominator

def sort_vertices(x):
    n = x.shape[0]
    vertices = np.arange(n).reshape(-1, 1)  # Indeksy wierzchołków
    data = np.hstack((x, vertices))  # Łączenie wartości x z indeksami

    # Sortowanie malejące względem pierwszej kolumny (wartości x_)
    sorted_data = data[np.argsort(-data[:, 0])]  

    return sorted_data  # Zwracamy posortowane wiersze

def conductance(S, E):
    S = S.reshape(-1, 1)
    # edges_S_to_rest = (E*S*(1-S).T).astype(int)
    cut_plus = np.sum((E*S*(1-S).T).astype(int))
    cut_minus = np.sum((E*(1-S)*S.T).astype(int))
    vol_S = vol(S, E)
    vol_VnoS = vol(1-S, E)
    # print("E", E)
    # print("Edges between S and V/S:", edges_between)
    # print("cut_plus and cut_minus:", cut_plus, cut_minus)
    # print("vol(S) and vol(V/S):", vol_S, vol_VnoS)
    return min(cut_plus, cut_minus).astype(float) / min(vol_S, vol_VnoS).astype(float)

############################################################
###################### Conductance approximation ######################
############################################################

def lemma_4_12_positive(x, E):
    # As mentioned at the end:
    n = x.shape[0]
    sorted_vertices = sort_vertices(x)
    # print("sorted vertices:", sorted_vertices) # We can see it has floats.
    S = np.zeros((n, 1))
    best_conductance = 1000000000. # Infinity should be here...
    best_S = S.copy()
    for i in range(n-1):
        # We cannot have S = V! This is why i < n-1.
        S[sorted_vertices[i,1].astype(int), 0] = 1
        new_conductance = conductance(S, E)
        if new_conductance < best_conductance:
            best_S = S.copy()
            best_conductance = new_conductance
    return best_S, best_conductance

def lemma_4_13_negative(x, E):
    # In lemma 4.13: "Similarly, we can show the following." lemma 4.12.
    # I am not sure how to interpret it. It seems that it should work the same as lemma 4.12, but with multiplication by negative.
    # TODO
    return lemma_4_12_positive(-x, E)

def find_S_approximation(x, E):
    assert len(x.shape) == 2
    assert x.shape[1] == 1
    # Assumption: <x, mu> = 0
    mu = mu_vector(E)
    assert np.isclose(x.T @ mu, 0)
    dplus = np.sum(E, axis=1)
    dminus = np.sum(E, axis=0)
    d = dplus+dminus
    assert np.all(~np.isclose(d, 0)) # We do not want here d_u near zero, because we want to divide by it. 
    # However it is not necessary for algorithm, just computation is faster.
    n = E.shape[0]
    x_ = x / np.sqrt(d).reshape(-1,1)
    c = find_c(x_, E)
    y_ = x_ + c*np.ones_like(x_)
    if compute_auxiliary_quotient(positive_part(y_), E) < compute_auxiliary_quotient(negative_part(y_), E):
        # y_* = y_+
        # delta = 1.
        v = positive_part(y_)**2
        v /= np.sqrt(np.dot(v.flatten(),v.flatten()))
        return lemma_4_12_positive(v, E)
    # y_* = y_-
    # delta = -1.
    v = -(negative_part(y_)**2)
    v /= np.sqrt(np.dot(v.flatten(),v.flatten()))
    return lemma_4_13_negative(-(negative_part(y_)**2), E)


def finding_conductance_approximation(Ed, x, dt, T, gap):
    assert len(x.shape) == 2
    assert x.shape[1] == 1
    mu = mu_vector(Ed)
    x = orthogonal_projection(x, mu)
    assert np.isclose(x.T @ mu, 0)
    # assert np.isclose(x.T*x.T, 1)
    x_values = [x.copy()]
    t = 0
    s = 0
    while t < T:
        L = normalized_laplacian_for_directed_graph(Ed, x)
        dx = -(orthogonal_projection(L@x, mu))*dt
        x += dx
        t += dt
        s += 1
        if s%gap == 0:
            x_values.append(x.copy())
        if t > 10.0005 and t < 10.0015:
            print(f"Time: {t}")
            # print("x:", x)
            # print("L:")
            # print(normalized_laplacian_for_directed_graph(Ed, x))
            # print("Undirected graph:")
            # print(undirected_graph_for_nonlinear_laplacian(Ed, x))
    S, conductanceS = find_S_approximation(x, Ed) # TODO
    return x_values, S, conductanceS

def Rayleigh_quotient(E, x):
    # Defined for x =/= 0
    # We want degree (d) to be positive for all coordinate - just for easier computation.
    assert len(E.shape) == 2
    assert E.shape[0] == E.shape[1]
    # assert np.any(~np.isclose(x, 0)), str((np.dot(x.flatten(), x.flatten())))+str(x.flatten())# Check that it is not zero.
    n = E.shape[0]
    x = x.reshape(-1,1)
    dplus = E.sum(axis=1)
    dminus = E.sum(axis=0)
    d = (dplus+dminus).astype(float).reshape(-1,1)
    xnormed = x/np.sqrt(d) # Should be shape: (-1,1)
    tail = xnormed*E.astype(float)
    head = xnormed.T*E.astype(float)
    return np.sum(positive_part(tail-head)**2)/(np.dot(x.flatten(), x.flatten()))

def find_best_conductance_brute_force(E):
    assert len(E.shape) == 2
    assert E.shape[0] == E.shape[1]
    n = E.shape[0]
    assert n < 23, "We do not want to use this algorithm for too big graph!" + str(n)
    bestS = None
    bestConductance = 10000000000000. # Should be infinity...
    for i in range(1,2**n):
        t = i
        S = np.zeros((n,1))
        for j in range(n):
            if t % 2 == 1:
                S[j,0] = 1
            t //= 2
        conductanceS = conductance(S, E)
        if conductanceS < bestConductance:
            bestConductance = conductanceS
            bestS = S
    return bestS, bestConductance
        

