import numpy as np
import directed_nonlinear_laplacian_module as nlin


############################################################
###################### DIRECTED GRAPH ######################
############################################################
def simple_test_getting_directed_graphs():
    print("SIMPLE TEST 1 - get_graph:")
    Ed = nlin.get_graph("simple_line.txt")
    Edexp = np.zeros((6, 6))
    edges = [
        [0,1],
        [1,2],
        [2,3],
        [3,4],
        [4,5]
    ]
    for e in edges:
        Edexp[e[0], e[1]] = 1
    assert np.array_equal(Ed, Edexp)
    Ed = nlin.get_graph("simple_cycle.txt")
    Edexp = np.zeros((6, 6))
    edges = [
        [1,2],
        [2,3],
        [3,4],
        [4,5],
        [5,1]
    ]
    for e in edges:
        Edexp[e[0], e[1]] = 1
    assert np.array_equal(Ed, Edexp)
    print("Success!")

def simple_test_get_min_max():
    print("SIMPLE TEST 2 - get_nr_min_max_vertex_until")
    minn, maxx = nlin.get_nr_min_max_vertex_until("simple_line.txt")
    assert minn == 0, "line min failed"
    assert maxx == 5, "line max failed"
    minn, maxx = nlin.get_nr_min_max_vertex_until("simple_cycle.txt")
    assert minn == 1, "cycle min failed"
    assert maxx == 5, "cycle max failed"
    print("Success!")

def simple_test_add_loops_for_isolated_vertices():
    Ed = nlin.get_graph("simple_line.txt")
    print("SIMPLE TEST 3 - add_loops_for_isolated_vertices:")
    Edexp = nlin.add_loops_for_isolated_vertices(Ed.copy())
    assert np.array_equal(Ed, Edexp), "line graph should not be changed"
    E = nlin.get_graph("simple_isolated.txt")
    Ed = nlin.add_loops_for_isolated_vertices(E.copy())
    Edexp = np.zeros((5,5))
    edges = [
        [0,1],
        [2,2],
        [3,4]
    ]
    for e in edges:
        Edexp[e[0], e[1]] = 1
    # print(f"Edexp: {Edexp}, Ed: {Ed}")
    assert np.array_equal(Ed, Edexp), "isolated graph should be changed"
    print("Success!")

simple_test_getting_directed_graphs()
simple_test_get_min_max()
simple_test_add_loops_for_isolated_vertices()

############################################################
##################### UNDIRECTED GRAPH #####################
############################################################
def simple_test_undirected_graph_for_nonlinear_laplacian():
    print("SIMPLE TEST 4 - undirected_graph_for_nonlinear_laplacian:")
    # We test with graph given in an article.
    # v1 = index 0,
    # v2 = index 1, etc.
    Ed = nlin.get_graph("article_graph.txt")
    x = np.array([1., 0.8, 0.4, 0.6]).reshape(-1, 1)
    E = nlin.undirected_graph_for_nonlinear_laplacian(Ed, x)
    # print("x:", x)
    # print("Ed:", Ed)
    dplus = np.sum(Ed, axis=1)
    dminus = np.sum(Ed, axis=0)
    d = dplus+dminus
    # print("d:", d)
    # print("E:", E)
    assert np.array_equal(E, np.array([[0., 1., 0., 1.],[1., 0., 1., 0.],[0., 1., 1., 0.],[1., 0., 0., 1.]]))
    print("Success!")


simple_test_undirected_graph_for_nonlinear_laplacian()

# # TEST getting directed graphs
# def test_getting_directed_graphs():
#     print("TEST 1: get_graph:")
#     Ed = nlin.get_graph("simple_directed_graph.txt")
#     Edexp = np.zeros((17, 17))
#     edges = [
#         [1,1],
#         [2,1],
#         [3,4],
#         [4,5],
#         [5,3],
#         [4,6],
#         [7,7],
#         [8,9],
#         [9,10],
#         [10,11],
#         [11,8],
#         [10,12],
#         [12,13],
#         [13,14],
#         [14,15],
#         [15,16],
#         [16,12],
#         [12,15]
#     ]
#     for e in edges:
#         Edexp[e[0], e[1]] = 1
#     assert Ed == Edexp
#     print("TEST 2: get_graph:")