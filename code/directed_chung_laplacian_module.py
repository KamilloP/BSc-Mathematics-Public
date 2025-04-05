import numpy as np

def make_out_degree_positive(E):
    assert E.shape[0] == E.shape[1]
    V = E.shape[0]
    dplus = np.sum(E, axis=1)
    d_out_0 = np.where(dplus == 0)
    Ec = np.copy(E)
    Ec[d_out_0, d_out_0] = 1
    return Ec

def compound_iteration(A, tol=1e-9, max_iter=1000):
    n, _ = A.shape
    v = np.ones((n, 1))
    v = v + A @ v
    v = v / np.linalg.norm(v)
    
    for i in range(max_iter):
        v_next = A @ v
        # v_next = np.dot(A, v)
        v_next = v_next / np.linalg.norm(v_next)
        if max_iter - i < 5:
            print(i, v, v_next)
        
        if np.linalg.norm(v_next - v) < tol:
            break
        v = v_next
    
    lambda_wlasna = v.T @ (A @ v)
    v = v / np.sum(v)
    return lambda_wlasna, v

def perron_vector(P, debug=False):
    """
    We want to find Perron vector phi such that it has all entries positive and
    it is left eigenvector of P, phi^T P = r phi^T, where r is scalar. We know
    from theory that r is 1. Perron vector is simple if and only if P is irreducible
    and aperiodic. We also know r has the biggest absolute value (r=1).
    Unfortunately it is possible, that there are more than one eigenvector with 
    eigenvalue 1, or -1 is eigenvalue. That may be big problem. It seems to be the case
    for cycle with 6 nodes. So we need method which works for not symmetric matrices.
    One known method is QR method, Ak = QkRk, A{k+1} = RkQk. It is implemented in
    np.linalg.eig method, which uses numerical package LAPACK. Source:
    https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html
    https://www.netlib.org/lapack/lug/node70.html
    """
    values, vectors = np.linalg.eig(P.T)
    valuesR = values.real
    val = 0.
    vec = np.zeros(P.shape[0])
    for vr, vi, eigvec in zip(values.real, values.imag, vectors.T):
        if np.isclose(vr, 1.) and np.isclose(vi, 0.):
            print(f"Found eigenvector with eigenvalue 1: {eigvec}!!!")
            val = 1.
            vec = np.abs(eigvec)
    if debug:
        print("Values:", values)
        print("Vectors:")
        print(vectors.T)
    return compound_iteration(P.T)
    # return val, vec - sometimes there are problems...

def chung_laplacian(P):
    n = P.shape[0]
    lambda_, phi = perron_vector(P)
    print("Compound iteration phi:", phi)
    print("Compound iteration lambda:", lambda_)
    srp = np.sqrt(phi)
    invsrp = 1/srp
    return np.eye(n)-((srp.reshape(-1, 1)*P*invsrp.reshape(1,-1) + invsrp.reshape(-1,1)*P.T*srp.reshape(1,-1))/2)

def combinatorial_chung_laplacian(P):
    _, phi = perron_vector(P)
    return np.diag(phi)-((phi.reshape(-1,1)*P + P.T*phi.reshape(1,-1))/2)

def euler_explicit_chung_step(L, x, dt):
    dx = -(L@x)*dt
    return x+dx

def euler_explicit_chung(Ed, x, dt, T, gap, laplacian=chung_laplacian, scale=False):
    """
    We assume that in Ed graph we have for each
    node u: d_u^{out} > 0.
    """
    x_values = [x]
    t = 0
    s = 0
    dplus = np.sum(Ed, axis=1)
    P = Ed / dplus.reshape(-1, 1) # Here we use the fact, that out degree is positive.
    print("P:")
    print(P)
    L = laplacian(P)
    while t < T:
        x = euler_explicit_chung_step(L, x, dt)
        t += dt
        s += 1
        if s%gap == 0:
            x_values.append(x)
    if scale:
        x_val
        x_values = list(map(lambda x: x/np.sqrt(dplus), x_values))
    return x_values

def euler_explicit_chung_scaled(Ed, x, dt, T, gap):
    return euler_explicit_chung(Ed, x, dt, T, gap, scale=True)

def euler_explicit_chung_combinatorial(Ed, x, dt, T, gap):
    return euler_explicit_chung(Ed, x, dt, T, gap, laplacian=combinatorial_chung_laplacian)

def euler_explicit_chung_combinatorial_scaled(Ed, x, dt, T, gap):
    return euler_explicit_chung(Ed, x, dt, T, gap, laplacian=combinatorial_chung_laplacian, scale=True)