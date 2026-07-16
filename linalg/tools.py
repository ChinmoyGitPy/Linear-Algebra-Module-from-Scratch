
import math
import random
from .vector import Vector
from .matrix import Matrix

def _complete_orthonormal_basis(vectors, dimension):
    basis = list(vectors)
    while len(basis) < dimension:
        random_vec = Vector([random.uniform(-1, 1) for _ in range(dimension)])
        for b in basis:
            projection = random_vec.dot(b)
            random_vec = random_vec - (b * projection)
        norm = random_vec.norm()
        if norm > 1e-12:
            basis.append(random_vec * (1.0 / norm))
        else:
            continue
    return basis

def svd(A):
    m, n = A.shape
    if m >= n:
        M = A.transpose @ A
        eigenvalues, eigenvectors = M.eig()
        sigma = [math.sqrt(ev) if ev > 1e-12 else 0.0 for ev in eigenvalues]
        Vt = Matrix(eigenvectors)
        u_columns = []
        for i in range(n):
            v_i = eigenvectors[i]
            Av_vec = Vector([A.rows[r].dot(v_i) for r in range(m)])
            u_i = Av_vec * (1.0 / sigma[i]) if sigma[i] > 1e-12 else Vector([0.0] * m)
            u_columns.append(u_i)
        r = sum(1 for s in sigma if s > 1e-12)
        good_u = u_columns[:r]
        full_u = _complete_orthonormal_basis(good_u, m)
        U = Matrix(full_u).transpose
        return U, sigma, Vt
    else:
        M = A @ A.transpose
        eigenvalues, eigenvectors = M.eig()
        sigma = [math.sqrt(ev) if ev > 1e-12 else 0.0 for ev in eigenvalues]
        U = Matrix(eigenvectors).transpose
        v_columns = []
        for i in range(m):
            u_i = eigenvectors[i]
            ATu_vec = Vector([A.transpose.rows[r].dot(u_i) for r in range(n)])
            v_i = ATu_vec * (1.0 / sigma[i]) if sigma[i] > 1e-12 else Vector([0.0] * n)
            v_columns.append(v_i)
        r = sum(1 for s in sigma if s > 1e-12)
        good_v = v_columns[:r]
        full_v = _complete_orthonormal_basis(good_v, n)
        Vt = Matrix(full_v)
        return U, sigma, Vt

def pinv(A):
    U, sigma, Vt = svd(A)
    m, n = A.shape
    s_inv = []
    r = len(sigma)
    for i in range(n):
        row = []
        for j in range(m):
            if i == j and i < r and sigma[i] > 1e-12:
                row.append(1.0 / sigma[i])
            else:
                row.append(0.0)
        s_inv.append(Vector(row))
    s_inv_mat = Matrix(s_inv)
    V = Vt.transpose
    return V @ s_inv_mat @ U.transpose

def pca(X, n_components=None):
    rows = [list(row) for row in X]
    n_samples, n_features = X.shape
    means = []
    for j in range(n_features):
        col_vals = [row[j] for row in rows]
        mean = sum(col_vals) / n_samples
        means.append(mean)
    centered_rows = []
    for row in rows:
        centered_rows.append([val - means[j] for j, val in enumerate(row)])
    X_c = Matrix(centered_rows)
    U, sigma, Vt = svd(X_c)
    total_var = sum(s**2 for s in sigma)
    if n_components is None:
        n_components = len(sigma)
    top_sigma = sigma[:n_components]
    explained_variance = [(s**2) / total_var if total_var > 0 else 0.0 for s in top_sigma]
    components = Matrix([Vt[i] for i in range(n_components)])
    V = Vt.transpose
    V_reduced = Matrix([V.get_column(i) for i in range(n_components)]).transpose
    transformed_matrix = X_c @ V_reduced
    return components, explained_variance, transformed_matrix
