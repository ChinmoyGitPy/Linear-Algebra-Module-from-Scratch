
import math
import random
from .vector import Vector

class Matrix:

    epsilon = 1e-12

    def __init__(self,rows):
        if not rows:
            raise ValueError("Matrix can't be empty")
        
        if isinstance(rows, Vector):
            rows = [rows]
            
        elif isinstance(rows, (list, tuple)) and all(isinstance(x, (int, float)) for x in rows):
            rows = [rows]

        processed_rows = []
        for r in rows:
            if isinstance(r, Vector):
                processed_rows.append(r)
            else:
                processed_rows.append(Vector(r))
                
        target_dimension = len(processed_rows[0])
        if not all(len(row) == target_dimension for row in processed_rows):
            raise ValueError("All rows in a matrix must have the same dimension.")
        self.rows = processed_rows
        self.num_rows = len(processed_rows)
        self.num_cols = target_dimension

    @classmethod
    def identity(cls,n):
        if isinstance(n,int) and n<= 0:
            raise ValueError("Identity matrix dimension must be a positive integer")
        grid = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        return cls(grid)
    
    @property
    def shape(self):
        return (self.num_rows, self.num_cols)
    def __repr__(self):
        row_strs = [f" {repr(row)}" for row in self.rows]
        return "Matrix([\n"+",\n".join(row_strs)+"\n])"
    
    def __len__(self):
        return self.num_rows
    
    def __getitem__(self, index):
        return self.rows[index]
    
    def get_column(self, col_index):
        if not (0 <= col_index < self.num_cols):
            raise IndexError("Column index out of range")
        return Vector([row[col_index] for row in self.rows])
    
    def augment(self,other):
        if not isinstance(other, Matrix):
            raise TypeError("Can only augment with another Matrix instance")
        if self.num_rows != other.num_rows:
            raise ValueError(f"Row mismatch for matrix augmentation: {self.num_rows} vs {other.num_rows}. "
                "Both matrices must have the exact same number of rows to append horizontally.")
        combined_rows = [list(r1) + list(r2) for r1,r2 in zip(self.rows, other.rows)]
        return Matrix(combined_rows)
    
    @property
    def transpose(self):
        return Matrix([self.get_column(j) for j in range(self.num_cols)])
    
    def __add__(self,other):
        if isinstance(other,(int,float)):
            return Matrix([row+other for row in self.rows])
        if isinstance(other,Matrix):
            if self.shape != other.shape:
                raise ValueError("Shape must be same")
            return Matrix([r1 + r2 for r1, r2 in zip(self.rows, other.rows)])
        return NotImplemented
    
    def  __radd__(self,other):
        return self.__add__(other)
    
    def __sub__(self,other):
        if isinstance(other,(int,float)):
            return Matrix([row-other for row in self.rows])
        if isinstance(other,Matrix):
            if self.shape != other.shape:
                raise ValueError("Shape must be same")
            return Matrix([r1 - r2 for r1, r2 in zip(self.rows, other.rows)])
        return NotImplemented
    
    def  __rsub__(self,other):
        if isinstance(other,(int,float)):
            return Matrix([other - row for row in self.rows])
        return NotImplemented
    
    def __mul__(self,other):
        if isinstance(other,(int,float)):
            return Matrix([row*other for row in self.rows])
        if isinstance(other,Matrix):
            if self.shape != other.shape:
                raise ValueError("Shape must be same")
            return Matrix([Vector([a * b for a, b in zip(r1, r2)]) for r1, r2 in zip(self.rows, other.rows)])
        return NotImplemented
    
    def  __rmul__(self,other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other,(int,float)):
            return Matrix([row/other for row in self.rows])
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Shape must be same")
            new_rows = []
            for r1,r2 in zip(self.rows,other.rows):
                row_data = []
                for a,b in zip(r1,r2):
                    if abs(b) < self.epsilon:
                        raise ZeroDivisionError("Can't divide element-wise by a matrix containing zero elements")
                    row_data.append(a/b)
                new_rows.append(Vector(row_data))
            return Matrix(new_rows)
        return NotImplemented
    
    def __rtruediv__(self, other):
        if isinstance(other,(int,float)):
            return Matrix([other/row for row in self.rows])
        return NotImplemented

    def __matmul__(self,other):
        if not isinstance(other,Matrix):
            return NotImplemented
        if self.num_cols != other.num_rows:
            raise ValueError(f"Dimension mismatch for matrix multiplication: {self.shape} @ {other.shape}. "
                            "Inner dimensions must match.")
        new_rows = []
        for i in range(self.num_rows):
            row_data = []
            for j in range(other.num_cols):
                col_vector = other.get_column(j)
                row_data.append(self.rows[i].dot(col_vector))
            new_rows.append(Vector(row_data))

        return Matrix(new_rows)
    
    def __eq__(self,other):
        if not isinstance(other,Matrix):
            return False
        if self.shape != other.shape:
            return False
        return all (r1 == r2 for r1,r2 in zip(self.rows,other.rows))
    
    @property
    def det(self):
        if self.num_rows != self.num_cols:
            raise ValueError("Determinant can only be calculated for square matrices")
        
        grid = [list(row) for row in self.rows]
        return self._recursive_det(grid)

    def _recursive_det(self,grid):
        n = len(grid)
        if n == 1:
            return grid[0][0]
        if n == 2:
            return grid[0][0]*grid[1][1]-grid[0][1]*grid[1][0]
        total_det = 0.0
        for c in range(n):
            sub_grid = [row[:c] + row[c+1:] for row in grid[1:]]
            sign = 1 if c % 2 == 0 else -1
            total_det += sign * grid[0][c] * self._recursive_det(sub_grid)

        return total_det
    
    def inverse(self):
        if self.num_rows != self.num_cols:
            raise ValueError
        n = self.num_rows
        I = Matrix.identity(n)
        augmented_matrix = self.augment(I)
        grid = [list(row) for row in augmented_matrix.rows]
        total_cols = augmented_matrix.num_cols

        for i in range(n):
            pivot_row = i
            for r in range(i+1,n):
                if abs(grid[r][i]) > abs(grid[pivot_row][i]):
                    pivot_row = r
            if pivot_row != i:
                grid[i], grid[pivot_row] = grid[pivot_row],grid[i]

            if abs(grid[i][i]) < self.epsilon:
                raise ValueError("Matrix is singular (determinant is 0) and cannot be inverted.")
            
            pivot_val = grid[i][i]
            grid[i] = [x/pivot_val for x in grid[i]]

            for r in range(n):
                if r != i:
                    factor = grid[r][i]
                    grid[r] = [grid[r][k] - factor*grid[i][k] for k in range(total_cols)]
        inverse_grid = [row[n:] for row in grid]
        return Matrix(inverse_grid)
    
    @property
    def I(self):
        return self.inverse()
    
    def qr(self):
        if self.num_rows != self.num_cols:
            raise ValueError("QR Decomposition is configured for square matrices.")
        n = self.num_cols
        m = self.num_rows

        columns = [self.get_column(j) for j in range(n)]
        q_columns = []

        for j in range(n):
            a_j = columns[j]
            u_j = a_j

            for q_k in q_columns:
                proj = a_j.dot(q_k)
                u_j = u_j - (q_k * proj)

            norm_u = u_j.norm()
            if norm_u < self.epsilon:
                random_vec = Vector([random.uniform(-1, 1) for _ in range(m)])
                for q_k in q_columns:
                    proj = random_vec.dot(q_k)
                    random_vec = random_vec - (q_k * proj)
                norm_random = random_vec.norm()
                if norm_random < self.epsilon:
                    raise ValueError("Failed to find an independent orthogonal vector.")
                q_columns.append(random_vec * (1.0 / norm_random))
            else:
                q_columns.append(u_j * (1.0 / norm_u))

        Q = Matrix(q_columns).transpose   
        R = Q.transpose @ self
        return Q, R

    def eig(self, max_iterations=2000, tolerance=None):
        if self.num_rows != self.num_cols:
            raise ValueError("Eigen-decomposition requires a square matrix")
        n = self.num_rows
        if tolerance is None:
            tolerance = self.epsilon * n

        Ak = self
        V = Matrix.identity(n)

        for _ in range(max_iterations):
            off = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    off += abs(Ak.rows[i][j])
            if off < tolerance:
                break

            # Wilkinson shift
            if n >= 2:
                a = Ak.rows[n-2][n-2]
                b = Ak.rows[n-2][n-1]
                c = Ak.rows[n-1][n-2]
                d = Ak.rows[n-1][n-1]
                trace = a + d
                det = a * d - b * c
                disc = trace * trace - 4.0 * det
                if disc >= 0:
                    sqrt_disc = math.sqrt(disc)
                    eig1 = (trace + sqrt_disc) / 2.0
                    eig2 = (trace - sqrt_disc) / 2.0
                    if abs(eig1 - d) < abs(eig2 - d):
                        shift = eig1
                    else:
                        shift = eig2
                else:
                    shift = d
            else:
                shift = Ak.rows[0][0]

            shifted = Ak - Matrix.identity(n) * shift
            Q, R = shifted.qr()
            Ak = R @ Q + Matrix.identity(n) * shift
            V = V @ Q

        eigenvalues = [Ak.rows[i][i] for i in range(n)]
        eigenvectors = [V.get_column(i) for i in range(n)]
        combined = sorted(zip(eigenvalues,eigenvectors),key=lambda pair: pair[0],reverse=True)
        sorted_eigenvalues = [p[0] for p in combined]
        sorted_eigenvectors = [p[1] for p in combined]
        return sorted_eigenvalues, sorted_eigenvectors
