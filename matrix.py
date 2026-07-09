import math
from vector import Vector

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




