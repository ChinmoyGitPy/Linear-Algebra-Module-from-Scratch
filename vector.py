import math

class Vector:

    epsilon = 1e-12

    def __init__(self,components):
        if not components:
            raise ValueError("Vector can't be empty")
        if not all(isinstance(x,(float,int)) for x in components):
            raise TypeError("All vector components must be integers or floats")
        
        self.data = [float(x) for x in components]
        self.dimension = len(self.data)

    def __len__(self):
        return self.dimension

    def __getitem__(self,index):
        return self.data[index]

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self,other):
        if isinstance(other,(int,float)):
            return Vector([x+other for x in self.data])
        if isinstance(other,Vector):
            if self.dimension != other.dimension:
                raise ValueError("Vectors must be of equal dimensions")
            return Vector([a + b for a,b in zip(self.data,other.data)])
        return NotImplemented

    def __radd__(self,other):
        return self.__add__(other)

    def __sub__(self,other):
        if isinstance(other,(int,float)):
            return Vector([x-other for x in self.data])
        if isinstance(other,Vector):
            if self.dimension != other.dimension:
                raise ValueError("Vectors must be of equal dimensions")
            return Vector([a - b for a,b in zip(self.data,other.data)])
        return NotImplemented

    def __rsub__(self,other):
        if isinstance(other, (int, float)):
            return Vector([other - x for x in self.data])
        return NotImplemented
    
    def __mul__(self,other):
        if isinstance(other,(int,float)):
            return Vector([x*other for x in self.data])
        return NotImplemented
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __neg__(self):
        return Vector([-x for x in self.data])

    def dot(self,other):
        if self.dimension != other.dimension:
            raise ValueError("Vectors must be of equal dimension for dot product")
        return sum(a*b for a,b in zip(self.data,other.data))
    
    def norm(self,n=2):
        if n==2:
            return math.sqrt(sum(x**2 for x in self.data))
        elif n == 1:
            return sum(abs(x) for x in self.data)
        return (sum(abs(x)**n for x in self.data))**(1.0/n)
    
    def normalise(self,n=2):
        current_norm = self.norm(n)
        if current_norm < self.epsilon:
            raise ValueError("Cannot normalize a zero-vector")
        return Vector([x / current_norm for x in self.data])

    def is_orthogonal_to(self,other):
        return abs(self.dot(other)) < self.epsilon

vec = Vector([2,5])
print(vec+2)