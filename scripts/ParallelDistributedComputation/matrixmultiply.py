import numpy as np

def generateLargeMatrix(size):
    return np.random.randint(10,size=(size,size))

def multiplymatrix(matrix_a,matrix_b):
    return np.dot(matrix_a,matrix_b)

if __name__ == "__main__":
    matrix_size=1000
    print(f"generating two {matrix_size} x {matrix_size} matrices...")
    matrix_a,matrix_b= generateLargeMatrix(matrix_size),generateLargeMatrix(matrix_size)
    print("multiplying the matrices...")
    result = multiplymatrix(matrix_a=matrix_a,matrix_b=matrix_b)
    print(f"multiplication of the two matrix has the shape of {result.shape}")
    
    