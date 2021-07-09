#Python script that implements Singular Value Decomposition
import numpy as np
from numpy.linalg import norm
from random import normalvariate
from math import sqrt

# generateUnitVector generates an array of unit vectors
# Input : n, an integer
# Output: a list of n unit vectors of float datatype
# Purpose: We will call this function in svdHelper to construct an initial vector containing datapoints ranging from 0 to 1. 
def generateUnitVector(n):
    #Compute initial_vector
    initial_vector = [normalvariate(0, 1) for i in range(n)]
    #Compute the norm for each element in initial_vector
    norm = sqrt(sum(x * x for x in initial_vector))
    return [x / norm for x in initial_vector]

# svdHelper generates an array of unit vectors
# Input : A, a matrix, 
#         tol, a level of tolerance
# Output: a vector that will be converted into one of the columns of U
# Purpose: We will call this function in svd to the columns of U and subsequently the columns of V 
def svdHelper(A, tol=1e-10 ):
    #Compute the dimension of A
    n, m = A.shape
    #Generate a vector containing unit vectors.
    #Note: We could have called np.zeros here, but that would result in division by zero in line 42
    v = generateUnitVector(min(n,m))
    #Initialize prevVector
    prevVector = None
    #Initialize currentVector
    currentVector = v

    #If the matrix is not square, compute A^TA
    if n > m:
        B = np.dot(A.T, A)
    #If the matrix is square, compute AA^T
    else:
        B = np.dot(A, A.T)

    while True:
        prevVector = currentVector
        #Compute B * prevVector
        currentVector = np.dot(B, prevVector)
        #Divide currentVector by the norm
        currentVector = currentVector / norm(currentVector)

        #If the product of currentVector and prevVector is greater than 1 - tol, return currentVector. Otherwise, keep going.
        if abs(np.dot(currentVector, prevVector)) > 1 - tol:
            return currentVector

# svd decompose a matrix A into U, sigma, V^T
# Input : A, a matrix,
#         k, an integer
#         tol, a level of tolerance
# Output: three matrices U, sigma, V^T
# Purpose: We will call this function to decompose A into three matrices
def svd(A, k=None, tol=1e-10):
    #Convert A from list to array
    A = np.array(A, dtype=float)
    #Compute the dimension of A
    n, m = A.shape
    #Initialize currentSVD
    currentSVD = []
    #If k is not given, then initialize k to the lowest of m and n
    if k is None:
        k = min(n, m)

    #Loop through k (the number of columns), each time creating a column of U, and subsequently, a column of V
    for i in range(k):
        matrixHelper = A.copy()
        #Compute A - (\sigma UV)
        for singularValue, u, v in currentSVD[:i]:
            matrixHelper = matrixHelper - singularValue * np.outer(u, v)

        if n > m:
            #Compute the column i-th column of U
            columnU = svdHelper(matrixHelper, tol)  
            u_unnormalized = np.dot(A, columnU)
            sigma = norm(u_unnormalized)  
            columnV = u_unnormalized / sigma
        else:
          #Compute the column i-th column of V
            columnV = svdHelper(matrixHelper, tol)  
            v_unnormalized = np.dot(A.T, columnV)
            sigma = norm(v_unnormalized)
            columnU = v_unnormalized / sigma

        #Add a tuple containing three variables to the list for each iteration
        currentSVD.append((sigma, columnV, columnU))

    #Group the singularValues in one array, columns of U in one array, and columns of V in one array
    singularValues, U, V = [np.array(x) for x in zip(*currentSVD)]

    #Construct Sigma by creating a zero matrix first, and then fill in the diagonal with the values held in singularValues
    Sigma = np.zeros((len(singularValues), (len(singularValues))))
    Sigma[:A.shape[1], :A.shape[1]] = np.diag(singularValues)
       
    return U.T, Sigma, V

#Test Script
if __name__ == "__main__":
    A = np.array([
        [2, 2, 0],
        [-1, 1, 0]
    ])

    U, Sigma, VT= svd(A)
    print("This is U")
    print(U)
    print("This is sigma")
    print(Sigma)
    print("This is VT")
    print(VT)
    print("This is UsigmaVT, which is A")
    print(U.dot(Sigma.dot(VT)))