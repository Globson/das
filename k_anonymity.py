# https://www.mdai.cat/code/v20230316/prog.vectors.matrices.web.txt

## Software:
## (c) VicenÃ§ Torra
## Current version: 20221201
##
## Companion code of software for non-additive measures and integrals
##    (these measures are also known as fuzzy measures, monotonic games,
##    and include belief functions, k-additive measures,
##    distorted probabilities)
##    Software includes Choquet and Sugeno integrals,
##    measure identification from examples, Shapley values, indices, etc.
##
## Companion code of software for data privacy, and papers on privacy.
##
## References:
##   E. Turkarslan, V. Torra, Measure Identification for the Choquet integral:
##        a Python module, I. J. of Comp. Intel. Systems 15:89 (2022)
##        https://doi.org/10.1007/s44196-022-00146-w
##   V. Torra, Y. Narukawa, Modeling Decisions, Springer, 2007.
##
##   V. Torra, Guide to data privacy, Springer, 2022.

from itertools import starmap
from functools import reduce
import math


### ---------------------------------------------------------------
### Functions for vectors and lists
### ---------------------------------------------------------------


def listSelection(l, loIndices):
    """listSelection([0,1,2,3,4,5],[2,3,4])"""
    return [l[i] for i in loIndices]


def argMin(values, indices=None):
    """Function: return the index of the minimum value
    Example:
    argMin([0,1,-2,3,-1]) ## 2
    argMin([0,1,-2,3,-1],range(0,5))   ## 2
    """
    if indices == None:
        indices = range(0, len(values))
    minV = min(values)
    return indices[values.index(minV)]


## Example:
##    vectorSum([1,2,3],[4,5,6])
def vectorSum(l1, l2):
    return list(map(lambda x, y: x + y, l1, l2))


def vectorProduct(vec1, vec2):
    """
    Function: inner product, or simply product, of two vectors
    Example: product([1,2,3],[5,6,7])
    """
    return sum(map(lambda x, y: x * y, vec1, vec2))


def vectorProductConstant(vec1, alpha):
    """
    Function: inner product, or simply product, of two vectors
    Example: vectorProductConstant([1,2,3],2)
    """
    return list(map(lambda x: x * alpha, vec1))


## Example:
##    mean([1,2,3])
def mean(l1):
    return sum(l1) / len(l1)


##
## Function:
##    normalize a vector so that \sum v_i = 1
## Example:
##    normalize([1,2,3,4]) == [0.1, 0.2, 0.3, 0.4]
def normalize(v):
    s = sum(v)
    return list(map(lambda x: x / s, v))


## Function:
##    Computes the square of the norm of the two vectors ||v1 - v2||^2
## Example:
##    fNorm2([1.0,2.0,3.0],[1.0,2.0,0.0])
##    timeF(lambda :fNorm2([1.0,2.0,3.0,1,2,3,4,5,6,7,8,9,0],[1.0,2.0,0.0,1,2,3,4,5,6,7,8,9,0]),10000)
##    distWEuclidean2([1.0,2.0,3.0],[1.0,2.0,0.0],[0.5,0.2,0.3])
def fNorm(v1, v2):
    return math.sqrt(sum(map(lambda e1, e2: (e1 - e2) ** 2, v1, v2)))


def fNorm2(v1, v2):
    return sum(map(lambda e1, e2: (e1 - e2) ** 2, v1, v2))


def distEuclidean(v1, v2):
    return math.sqrt(sum(map(lambda e1, e2: (e1 - e2) * (e1 - e2), v1, v2)))


##  Example:
##    distWEuclidean2([1.0,2.0,3.0],[1.0,2.0,0.0],[0.5,0.2,0.3])
def distWEuclidean2(v1, v2, weights):
    return sum(map(lambda e1, e2, w: w * (e1 - e2) * (e1 - e2), v1, v2, weights))


## Function:
##    distWeights2WEuclidean ([0.5,0.2,0.3])([1,1,1],[0,0,0])
def distWeights2WEuclidean(weights):
    return lambda v1, v2: distWEuclidean2(v1, v2, weights)


def vectNorm(v1):
    return np.sqrt(sum(map(lambda e: e * e, v1)))


def vectCosineSim(v1, v2):
    """vectCosineSim ([0.0,1.0,0.0],[0.0,1.0,0.0])"""
    return vectorProduct(v1, v2) / (vectNorm(v1) * vectNorm(v2))


## Function:
##   Computes the maximum but there is a minimum value for this maximum
## Example:
##   maxWithMinimum([-1,-2],0.0)
def maxWithMinimum(lof, mn):
    if lof == []:
        mx = mn
    else:
        mx = max(lof)
    mx = max([mx, mn])
    return mx


## Function:
##    Select lofe[i] if f(lofv[i)) is true
## Example:
##    mat = [[1.0,2.0,3.0],[2.0,2.0,4.0],[3.0,2.0,5.0],[2.0,2.0,4.0]]
##    selectFv(mat, [0,1,1,1], lambda x:x==0)
##    selectFv(mat, [0,1,1,1], lambda x:x==1)
##    selectFv(mat, [0,1,1,1], lambda x:x==1)
def selectFv(lofe, lofv, f):
    res = []
    for i in range(0, len(lofe)):
        if f(lofv[i]):
            res.append(lofe[i])
    return res


### ---------------------------------------------------------------
### Functions for matrices (i.e., lists of lists)
### ---------------------------------------------------------------


def vec2mat(vec):
    """
    Transforms a list into column matrix
    """
    return list(map(lambda e: [e], vec))


## Select a set of records, and the remaining set of records
def matrixSelection(X, loIndices):
    notInLoIndices = [i for i in range(0, len(X)) if i not in loIndices]
    return (listSelection(X, loIndices), listSelection(X, notInLoIndices))


## Select a set of columns records, and the remaining set of colums
## mat = [[1.0,2.0,3.0],[2.0,2.0,4.0],[3.0,2.0,5.0],[2.0,2.0,4.0]]
## matrixColumnSelection(mat,[1,2])
def matrixColumnSelection(X, loIndices):
    notInLoIndices = [i for i in range(0, len(X[0])) if i not in loIndices]
    # print(notInLoIndices)
    selectedX = []
    unSelectedX = []
    for i in range(0, len(X)):
        colsSelected, colsOther = matrixSelection(X[i], loIndices)
        selectedX.append(colsSelected)
        unSelectedX.append(colsOther)
    return (selectedX, unSelectedX)


## matrixProduct([[1,2],[2,3]],[[1],[2]])   # [[5], [8]]
## matrixProduct([[1],[2],[3]],[[1],[2],[3]]) # [[1], [2], [3]] -- WRONG DIMENSIONS
## matrixProduct([[1],[2],[3]],[[1,2,3]])   # [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
def matrixProduct(mat1, mat):
    mat2 = transpose(mat)
    return list(
        map(lambda row: list(map(lambda col: vectorProduct(row, col), mat2)), mat1)
    )


def matrixProductConstant(mat, a):
    """
    matrixProductConstant([[1,2],[2,3]],4.1)
    """
    return list(map(lambda row: list(map(lambda col: col * a, row)), mat))


def matrixSum(mat1, mat2):
    """
    matrixSum([[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]])
    """
    return list(
        map(
            lambda row1, row2: list(map(lambda e1, e2: e1 + e2, row1, row2)), mat1, mat2
        )
    )


def transpose(mat):
    """
    Example:
    transpose([[1,2],[3,4]])           #  [[1, 3], [2, 4]]
    transpose([[1,2],[3,4],[5,6]])     #  [[1, 3, 5], [2, 4, 6]]
    """
    nRows = len(mat)
    nCols = len(mat[0])
    nMat = []
    for i in range(0, nCols):
        nRow = []
        for j in range(0, nRows):
            nRow.append(mat[j][i])
        nMat.append(nRow)
    return nMat


# def transpose(mat):
#    if (mat[0]==[]):
#        return []
#    else:
#        firstRow = list(map(lambda row: row[0], mat))
#        return ([firstRow] +
#                transpose(list(map(lambda row: row[1::], mat))))


## Function:
##    apply e^alpha to all elements e in the matrix
## Example:
##    matrix2alpha([[1,2,3],[0,1,2],[-1,-2,-3]], 2)
def matrix2alpha(mat, alpha):
    return list(map(lambda row: list(map(lambda e: e**alpha, row)), mat))


## Function:
##    Compute the means of all columns in the "matrix"
## Examples:
##    mat = [[1.0,2.0,3.0],[2.0,2.0,4.0],[3.0,2.0,5.0],[2.0,2.0,4.0]]
##    matrixColumnSums(mat)
##    matOfMat = [mat,mat,mat]
##    list(map(matrixColumnSums,matOfMat))
##    matrixColumnSums([])
def matrixColumnSums(mat):
    numRows = len(mat)
    res = []
    if numRows != 0:
        numCols = len(mat[0])
        # print(numCols)
        for i in range(numCols):
            res.append(sum(map(lambda x: x[i], mat)))
    return res


## Function:
##    Apply f column wise
## Examples
##    matrixColumnFunction(mat,max)
##    matrixColumnFunction(mat,min)
def matrixColumnFunction(mat, f):
    numRows = len(mat)
    res = []
    if numRows != 0:
        numCols = len(mat[0])
        # print(numCols)
        for i in range(numCols):
            res.append(f(list(map(lambda x: x[i], mat))))
    return res


## Function:
##    Compute the means of all columns in the "matrix"
##    At least one row in the matrix
## Example:
##    matrixColumnSumsReduce(mat)
##    matrixColumnSumsReduce([])
def matrixColumnSumsReduce(mat):
    return list(
        reduce(
            lambda acc, rowi: (map(lambda x, y: x + y, rowi, acc)),
            mat,
            [0] * len(mat[0]),
        )
    )


## Function:
##    Compute the min and maximum of all columns
## Example:
##    matrixColumnMinMax(mat)
def matrixColumnMinMax(mat):
    lMin = mat[0].copy()
    lMax = mat[0].copy()
    for i in range(0, len(mat)):
        lMin = list(map(lambda x, y: min(x, y), lMin, mat[i]))
        lMax = list(map(lambda x, y: max(x, y), mat[i], lMax))
    return (lMin, lMax)


## Function:
##    Compute the means of all columns in the "matrix"
## Examples:
##    mat = [[1.0,2.0,3.0],[2.0,2.0,4.0],[3.0,2.0,5.0],[2.0,2.0,4.0]]
##    matrixColumnMeans(mat)
##    matOfMat = [mat,mat,mat]
##    list(map(matrixColumnMeans,matOfMat))
##    matrixColumnMeans([])
def matrixColumnMeans(mat):
    numRows = len(mat)

    res = []

    if numRows != 0:
        res = np.mean(mat, axis=0)

    return res


def matrixColumnMeansAbs(mat):
    numRows = len(mat)
    res = []
    if numRows != 0:
        numCols = len(mat[0])
        # print(numCols)
        for i in range(numCols):
            res.append((sum(map(lambda x: abs(x[i]), mat))) / numRows)
    return res


## Function:
##    compute the standard deviation of each column
## Examples:
##    mat = [[1.0,2.0,3.0],[2.0,2.0,4.0],[3.0,2.0,5.0],[2.0,2.0,4.0]]
##    matrixColumnSD(mat)
def matrixColumnSD(mat):
    numRows = len(mat)
    numCols = len(mat[0])
    colMeans = matrixColumnMeans(mat)
    res = []
    for i in range(numCols):
        res.append(
            math.sqrt(
                sum(map(lambda x: (x[i] - colMeans[i]) ** 2.0, mat)) / (numRows - 1)
            )
        )
    return res


## Function:
##    compute the variance of each column
## Examples:
##    mat = [[1.0,2.0,3.0],[2.0,2.0,4.0],[3.0,2.0,5.0],[2.0,2.0,4.0]]
##    matrixColumnVariance(mat)
def matrixColumnVariance(mat):
    sd = matrixColumnSD(mat)
    return list(map(lambda x: x * x, sd))


## mat = [[1.0,2.0,3.0],[2.0,2.0,4.0],[3.0,2.0,5.0],[2.0,2.0,4.0]]
## matrixColumnSD(mat)


## Function:
##    (x_ij -  mean(x_.j))/sd(x_.j)
## Example:
##     mat = [[1.0,2.0,3.0],[2.0,2.0,4.0],[3.0,2.0,5.0],[2.0,2.0,4.0]]
##     normalizeMatrixByColumn(mat)
##     normalizeMatrixByColumn(mat)
def normalizeMatrixByColumn(mat):
    colMeans = matrixColumnMeans(mat)
    colSD = matrixColumnSD(mat)
    numRows = len(mat)
    res = []
    for i in range(numRows):
        res.append(
            list(
                map(
                    lambda xij, meanj, sdj: 0 if (sdj == 0) else (xij - meanj) / sdj,
                    mat[i],
                    colMeans,
                    colSD,
                )
            )
        )
    return (colMeans, colSD, res)


def restoreNormalizationMatrixByColumn(colMeans, colSD, matNorm):
    numRows = len(matNorm)
    res = []
    for i in range(numRows):
        res.append(
            list(
                map(
                    lambda xNij, meanj, sdj: meanj
                    if (sdj == 0)
                    else xNij * sdj + meanj,
                    matNorm[i],
                    colMeans,
                    colSD,
                )
            )
        )
    return res


## Function:
##    (x_ij - min(x_.j))/(max(x_.j) - min(x_.j))
## Example:
##     mat = [[1.0,2.0,3.0],[2.0,2.0,4.0],[3.0,2.0,5.0],[2.0,2.0,4.0]]
##     lMin, lMax, matp = matrixScaleMinMax(mat)
def matrixScaleMinMax(mat):
    lMin, lMax = matrixColumnMinMax(mat)
    res = []
    for i in range(0, len(mat)):
        res.append(
            list(
                map(
                    lambda ri, mn, mx: (ri - mn) / (mx - mn) if mn != mx else 0.5,
                    mat[i],
                    lMin,
                    lMax,
                )
            )
        )
    return (lMin, lMax, res)


## Function:
##    min(x_.j) + (max(x_.j) - min(x_.j))*sx_ij
## Example:
##     mat = [[1.0,2.0,3.0],[2.0,2.0,4.0],[3.0,2.0,5.0],[2.0,2.0,4.0]]
##     lMin, lMax, matp = matrixScaleMinMax(mat)
##     matrixUnScaleMinMax(lMin, lMax, matp)
def matrixUnScaleMinMax(lMin, lMax, mat):
    res = []
    for i in range(0, len(mat)):
        res.append(
            list(
                map(
                    lambda sri, mn, mx: mn + (mx - mn) * sri if mn != mx else mx,
                    mat[i],
                    lMin,
                    lMax,
                )
            )
        )
    return res


## Function:
##     return position of the farthest record to vect
## Example:
##     mat = [[1.0,2.0,3.0],[2.0,2.0,4.0],[3.0,2.0,5.0],[2.0,2.0,4.0]]
##     farthestRow(mat,[1.0,2.0,3.0])
def farthestRow(db, vect, vDistance=fNorm2):
    selectedRow = 0
    dRow = vDistance(vect, db[selectedRow])
    for i in range(1, len(db)):
        d = vDistance(vect, db[i])
        if d > dRow:
            # print("Row="+str(i)+" d="+str(d))
            dRow = d
            selectedRow = i
    return selectedRow


### ---------------------------------------------------------------
### Functions: other functions
### ---------------------------------------------------------------


## From:
##   https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
## Example:
##   flatten([[[1,1],[2,2]],[[3,3,3]],[[4,4,4],[4,4,4]]])
##   --> [[1, 1], [2, 2], [3, 3, 3], [4, 4, 4], [4, 4, 4]]
##   flatten([[1,2,3],[4,4,4],[3],[[[100]]]])
##   -->  [1, 2, 3, 4, 4, 4, 3, [[100]]]
def flatten(t):
    return [item for sublist in t for item in sublist]


## Function:
##   Input: pairs (n_i, v_i)
##   Output: a list with n_i copies of each v_i
## Example:
##   nCopies([(1,[1,1,1]), (2,[4,4,4]), (5,[8,9])])
##   --> [[1, 1, 1], [4, 4, 4], [4, 4, 4], [8, 9], [8, 9], [8, 9], [8, 9], [8, 9]]
## IMPORTANT NOTE: the elements in v are shared !!
##   nCopies([(1,[1,1,1]), (2,[4,4,4]), (5,[8,9])])
##   a=[1,2,3]
##   b=nCopies([(1,a),(2,a),(3,a)])
##   a[2]=1000
##   b
def nCopies(listOfPairs):
    return list(flatten(starmap(lambda n, v: [v] * n, listOfPairs)))


def remove(l, e):
    """Function: Delete, remove an element from a list.
    Example: remove([1,2,3,4,5],1)"""
    if l == []:
        return l
    elif e == l[0]:
        return l[1:]
    else:
        return [l[0]] + remove(l[1:], e)


### ---------------------------------------------------------------
### Files
### ---------------------------------------------------------------

import csv


## Function:
##   read a file. First line is header
## Example:
##   fheader, fRows = readNumericalFileWithHeader("Concrete_DataVarsV1V7.csv")
##   print(fRows)
##   print(fheader)
def readNumericalFileWithHeader(nameFile):
    file = open(nameFile)
    csvreader = csv.reader(file)
    header = next(csvreader)
    ## print(header)
    rows = []
    for row in csvreader:
        nrow = list(map(lambda e: (float(e)), row))
        rows.append(nrow)
    file.close()
    return (header, rows)


def readNumericalFileWithComments(nameFile):
    file = open(nameFile)
    csvreader = csv.reader(file)
    ## print(header)
    rows = []
    for row in csvreader:
        if row[0][0] != "#":
            nrow = list(map(lambda e: (float(e)), row))
            rows.append(nrow)
    file.close()
    return rows


# db = readNumericalFileWithComments("abalone.data.numeric.csv")


### ---------------------------------------------------------------
### OTHER AUXILIARY FUNCTIONS
### ---------------------------------------------------------------


def printAndReturn(e):
    print(e)
    return e


### ---------------------------------------------------------------
###  FUNCTIONS ON SETS
### ---------------------------------------------------------------


def equalSets(mySet1, mySet2, n):
    """
    Function: Check if the two sets are equal
    Example: equalSets([0,1,1,0],[0,1,1,0],4)
    """
    return mySet1 == mySet2


def setIntersection(mySet1, mySet2, n):
    """
    Function: Compute the intersection of two sets.
    Example:
    setIntersection([0,1,1],[1,1,0],3)
    """
    return list(map(lambda x, y: min(x, y), mySet1, mySet2))


def setUnion(mySet1, mySet2, n):
    """
    Function: Compute the intersection of two sets.
    Example:
    setUnion([0,1,1],[1,1,0],3)
    """
    return list(map(lambda x, y: max(x, y), mySet1, mySet2))


def subseteq(mySet1, mySet2, n):
    """
    Function: mySet1 \\subseteq mySet2?
    Example:
    subseteq([0,1,1,0],[0,1,1,1],4) # == True
    subseteq([1,1,1,0],[0,1,1,1],4) # == False
    """
    subset = True
    i = 0
    while subset and (i < n):
        if mySet1[i] == 1:
            subset = mySet2[i] == 1
        i = i + 1
    return subset


def belongSet(i, mySet, n):
    """
    Function: Check if an element is in the set
    Example:
    belongSet(2, [0,1,1,0],4)
    belongSet(0, [0,1,1,0],4)
    """
    return mySet[i] == 1


def addElement2Set(i, mySet, n):
    """
    Function: add an element to the set
    Example:
    addElement2Set(2, [0,1,1,0],4)
    addElement2Set(0, [0,1,1,0],4)
    """
    mySet[i] = 1
    return mySet


def removeElement2Set(i, mySet, n):
    """
    Function: remove an element from a set
    Example:
    removeElement2Set(2, [0,1,1,0],4)
    removeElement2Set(0, [0,1,1,0],4)
    """
    mySet[i] = 0
    return mySet


def firstSet(n):
    """
    Function: returns first set == [0,...,0]
    Example: firstSet(4) # == [0,0,0,0]
    """
    return [0] * n


def lastSet(mySet):
    """
    Function: Last possible set?
    Example:
    lastSet([1,1,1,1]) # == True
    lastSet([1,0,1,1]) # == False
    """
    return not (0 == min(mySet))


def notLastSet(mySet):
    """
    Function: still other sets?
    Example:
    notLastSet([1,1,1,1]) # == False
    notLastSet([1,0,1,1]) # == True
    """
    return 0 == min(mySet)


def nextSet(mySet, n):
    """
    Function: Given a set, next one (assume not last one!)
    Example: nextSet([0,0,1,0],4) #==[0,0,1,1]
    """
    i = n - 1
    if mySet[i] == 0:
        mySet[i] = 1
    else:
        while mySet[i] == 1:
            mySet[i] = 0
            i = i - 1
        mySet[i] = 1
    return mySet


def nElemsSet(mySet, n):
    """
    Function: count number of elements in the set
    Example: nElemsSet([1,0,1,1],4)
    """
    return sum(mySet)


##  In Scala vect2int
def set2Int(vec):
    """
    Function: given a set [0,1,1,0] returns the integer that represents
    Example:
    set2Int([1,1,1,1]) # 15
    set2Int([1,1,0,1]) # 13
    """
    return sum(map(lambda x, i: x * (2**i), vec, list(range(len(vec) - 1, -1, -1))))


def setSet2Index(mySet, n, nameElements=None):
    """
    Function: Given a set [0,1,1,0] returns the list of elements in the set
    Examples:
    setSet2Index ([0,1,1,0,1],5)    # [1, 2, 4]
    setSet2Index ([0,1,1,0,1],5,["a","b","c","d","e"])
    """
    if nameElements == None:
        nameElements = range(0, n)
    return [nameElements[i] for i in range(0, n) if mySet[i] == 1]


def setIndex2Set(vec, n):
    """
    Function: given a set of indices [1,4] returns a set (i.e., a list)
    Example:
    setIndex2Set([0],6) # == [1, 0, 0, 0, 0, 0]
    setIndex2Set([1],6) # == [0, 1, 0, 0, 0, 0]
    setIndex2Set([0,1],6) # == [1, 1, 0, 0, 0, 0]
    setIndex2Set([5],6) # == [0, 0, 0, 0, 0, 1]
    """
    mySet = [0] * n
    for i in vec:
        mySet[i] = 1
    return mySet


def setIndex2Int(vec, n):
    """
    Function: given a set of indices [1,4] returns the integer that represents
    Example:
    setIndex2Int([0],6) # == 32
    setIndex2Int([1],6) # == 16
    setIndex2Int([0,1],6)# == 48
    setIndex2Int([0,3],6)# == 36
    setIndex2Int([5],6)# == 1
    """
    return set2Int(setIndex2Set(vec, n))


def setAllElems2ElemsInSet(mySet, allElems):
    """Example:
    setAllElems2ElemsInSet([0,1,0,1],[1,0,-45,2],4)
    """
    return [x[0] for x in zip(allElems, mySet) if x[1] == 1]


def permutations(l):
    """Function: Build al permutations for the elements in l.
    Example: permutations([1,2,3,4,5])"""
    if l == []:
        return []
    elif len(l) == 1:
        return [[l[0]]]
    else:
        return flatten(
            list(
                map(
                    lambda e: list(
                        map(lambda perm: [e] + perm, permutations(remove(l, e)))
                    ),
                    l,
                )
            )
        )


#######################################################################################################
# https://www.mdai.cat/code/v20230316/prog.sdc.web.txt

## Software for anonymization.
## (c) VicenÃ§ Torra
## Current version: 20221201
##
## References on data privacy:
##    V. Torra, Guide to data privacy, Springer, 2022.
##
## Main functions:
##
## Masking methods:
##    mdav (db, k)                 ## MDAV, k size of clusters
##    noiseAddition (x, k)         ## noise addition, k parameter for quantity of noise
##    noiseAdditionLaplace (x, k)  ## noise addition Laplace, k parameter for quantity of noise
##    maskingNMF(data, quality)    ## masking using non-negative matrix factorization, parameter: quality
##    maskingSVD (data, quality)   ## masking using singular value decomposition, parameter: quality
##    maskingPCA (data, quality)   ## masking using principal components, parameter: quality
##    mondrian (db, k)             ## mondrian
##
## Information loss:
##    sdcIL_stats (x, xp)
##    sdcIL_mlRE (x, xp, y, testX, testY, modelRegression)
##    sdcIL_mlRE_GivenOErr (oError, xp, y, testX, testY, modelRegression)
##
## Disclosure risk using record linkage
##    sdcRecordLinkage (x, xp):
##    sdcRecordLinkageLoL (x, xp, varsX, varsXp):
##
## Other functions related to masking
##    maFromCl2DB (db, assignedCl) ## from clusters to values
##    mdavCl (db, k)               ## mdav clusters
##    mondrianCl (db, k)           ## mondrian clusters
##
##
## OTHER FUNCTIONS for microaggregation
##    remainingCl(clNumber, assignedClass)  ## from clusters to
##    updateCl (indexCls, clNumber, assignedClass):
##    kClosestToVect (db, assignments, vect, k):
##    orderWithList (toOrder, db, indexDb):
##    addOrderedWithList(toOrder, dbOrder, indexDb, d, vect, indexVect):
##
## OTHER FUNCTIONS for PCA
##    graphFromDiagonal2MatrixRC (nRows, nColumns, diagonal):
##
## OTHER FUNCTIONS for Mondrian:
##    mondrianWithUnassigned (db, k, assignedCl=None, firstNCl = 0, stillToProcess = -1):
##    partitionAttributeValue(unassigned, selectedAttribute):
##    indexLowMidHighValues(db, assignedCl, idCl, midValue, selectedAttribute):


import numpy as np
import pandas as pd


## This function computes a masked file from the db and an assignment of records to clusters
## This is the common part of both mdav and mondrian
def maFromCl2DB(db, assignedCl):
    values = set(assignedCl)
    centroids = []
    for cl in values:
        rowsCl = selectFv(db, assignedCl, lambda x: x == cl)
        meanRow = matrixColumnMeans(rowsCl)
        centroids.append(meanRow)
    newDb = [centroids[assignedCl[i]] for i in range(0, len(db))]
    return newDb


## Examples:
##    mdav([[100],[200],[300],[400]],3)
##    mdav([[100],[200],[300],[400],[500],[600]],3)
##    mdav([[100],[200],[300],[400],[500],[600],[700]],3)
##    mdav([[100],[100],[200],[300],[400],[500],[600],[700]],3)
##    mdav([[100],[100],[100],[200],[300],[400],[500],[600],[700],[700]],3)
##    mdav([[100],[100],[100],[200],[300],[400],[500],[600],[700],[700]],3)
##    mdav([[100],[100],[100],[200],[300],[400],[500],[600],[600],[700],[700]],3)
##    mdav([[100],[100],[100],[200],[300],[400],[500],[600],[600],[600],[700],[700]],3)
##    mdav([[100],[100],[100],[200],[300],[400],[500],[600],[650],[600],[700],[700]],3)
##    mdav(fRows, 3)
def mdav(df: pd.DataFrame, QIs: list, k: int) -> pd.DataFrame:
    new_df = df.copy()
    db = new_df[QIs].to_numpy()
    assignedCl = mdavCl(db, k)
    values = set(assignedCl)
    centroids = []

    max_value = db.max(axis=0)
    min_value = db.min(axis=0)
    ncp = 0

    for cl in values:
        rowsCl = selectFv(db, assignedCl, lambda x: x == cl)
        mat = np.array(rowsCl)

        max_row = mat.max(axis=0)
        min_row = mat.min(axis=0)

        # print(mat)
        # print(max_row)
        # print(min_row)

        ncp += sum(((max_row - min_row) * mat.shape[0]) / (max_value - min_value))

        meanRow = matrixColumnMeans(mat)
        centroids.append(meanRow)

    newDb = [centroids[assignedCl[i]] for i in range(0, len(db))]
    new_df[QIs] = newDb

    n_registers = new_df.shape[0]
    n_attributes = new_df.shape[1]
    ncp = ncp / (n_registers * n_attributes)
    print(f'ncp={ncp}')

    return new_df


## Function:
##    clustering MDAV for a database db and parameter k
## Examples:
##    mdavCl([[100],[200],[300],[400]],3)
##    mdavCl([[100],[200],[300],[400],[500],[600]],3)
##    mdavCl([[100],[200],[300],[400],[500],[600],[700]],3)
##    mdavCl([[100],[100],[200],[300],[400],[500],[600],[700]],3)
##    mdavCl([[100],[100],[100],[200],[300],[400],[500],[600],[700],[700]],3)
##    mdavCl([[100],[100],[100],[200],[300],[400],[500],[600],[700],[700]],3)
##    mdavCl([[100],[100],[100],[200],[300],[400],[500],[600],[600],[700],[700]],3)
##    mdavCl([[100],[100],[100],[200],[300],[400],[500],[600],[600],[600],[700],[700]],3)
##    mdavCl([[100],[100],[100],[200],[300],[400],[500],[600],[650],[600],[700],[700]],3)
##    mdavCl(fRows, 3)
def mdavCl(db, k):
    ## if (len(db)<2*k):
    ##     cl = [0]*len(db)
    ## else:
    assignedClass = [-1] * len(db)
    C = []
    clNumber = -1
    nPendingElements = len(db)
    while nPendingElements >= 3 * k:
        unassigned = selectFv(db, assignedClass, lambda x: x == -1)
        meanX = matrixColumnMeans(unassigned)
        xr = farthestRow(unassigned, meanX)
        xs = farthestRow(unassigned, unassigned[xr])
        # print("xr="+str(xr))
        # print("xs="+str(xs))
        toO, dbO, indexCr = kClosestToVect(db, assignedClass, unassigned[xr], k)
        clNumber = clNumber + 1
        assignedClass = updateCl(indexCr, clNumber, assignedClass)
        toO, dbO, indexCs = kClosestToVect(db, assignedClass, unassigned[xs], k)
        clNumber = clNumber + 1
        assignedClass = updateCl(indexCs, clNumber, assignedClass)
        nPendingElements = nPendingElements - 2 * k
    if nPendingElements >= 2 * k:
        unassigned = selectFv(db, assignedClass, lambda x: x == -1)
        meanX = matrixColumnMeans(unassigned)
        xr = farthestRow(unassigned, meanX)
        # print("xr="+str(xr))
        toO, dbO, indexCr = kClosestToVect(db, assignedClass, unassigned[xr], k)
        clNumber = clNumber + 1
        assignedClass = updateCl(indexCr, clNumber, assignedClass)
        nPendingElements = nPendingElements - k
    clNumber = clNumber + 1
    assignedClass = remainingCl(clNumber, assignedClass)
    return assignedClass


# Function:
#   assign unassigned positions to class clNumber
# Example:
#   remainingCl(500,[-1,-1,2,3,4,5,6,-1,8,9,0])
def remainingCl(clNumber, assignedClass):
    for i in range(0, len(assignedClass)):
        if assignedClass[i] == -1:
            assignedClass[i] = clNumber
    return assignedClass


## remainingCl(500,[-1,-1,2,3,4,5,6,-1,8,9,0])


# Function:
#    add to all indices in indexCs the class identifier: clNumber
# Example:
#    updateCl([0,1,2,6], 500, [0,1,2,3,4,5,6,7,8,9,10])
# NOTE:
#    the index should be within the range
def updateCl(indexCls, clNumber, assignedClass):
    for i in range(0, len(indexCls)):
        assignedClass[indexCls[i]] = clNumber
    return assignedClass


# Function:
#   select the nearest k rows in Db and return them with the corresponding assignments
# Example:
#   kClosestToVect([[50],[60],[10],[40],[80],[70]],[-1,-1,-1,-1,-1,-1], [40], 3) == ([0, 100, 400], [[40], [50], [60]], [3, 0, 1])
#   kClosestToVect([[50],[60],[10],[40],[80],[70]],[-1,-1,2,-1,-1,1], [40], 3) == ([0, 100, 400], [[40], [50], [60]], [3, 0, 1])
def kClosestToVect(db, assignments, vect, k):
    toOrder = []
    dbOrder = []
    indexDb = []
    i = 0
    addedRows = 0
    while addedRows < k:
        if assignments[i] == -1:
            # print("addedRows="+str(addedRows)+":(v,db[i="+str(i)+"])="+str(vect)+","+str(db[i]))
            toOrder.append(fNorm2(vect, db[i]))
            dbOrder.append(db[i])
            indexDb.append(i)
            addedRows = addedRows + 1
        i = i + 1
    toOrder, dbOrder, indexDb = orderWithList(toOrder, dbOrder, indexDb)
    while i < len(db):
        if assignments[i] == -1:
            d = fNorm2(vect, db[i])
            if d < toOrder[k - 1]:
                toOrder, dbOrder, indexDb = addOrderedWithList(
                    toOrder, dbOrder, indexDb, d, db[i], i
                )
        i = i + 1
    return (toOrder, dbOrder, indexDb)


## Function:
##    Order in increasing order using a vector
## Example
##    orderWithList([1,4,6,3],[11,44,66,33],[100,400,600,300])
## == ([1, 3, 4, 6], [11, 33, 44, 66], [100, 300, 400, 600])
def orderWithList(toOrder, db, indexDb):
    for i in range(0, len(toOrder)):
        for j in range(i + 1, len(toOrder)):
            if toOrder[j] < toOrder[i]:
                swap = toOrder[j]
                toOrder[j] = toOrder[i]
                toOrder[i] = swap
                swap = db[j]
                db[j] = db[i]
                db[i] = swap
                swap = indexDb[j]
                indexDb[j] = indexDb[i]
                indexDb[i] = swap
    return (toOrder, db, indexDb)


## Function:
##   add a tuple (vect, distance, index) in an already ordered list with distances
##   in increasing order
## Example:
##   addOrderedWithList([1,3,4,6],[11,33,44, 66],[100,300, 400, 600], 2, 22, 200)
##   addOrderedWithList([1,3,4,6],[11,33,44, 66],[100,300, 400, 600], -2, -22, -200)
def addOrderedWithList(toOrder, dbOrder, indexDb, d, vect, indexVect):
    l = len(toOrder)
    if toOrder[l - 1] > d:
        i = 0
        while toOrder[i] <= d:
            i = i + 1
        j = l - 1
        ## toOrder[i]>d,    put at toOrder[i]=d
        while j > i:
            toOrder[j] = toOrder[j - 1]
            dbOrder[j] = dbOrder[j - 1]
            indexDb[j] = indexDb[j - 1]
            j = j - 1
        toOrder[i] = d
        dbOrder[i] = vect
        indexDb[i] = indexVect
    return (toOrder, dbOrder, indexDb)


## Function:    NOTE normal (0, variance = sigma^2)     sd = \sqrt(variance)
##    additive noise with parameter k
## Example:
##    noiseAddition (mat, 0)
##    noiseAddition (mat, 1)
##    noiseAddition (mat, 2)
def noiseAddition(x, k):
    variance = matrixColumnVariance(x)
    vectorParam = list(map(lambda v: np.sqrt(v * k), variance))
    xp = []
    for i in range(0, len(x)):
        xp.append(
            vectorSum(x[i], list(map(lambda sd: np.random.normal(0, sd), vectorParam)))
        )
    return xp


## NOTE laplace (0, b)               Variance =	2 b^2       b = sqr(variance/2)
##
def noiseAdditionLaplace(x, k):
    variance = matrixColumnVariance(x)
    vectorParam = list(map(lambda v: np.sqrt(v * k / 2), variance))
    xp = []
    for i in range(0, len(x)):
        xp.append(
            vectorSum(x[i], list(map(lambda b: np.random.laplace(0, b), vectorParam)))
        )
    return xp


from sklearn.decomposition import NMF


## Function:
##    masking using non-negative matrix factorization, with parameter quality (n_components)
## Example:
##    maskingNMF(mat, 1)
##    maskingNMF(mat, 2)
def maskingNMF(data, quality, scale=False):
    if scale:
        lMin, lMax, data = matrixScaleMinMax(data)
    dArray = np.array(data)
    nmf = NMF(
        n_components=quality, init="random", max_iter=1000
    )  # maskingNMF() ## mat, 2, None)
    W = nmf.fit_transform(dArray)
    H = nmf.components_
    dArrayMasked = np.dot(W, H)
    if scale:
        dArrayMasked = matrixUnScaleMinMax(lMin, lMax, dArrayMasked)
    # print(dArrayMasked)
    ## %%# NMF(beta=0.001, eta=0.0001, init='random', max_iter=2000,nls_max_iter=20000, random_state=0, sparseness=None,tol=0.001)
    ## d = model.fit_transform(data, y=None, W=None, H=None)
    return dArrayMasked


# sdcIL_stats(f1080_fRows,maskingNMF(f1080_fRows, 13))
# sdcIL_stats(f1080_fRows,maskingSVD(f1080_fRows, 13))

# testa0 = list(map(lambda x:sdcIL_stats(f1080_fRows,maskingSVD(f1080_fRows, x)), [5,10,15,20,25,30,40,50,60]))
# testa1 = list(map(lambda x:sdcIL_stats(f1080_fRows,maskingNMF(f1080_fRows, x)), [5,10,15,20,25,30,40,50,60,70,80]))
# testa2 = list(map(lambda x:sdcIL_stats(f1080_fRows,maskingNMF(f1080_fRows, x)), [5,10,15,20,25,30,40,50,60,70,80]))
# testa3 = list(map(lambda x:sdcIL_stats(f1080_fRows,maskingNMF(f1080_fRows, x)), [5,10,15,20,25,30,40,50,60,70,80,90,100]))

import copy


## Function:
##    masking using singular value decomposition, with parameter quality (n_components)
## Example:
##    maskingSVD(mat, 1)
##    maskingSVD(mat, 2)
def maskingSVD(data, quality):
    nRows = len(data)
    kMin = min(quality, nRows)
    U, s, Vh = np.linalg.svd(np.array(data))
    sZero = copy.deepcopy(s)
    # for i in range(k+1, len(s)):
    #    sZero[i]=0
    mS = graphFromDiagonal2MatrixRC(nRows, len(data[0]), sZero)
    Uk = U[:, :kMin]
    mSk = mS[:kMin, :kMin]
    Vhk = Vh[:kMin, :]
    # SVhk = mSk.dot(Vhk)
    USVhk = Uk.dot(mSk.dot(Vhk))
    return USVhk
    # return(U, mS, Vh, USVhk, s)


def graphFromDiagonal2MatrixRC(nRows, nColumns, diagonal):
    """
    From the diagonal of a matrix to a matrix of nRowsxnColumns
    Example:
    graphFromDiagonal2MatrixRC (3, 4, [1,2,3])
    """
    S = np.zeros((nRows, nColumns))
    lD = len(diagonal)
    S[:lD, :lD] = np.diag(diagonal)
    return S


from sklearn.decomposition import PCA


## Function:
##    masking using PCA, with parameter quality (n_components)
## Example:
##    maskingPCA(mat, 1)
##    maskingPCA(mat, 2)
def maskingPCA(data, quality):
    pca = PCA(n_components=quality)
    data_pca = pca.fit_transform(data)
    data_pca_invers = pca.inverse_transform(data_pca)
    return data_pca_invers


# testpca1 = list(map(lambda x:sdcIL_stats(f1080_fRows,maskingPCA(f1080_fRows, x)), [1,2,3,4,5,6,7,8,9,10,11,12,13]))
# testpca1 = list(map(lambda x:sdcIL_stats(teX,maskingPCA(teX, x)), [1,2,3,4,5,6,7,8,9,10]))


## Information Loss: IL ------------------------------------------


## Function:
##   a function to compute information loss in terms of
##     1) difference of means
##     2) difference of standard deviations
##     3) maximum difference between the two matrices
## Example:
##   sdcIL_stats(mdav(fRows, 3), fRows)
def sdcIL_stats(df_anon: pd.DataFrame, df_orig: pd.DataFrame, QIs: list):
    # x = [[i] for i in df_anon[QI]]
    # xp = [[i] for i in df_orig[QI]]
    x = df_anon[QIs].to_numpy()
    xp = df_orig[QIs].to_numpy()
    meanX = matrixColumnMeans(x)
    sdX = matrixColumnSD(x)
    meanXp = matrixColumnMeans(xp)
    sdXp = matrixColumnSD(xp)
    dMean = fNorm2(meanX, meanXp)
    dSD = fNorm2(sdX, sdXp)
    dMax = max(list(map(lambda x: max(x), np.subtract(x, xp))))
    return (dMean, dSD, dMax)


def sdcRegressionError(XTrIndep, yTrDep, XTeIndep, yTeDep, modelRegression):
    model = modelRegression()
    model.fit(XTrIndep, yTrDep)
    diff = model.predict(XTeIndep) - yTeDep
    myError = np.sqrt(sum(diff * diff))
    score = model.score(XTeIndep, yTeDep)
    # print("Score="+str(score)+", myError="+str((myError/len(yTeDep))))
    return (myError / len(yTeDep), diff)


def sdcIL_mlRE(x, xp, y, testX, testY, modelRegression):
    """
    Function:
    #  Compare error of models for both original and masked file
    #    for a given test set
    Input:
    #  x:  Input variables, Original file
    #  xp: Input variables, Protected file
    #  y:  Output variables, for both original and protected (not modified)
    #  testX: Input variable, records for testing
    #  testY: Output variables, records for testing
    #  modelRegression:
    """
    originalError = sdcRegressionError(x, y, testX, testY, modelRegression)
    maskedError = sdcRegressionError(xp, y, testX, testY, modelRegression)
    return maskedError[0] - originalError[0]


def sdcIL_mlRE_GivenOErr(oError, xp, y, testX, testY, modelRegression):
    """
    Function:
    #  Compare error of models for both original and masked file
    #    for a given test set
    Input:
    #  originalError:  Original Error is given
    #  xp: Input variables, Protected file
    #  y:  Output variables, for both original and protected (not modified)
    #  testX: Input variable, records for testing
    #  testY: Output variables, records for testing
    #  modelRegression:
    """
    maskedError = sdcRegressionError(xp, y, testX, testY, modelRegression)
    return maskedError[0] - oError


##
import sklearn.linear_model

def testSdcIL_ml(x, y, testX, testY, modelRegression, masking, loParameters):
    """
    Function:
    #   Test IL in terms of prediction error, for a masking method, several parameters
    Parameters:
    # x:             Training data set, input variable
    # y:             Training data set, output variable
    # testX:         Test data set, input variable
    # testY:         Test data set, output variable
    # masking (x,k): function that masks with parameter k
    # loParameters:  list of parameters for masking
    """
    originalError = sdcRegressionError(x, y, testX, testY, modelRegression)
    SDCerror = []
    for k in loParameters:
        SDCerror.append(
            sdcIL_mlRE_GivenOErr(
                originalError[0],
                masking(x, k),
                y,
                testX,
                testY,
                sklearn.linear_model.LinearRegression,
            )
        )
        print(k)
    return SDCerror


# EXAMPLES:
#   from sklearn.kernel_ridge import KernelRidge
#   lp_mdav = [2,3,4,5,10,15,20,25,30,40,41,42,43,44,45,46,50,60,70,80,90,100]
#   DXErrorMDAV_LR = testSdcIL_mlRE (trDX, trDY, teDX, teDY,sklearn.linear_model.LinearRegression, mdav, lp_mdav)
#   DXErrorMDAV_SG = testSdcIL_mlRE (trDX, trDY, teDX, teDY,sklearn.linear_model.SGDRegressor, mdav, lp_mdav)
#   DXErrorMDAV_KR = testSdcIL_mlRE (trDX, trDY, teDX, teDY,sklearn.kernel_ridge.KernelRidge, mdav, lp_mdav)
#   DXErrorMDAV_SV = testSdcIL_mlRE (trDX, trDY, teDX, teDY,sklearn.svm.SVR, mdav, lp_mdav)


## DR: Disclosure Risk ------------------------------------------


#   sdcRecordLinkage (mdav(fRows, 3), fRows)
# def sdcRecordLinkage(df_anon: pd.DataFrame, df_orig: pd.DataFrame, QIs: list):
#     # x = df_anon[QI].to_list()
#     # xp = df_orig[QI].to_list()
#     # x = [[i] for i in df_anon[QI]]
#     x = df_anon[QIs].to_numpy()
#     xp = df_orig[QIs].to_numpy()
#     match = 0
#     for i in range(0, len(x)):
#         iMin = 0
#         dMin = fNorm2(x[i], xp[iMin])
#         for j in range(1, len(xp)):
#             d = fNorm2(x[i], xp[j])
#             if d < dMin:
#                 dMin = d
#                 iMin = j
#         if iMin == i:
#             match = match + 1
#     return (match, len(x))


def sdcRecordLinkage(df_anon: pd.DataFrame, df_orig: pd.DataFrame, QIs: list):
    x = df_anon[QIs].to_numpy()
    xp = df_orig[QIs].to_numpy()

    match = 0

    for i_x, r1 in enumerate(x):
        # Given a record r1 in x, we compute its distance to each record r2 in xp
        sum_squares = np.sum((r1 - xp) ** 2, axis=1)

        # Then, we select the most similar to r1
        i_xp = np.argmin(sum_squares)

        if (i_x == i_xp):
            match += 1

    return (match, len(x))


## Function:
##   (1) Compute an assignment of x to xp,
##   (2) the number of correct matches (assumed x and xp aligned)
##   (3) length of x
## Example:
##   f1080_fRows
##   f1080p = noiseAddition (f1080_fRows, 2)
##   sdcRecordLinkageLoL(f1080_fRows[1:10],f1080p[1:10], [1],[1])
##   iris_fRows
##   irisp = noiseAddition(iris_fRows,1)
##   sdcRecordLinkageLoL(iris_fRows[1:10],irisp[1:10], [1],[1])
##   sdcRecordLinkageLoL(iris_fRows[1:10],irisp[1:10], [1,2],[1,2])
def sdcRecordLinkageLoL(x, xp, varsX, varsXp):
    lol = []
    match = 0
    for i in range(0, len(x)):
        iMin = 0
        dMin = fNorm2(listSelection(x[i], varsX), listSelection(xp[iMin], varsXp))
        for j in range(1, len(xp)):
            d = fNorm2(listSelection(x[i], varsX), listSelection(xp[j], varsXp))
            if d < dMin:
                dMin = d
                iMin = j
        if iMin == i:
            match = match + 1
        lol.append(iMin)
    return (lol, match, len(x))


## DXRL_MDAV = testSdcRL (trDX, mdav, lp_mdav)


## fheader, frows = readNumericalFileWithHeader("Concrete_DataVarsV1V7.csv")
##
## resRL = [(sdcRecordLinkage (mdav(fRows, k), fRows)[0])/1030.0 for k in range(1,5)]
## resIL = [sdcIL_stats(mdav(fRows, k), fRows)[1] for k in range(1,5)]
## resRLmondrian = [(sdcRecordLinkage (mondrian(fRows, k), fRows)[0])/1030.0 for k in range(1,5)]
## resILmondrian = [sdcIL_stats(mondrian(fRows, k), fRows)[1] for k in range(1,5)]

def mondrian(df: pd.DataFrame, QIs: list, k: int) -> pd.DataFrame:
    new_df = df.copy()
    db = new_df[QIs].to_numpy()
    assignedCl = mondrianCl(db, k)
    values = set(assignedCl)
    centroids = []

    max_value = db.max(axis=0)
    min_value = db.min(axis=0)
    ncp = 0

    for cl in values:
        rowsCl = selectFv(db, assignedCl, lambda x: x == cl)
        mat = np.array(rowsCl)

        max_row = mat.max(axis=0)
        min_row = mat.min(axis=0)

        # print(mat, mat.shape)
        # print(max_row)
        # print(min_row)


        ncp += sum(((max_row - min_row) * mat.shape[0]) / (max_value - min_value))

        meanRow = matrixColumnMeans(mat)
        centroids.append(meanRow)

    newDb = [centroids[assignedCl[i]] for i in range(0, len(db))]
    new_df[QIs] = newDb

    n_registers = new_df.shape[0]
    n_attributes = new_df.shape[1]
    ncp = ncp / (n_registers * n_attributes)
    print(f'ncp={ncp}')

    return new_df


def mondrianCl(db, k):
    clId, assignedCl = mondrianWithUnassigned(db, k, [-1] * len(db), 0, -1)
    return assignedCl


def mondrianWithUnassigned(db, k, assignedCl=None, firstNCl=0, stillToProcess=-1):
    if assignedCl == None:
        assignedCl = [-1] * len(db)
    # print(assignedCl)
    unassigned = selectFv(db, assignedCl, lambda x: x == stillToProcess)
    if len(unassigned) < 2 * k:
        index2Update = list(
            filter(lambda i: assignedCl[i] == stillToProcess, range(0, len(assignedCl)))
        )
        newClNumber = firstNCl
        newAssignedCl = updateCl(index2Update, newClNumber, assignedCl)
        return (firstNCl + 1, newAssignedCl)
    else:
        loMaxMinusMin = matrixColumnFunction(unassigned, lambda x: (max(x) - min(x)))
        selectedAttribute = loMaxMinusMin.index(max(loMaxMinusMin))
        cutPoint = partitionAttributeValue(unassigned, selectedAttribute)
        iLow, iHigh = indexLowMidHighValues(
            db, assignedCl, stillToProcess, cutPoint, selectedAttribute
        )
        clLowValues = stillToProcess * 2
        newAssignedCl = updateCl(iLow, clLowValues, assignedCl)
        new1FirstNCl, new1AssignedCl = mondrianWithUnassigned(
            db, k, newAssignedCl, firstNCl, clLowValues
        )
        clHighValues = stillToProcess * 2 + 1
        allAssignedCl = updateCl(iHigh, clHighValues, newAssignedCl)
        new2FirstNCl, new2AssignedCl = mondrianWithUnassigned(
            db, k, new1AssignedCl, new1FirstNCl, clHighValues
        )
        return (new2FirstNCl, new2AssignedCl)


## mondrianWithUnassigned([[100],[100],[100],[200],[300],[400],[500],[600],[650],[600],[700],[700]],3, [-1]*12)


def partitionAttributeValue(unassigned, selectedAttribute):
    allValues = list(map(lambda x: x[selectedAttribute], unassigned))
    allValues.sort()
    midValue = allValues[len(allValues) // 2]
    # midValue = (allValues[0]+allValues[len(allValues)-1])/2
    return midValue


def indexLowMidHighValues(db, assignedCl, idCl, midValue, selectedAttribute):
    indexLowValues = list(
        filter(
            lambda i: assignedCl[i] == idCl and db[i][selectedAttribute] < midValue,
            range(0, len(db)),
        )
    )
    indexMidValues = list(
        filter(
            lambda i: assignedCl[i] == idCl and db[i][selectedAttribute] == midValue,
            range(0, len(db)),
        )
    )
    indexHighValues = list(
        filter(
            lambda i: assignedCl[i] == idCl and db[i][selectedAttribute] > midValue,
            range(0, len(db)),
        )
    )
    numberValues = len(indexLowValues) + len(indexMidValues) + len(indexHighValues)
    toLow = max(numberValues // 2 - len(indexLowValues), 0)
    # print(midValue)
    # print(toLow)
    indexLow = indexLowValues + indexMidValues[0:toLow]
    indexHigh = indexMidValues[toLow:] + indexHighValues
    return (indexLow, indexHigh)


## indexLowMidHighValues([[1],[1],[1],[1],[1],[4]],[-1]*6,-1,1,0)
## indexLowMidHighValues(mat, [-1,-1,-1,-1], -1, 2, 0)
## mondrianWithUnassigned([[100],[100],[100],[200],[300],[400],[500],[600],[650],[600],[700],[700]],3, [-1]*12)

def main():
    df = pd.read_csv("./datasets/wearable-exercise-frailty.csv")
    QIs = ["Age", "Height(cm)", "Weight(kg)"]
    print(df)
    print(mdav(df, QIs, 9))

if __name__ == "__main__":
    main()
