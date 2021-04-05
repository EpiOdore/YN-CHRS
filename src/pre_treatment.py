import random
import math

def aleaGauss(sigma):
    U1 = random.random()
    U2 = random.random()
    return sigma*math.sqrt(-2*math.log(U1))*math.cos(2*math.pi*U2)

def addBruitGaussien(dictionnary):
    for _, value in dictionnary.items():
        for i in range(len(value[0])):
            for j in range(len(value[0][i])):
                value[0][i][j] = value[0][i][j] + aleaGauss(0.025)
    return dictionnary
