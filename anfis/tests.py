import anfis
import membershipfunction #import membershipfunction, mfDerivs
import numpy as np
import csv

ts = np.loadtxt("trainingSet.txt", usecols=[1,2,3])
X = ts[:,0:2]
print(np.shape(X))
Y = ts[:,2]
print(np.shape(Y))
mf = [[['gaussmf',{'mean':-11.,'sigma':5.}],['gaussmf',{'mean':-8.,'sigma':5.}],['gaussmf',{'mean':-14.,'sigma':20.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
            [['gaussmf',{'mean':-10.,'sigma':20.}],['gaussmf',{'mean':-20.,'sigma':11.}],['gaussmf',{'mean':-9.,'sigma':30.}],['gaussmf',{'mean':-10.5,'sigma':5.}]]]
mfc = membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X, Y, mfc)
anf.trainHybridJangOffLine(epochs=10)
# print("anf")
# print(round(anf.consequents[-1][0],6))
# print(round(anf.consequents[-2][0],6))
# print(round(anf.fittedValues[9][0],6))
if round(anf.consequents[-1][0],6) == -5.275538 and round(anf.consequentsz[-2][0],6) == -1.990703 and round(anf.fittedValues[9][0],6) == 0.002249:
	print ('test is good')
anf.plotErrors()
anf.plotResults()