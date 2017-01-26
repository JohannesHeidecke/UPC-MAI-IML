from scipy.io import arff
import numpy as np
import time
from sklearn.feature_selection import chi2
from skfeature.function.similarity_based import reliefF

def getKeyMinMaxDict(data, meta):
    keyMinMaxDict = {}
    # loop over all features:
    for key in meta:
        # find all numeric features:
        if meta[key][0] == 'numeric':
            # estimate min and max:
            min = np.min(data[key])
            max = np.max(data[key])
            # persist min and max:
            minMaxDict = {}
            minMaxDict['min'] = min
            minMaxDict['max'] = max
            keyMinMaxDict[key] = minMaxDict
    return keyMinMaxDict

def getKeyCategoriesDict(data, meta):
    keyCategoriesDict = {}
    # loop over all features:
    for key in meta:
        # find all nominal features:
        if meta[key][0] == 'nominal':
            keyCategoriesDict[key] = {}
            counter = 1
            for category in meta[key][1]:
                # assign a numeric value to each category
                # counter starts at 1 since 0 is reserved for missing values
                keyCategoriesDict[key][category] = counter
                counter = counter + 1
    return keyCategoriesDict

def normalizeNumericFeature(col, min, max):
    col = np.subtract(col, min)
    col = np.divide(col, (max-min))
    return col

def normalizeNominalFeature(col, categories):
    oneHotCodes = []
    oneHotLength = len(categories) + 1
    for value in col:
        category = str(value)
        categoryToNumeric = 0
        if category in categories:
            # if category exists in dictionary: obtain category number
            categoryToNumeric = categories[category]
        # construct one hot code with 1 at the correct position:
        oneHot = ['0'] * oneHotLength
        oneHot[categoryToNumeric] = '1'
        oneHot = ''.join(oneHot)
        oneHotCodes.append(oneHot)
    oneHotCodes = np.array(oneHotCodes)
    return oneHotCodes

def getNormalizedData(data, meta, keyMinMaxDict, keyCategoriesDict):
    features = []
    classes = []
    for colKey in meta:
        if meta[colKey][0] == 'numeric':
            min = keyMinMaxDict[colKey]['min']
            max = keyMinMaxDict[colKey]['max']
            col = normalizeNumericFeature(data[colKey], min, max)
            features.append(col)
        else:
            if colKey != meta.names()[-1]:
                col = normalizeNominalFeature(data[colKey], keyCategoriesDict[colKey])
                nomMat = []
                for value in col:
                    row = []
                    for c in str(value):
                        if int(c) == 0:
                            row.append(0)
                        else:
                            row.append(np.divide(1, np.sqrt(2)))
                    nomMat.append(row)
                nomMat = np.array(nomMat)
                features.append(nomMat)
            else:
                classes = data[colKey]
    resFeatures = []
    for el in features:
        if el.ndim == 2:
            for col in el.T:
                resFeatures.append(col.T)
        else:
            resFeatures.append(el)
    features = np.array(resFeatures).T
    classes = np.array(classes)
    
    return features, classes

def dataToCaseAndTestBase(dataset, s):
    f = 'datasetsCBR/' + dataset + '/' + dataset
    f += '.fold.{:0>6d}'.format(s)
    fTrain = f + '.train.arff'
    fTest = f + '.test.arff'
    dataTrain, metaTrain = arff.loadarff(fTrain)
    dataTest, metaTest = arff.loadarff(fTest)
    keyMinMaxDict = getKeyMinMaxDict(dataTrain, metaTrain)
    keyCategoriesDict = getKeyCategoriesDict(dataTrain, metaTrain)
    CBproblems, CBsolutions = getNormalizedData(dataTrain, metaTrain, keyMinMaxDict, keyCategoriesDict)
    TCproblems, TCsolutions = getNormalizedData(dataTest, metaTest, keyMinMaxDict, keyCategoriesDict)    
    return (CBproblems, CBsolutions, TCproblems, TCsolutions)

def getDistance(p1, p2):
    return np.sum(np.square(np.abs(np.subtract(p1, p2))))

def getWeightedDistance(p1, p2, weights):
    p1 = np.multiply(p1, weights)
    p2 = np.multiply(p2, weights)
    return np.sum(np.square(np.abs(np.subtract(p1, p2))))

def getKnn(CBproblems, problem, k, weights=None):
    knnIndices = [0] * k
    knnDistances = [float("inf")] * k
    weighted = (type(weights) != type(None))
    for otherProblemIndex in range(0, len(CBproblems)):
        otherProblem = CBproblems[otherProblemIndex]
        if weighted:
            distance = getWeightedDistance(problem, otherProblem, weights)
        else:
            distance = getDistance(problem, otherProblem)
        if distance < knnDistances[-1]:
            # find correct location in sortedList of knnIndices:
            tempIndices = []
            tempDistances = []
            for i in range(0,k):
                if knnDistances[i] > distance:
                    # insert newly found point at correct location
                    tempIndices.append(otherProblemIndex)
                    tempDistances.append(distance)
                    for j in range (i,k):
                        tempIndices.append(knnIndices[j])
                        tempDistances.append(knnDistances[j])
                else:
                    tempIndices.append(knnIndices[i])
                    tempDistances.append(knnDistances[i])
            # cut off the result to correct length again
            knnIndices = tempIndices[:k]
            knnDistances = tempDistances[:k]
            
    return knnIndices, knnDistances

def getFeatureWeights(CBproblems, CBsolutions, metric='chi_square'):
    y = CBsolutions
    if metric == 'chi_square':
        weights = chi2(CBproblems, y)
    if metric == 'reliefF':
        weights = reliefF.reliefF(CBproblems, y)
    weights = np.nan_to_num(weights)
    min = np.min(weights)
    max = np.max(weights)
    weights = np.subtract(weights, min)
    weights = np.divide(weights, (max-min))
    return weights

def acbrAlgorithm(dataset, fold, k=5, alpha=0.2, featureSelect='chi_square',  retention='AR', useOblivion=True):
    (CBproblems, CBsolutions, TCproblems, TCsolutions) = dataToCaseAndTestBase(dataset, fold)

    weights = None
    if featureSelect != None:
        weights = getFeatureWeights(CBproblems, CBsolutions, metric=featureSelect)

    confusionMatrix = [0,0]
    # print('Initial case base size: ' + str(CBproblems.shape))
    goodnesses = [0.5] * len(CBproblems)
    goodnesses = np.array(goodnesses)
    CM = [goodnesses, goodnesses]
    for j in range(0, len(TCproblems)):
        # Get the new problem
        cNew = TCproblems[j]
        
        # K phase
        K = acbrRetrievalPhase(CBproblems,CBsolutions, cNew, k, featureSelect, weights)
        
        # Reuse phase
        cSol = acbrReusePhase(cNew, K, CBsolutions)
        
        # Revision phase
        confusionMatrix = acbrRevisionPhase(cSol, TCsolutions[j], confusionMatrix)
        
        # Review phase
        newCBproblems, newCBsolutions, newCM = acbrReviewPhase(CBproblems, CBsolutions, K, TCsolutions[j], CM, alpha, useOblivion)
        
        #Retention phase
        CBproblems, CBsolutions, CM = acbrRetentionPhase(cNew, cSol, K, CBproblems, CBsolutions, newCBproblems, newCBsolutions, CM, newCM, policy=retention)

    accuracy = float(confusionMatrix[0]) / (confusionMatrix[0] + confusionMatrix[1])
    return CM, CBproblems, CBsolutions, accuracy

def acbrRetrievalPhase(CBproblems, CBsolutions, cNew, k, featureSelect, weights):
    return getKnn(CBproblems, cNew, k, weights)[0]

def acbrReusePhase(cNew, K, CBsolutions):
    return CBsolutions[K[0]]

def acbrRevisionPhase(cSol, correctSol, confusionMatrix):
    if cSol == correctSol:
        confusionMatrix[0] += 1
    else:
        confusionMatrix[1] += 1
    return confusionMatrix

def acbrReviewPhase(CBproblems, CBsolutions, K, cNewClass, CM, alpha, useOblivion):
    lastGoodnesses = CM[-1]
    newGoodnesses = []
    for goodness in lastGoodnesses:
        newGoodnesses.append(goodness)
    for k in K:
        cKClass = CBsolutions[k]
        if cKClass == cNewClass:
            r = 0
        else:
            r = 1
        g = lastGoodnesses[k]
        newGoodnesses[k] = g + alpha * (r - g)
    newGoodnesses = np.array(newGoodnesses)
    if useOblivion:
        CBproblems, CBsolutions, newGoodnesses, newCM = oblivionByGoodnessFS(K, CM, CBproblems, CBsolutions, newGoodnesses)
    else:
        newCM = []
        newCM.append(CM[0])
    newCM.append(newGoodnesses)
    newCM = [CM[0], newGoodnesses]
    return CBproblems, CBsolutions, newCM
    
def oblivionByGoodnessFS(K, CM, CBproblems, CBsolutions, newGoodnesses):
    firstGoodnesses = CM[0]
    deleteRows = []
    for k in K:
        if newGoodnesses[k] < firstGoodnesses[k]:
            deleteRows.append(k)
    CBproblems = np.delete(CBproblems, deleteRows, axis=0)
    CBsolutions = np.delete(CBsolutions, deleteRows, axis=0)
    newGoodnesses = np.delete(newGoodnesses, deleteRows, axis=0)
    newCM = []
    newCM.append(CM[0])
    newCM[0] = np.delete(CM[0], deleteRows, axis=0)
    
    return CBproblems, CBsolutions, newGoodnesses, newCM


# # # # # # # # # # # # # # #

def acbrRetentionPhase(cNew, cSol, K, CBproblems, CBsolutions, newCBproblems, newCBsolutions, CM, newCM, policy="NR"):
    if policy == "NR":
        result =  acbrNoRetentionPhase(cNew, cSol, K, CBproblems, CBsolutions, CM)
    elif policy == "AR":
        result =  acbrAlwaysRetentionPhase(cNew, cSol, K, CBproblems, CBsolutions, CM)
    elif policy == "DD":
        result =  acbrDDRetentionPhase(cNew, cSol, K, CBproblems, CBsolutions, CM)
    elif policy == "MG":
        result =  acbrMGRetentionPhase(cNew, cSol, K, CBproblems, CBsolutions, CM)
    if type(result) != type(None):
        newCBproblems = np.vstack([newCBproblems,result[0]])
        newCBsolutions = np.append(newCBsolutions,result[1])
        newCM[0] = np.append(newCM[0], result[2])
        newCM[1] = np.append(newCM[1], result[2])
    return newCBproblems, newCBsolutions, newCM




def acbrNoRetentionPhase(cNew, cSol, K, CBproblems, CBsolutions, CM):
    return None

def acbrAlwaysRetentionPhase(cNew, cSol, K, CBproblems, CBsolutions, CM):
    return (cNew, cSol, 0.5)

def acbrDDRetentionPhase(cNew, cSol, K, CBproblems, CBsolutions, CM):
    # count classes in CBj
    C = len(np.unique(CBsolutions))
    (values,counts) = np.unique(CBsolutions[K],return_counts=True)
    maxcount= np.argmax(counts)
    
    MK_module = counts[maxcount]
    MC=values[maxcount]
    
    MK=CBproblems[np.array(K)[CBsolutions[K] == MC]]

    normCK = np.max((2,len(values)))
    disagreement = float(len(K) - MK_module)/((normCK-1)*MK_module)
    if disagreement >= 0.3: # Threshold in this case is 0.3
        goodness = (CM[1][np.array(K)[CBsolutions[K] == MC]]).max()
        return (cNew, cSol, goodness)
    return None

def acbrMGRetentionPhase(cNew, cSol, K, CBproblems, CBsolutions, CM):
    (values,counts) = np.unique(CBsolutions[K],return_counts=True)
    maxcount= np.argmax(counts)
    
    MK_module = counts[maxcount]
    MC=values[maxcount]
    
    MK=CBproblems[np.array(K)[CBsolutions[K] == MC]]
    goodness = (CM[1][np.array(K)[CBsolutions[K] == MC]]).max()
    G_MC = (CM[1][[CBsolutions == MC]])
    g_max = G_MC.max()
    g_min = G_MC.min();
    
    threshold = (g_max+g_min)/2
    
    if goodness <= threshold:
        return (cNew, cSol, goodness)
    return None

# # # # # # # # # # # # # # #

def crossValidation(dataset, folds, k=5, featureSelect=None, retention='AR', useOblivion=True):
    accuracies = []
    efficiencies = []
    finalCaseBaseSizes = []
    for s in range(0, folds):
        # print('Fold ' + str(s+1) + '...')
        start = time.time()
        CM, CBproblems, CBsolutions, accuracy = acbrAlgorithm(dataset, s, k=k, featureSelect=featureSelect, retention=retention, useOblivion=useOblivion)
        end = time.time()
        # print('acbrAlgorithm terminated after ' +str(end-start) + ' seconds')
        # print('New case base size: ' + str(CBproblems.shape))
        # print('accuracy: ' + str(accuracy))
        # print('- ' * 30)
        accuracies.append(accuracy)
        efficiencies.append(end-start)
        finalCaseBaseSizes.append(len(CBproblems))
    return accuracies, efficiencies, finalCaseBaseSizes

# # # # # # # # # # # # # # # start executing here:      

dataset = 'pen-based' # vowel audiology
KS = [3, 5, 7]
RS = ['NR', 'DD', 'MG', 'AR'] 
OS = [True, False]
f = None # 'reliefF' 'chi_squared'

print(dataset)
print('k\tf\tr\to\taccuracy\tefficiency\tCB size')

for k in KS:
    for o in OS:
        for r in RS:
            accuracies, efficiencies, finalCaseBaseSizes = crossValidation(dataset, 10, k=k, featureSelect=f, retention=r, useOblivion=o)
            s = ''
            s += str(k) + '\t'
            s += str(f) + '\t'
            s += str(r) + '\t'
            s += str(o) + '\t'
            s += str(np.mean(accuracies)) + '\t'
            s += str(np.mean(efficiencies)) + '\t'
            s += str(np.mean(finalCaseBaseSizes))
            print(s)
