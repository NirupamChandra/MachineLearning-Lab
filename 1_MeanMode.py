import statistics as st

n = int(input('Enter no.of items : '))

def takeInput():
    print('Enter data : ')
    data = []
    for x in range(n):
        data.append(int(input()))
    return data

def getMean(data):
    
    return sum(x for x in data)/n

def getMode(data):

    return st.mode(data)

def getMedian(data):

    return st.median(data)

def calcTendencyMeasure(data):

    mean = getMean(data)
    mode = getMode(data)
    median = getMedian(data)

    return mean, mode, median

def getVariance(data, mean):

    diffSum = sum((x - mean)**2 for x in data)
    return diffSum / n-1

def calcDispersion(data):

    var = getVariance(data, getMean(data))
    stdDev = var**0.5

    return var, stdDev

data = takeInput()

mean, mode, median = calcTendencyMeasure(data)
var, stdDev = calcDispersion(data)

print(data)

print('Mean : ', mean)
print('Mode : ', mode)
print('Median : ', median)
print('Variance : ', var)
print('Std.Deviation : ', stdDev)
