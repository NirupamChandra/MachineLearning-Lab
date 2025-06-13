import time as t

paths = ['1_MeanMode.py', '4_linearReg.py', '5_multiLinReg.py', '6_DecsnTreeReg.py', '7_Knn.py', '8_LogisticReg.py', '9_KMeansCluster.py']

def readExtract(path:str):

    data = ''
    with open(path, 'r+') as file:
        data = file.read()
        data = data + '\n\n'
    
    title = addTitle(path)
    return title + data

def addTitle(path:str):

    splitted = path.split('.') #splits into ['1_MeanMode', 'py]
    return '#'+ splitted[0]+ '\n'

def writeContent():

    start = t.time()
    with open ('inOneFile.txt', 'w') as files:
        content = ''

        for path in paths:
            content += readExtract(path) + '\n\n'

        files.write(content)
    end = t.time()
    return end - start

print(f'File write completed in {writeContent():.3f} seconds')

