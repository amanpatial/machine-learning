#logistic regression model import
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#reading training set file
print("Reading trained sale price from train_data.csv...")
tr=open("train_data.csv","r")
records=tr.readlines()
tr.close()

#Making training set X and y vectors
X=[[] for i in range(1460)]
y=[]
for i in range(1,len(records)):
    for j in range(len(records[i].strip().split(","))-1):
        X[i - 1].append(int(records[i].strip().split(",")[j]))
    y.append(int(records[i].strip().split(",")[36]))
print("Finished reading trained sale price")

# Prints the data provided
#print("Printing X axis: all independent feature values from left to right...")
#print(X)
# Prints the data provided
#print("Printing Y axis: dependent price value on independent values from top to bottom...")
#print(y)

#training our logistic regression model
lr = LogisticRegression()
lr.fit(X,y)

#reading testing set file without SalePrice 
print("Reading test data without sale price from test_data.csv...")
te=open("test_data.csv","r")
records1=te.readlines()
te.close()
print("Finished reading test data without sale price")

#Making testing set X vector
XX=[[] for i in range(1459)]
yy=[]
for i in range(1,len(records1)):
    for j in range(len(records1[i].strip().split(","))):
        XX[i - 1].append(int(records1[i].strip().split(",")[j]))

yy = lr.predict(XX)

# writing predicted house price to new file
print("Writing predicted house price to result.csv...")
result=open("result.csv","w")
result.write("House No,Predicted Price" + "\n")
for i in range(len(yy)):
    result.write(str(i+1) + "," + str(yy[i]) + "\n")
result.close()

#print("Printing yy : all predicting Y price values for all given x test values ...")
#print(yy)

#Checking for model accuracy by applying model on training set
yyy = lr.predict(X)
print(yyy)
print("Finished writing predicted house price.")
print ("Model accuracy: {}%".format(accuracy_score(y,yyy)*100))



