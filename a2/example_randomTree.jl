using Printf
using Statistics

# Load data
using JLD
fileName = "vowel.jld"
X = load(fileName, "X")
y = load(fileName, "y")
Xtest = load(fileName, "Xtest")
ytest = load(fileName, "ytest")

# Fit a decision tree classifier
include("decisionTree_infoGain.jl")
depth = Inf
model = decisionTree_infoGain(X, y, depth)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d decision tree: %.3f\n", depth, trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d decision tree: %.3f\n", depth, testError)

######################################

# Fit a random tree classifier
include("randomTree.jl")
depth = Inf
model = randomTree(X, y, depth)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d random tree: %.3f\n", depth, trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d random tree: %.3f\n", depth, testError)

#################################

# Fit a random forest classifier
include("randomForest.jl")
depth = Inf
nTrees = 50
model = randomForest(X, y, depth, nTrees)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d random FOREST: %.3f\n", depth, trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d random FOREST: %.3f\n", depth, testError)

