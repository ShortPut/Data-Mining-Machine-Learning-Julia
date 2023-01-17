using Printf
using Plots

# Load X and y variable
using JLD
X = load("citiesSmall.jld", "X")
y = load("citiesSmall.jld", "y")
print(y)
n = size(X, 1)

# Maximum depth we will plot
maxDepth = 15

# Evaluate training error
include("decisionTree_infoGain.jl")

for depth in 1:maxDepth
    model = decisionTree_infoGain(X, y, depth)
    yhat = model.predict(X)
    trainError = sum(yhat .!= y) / n
    @printf("Training error with depth-%d infogain-based decision tree: %.2f\n", depth, trainError)
end

# Evaluate the test error
Xtest = load("citiesSmall.jld", "Xtest")
Xtest = Xtest[401, end]
ytest = load("citiesSmall.jld", "ytest")
yests = ytest[401:end]
t = size(Xtest, 1)

for depth in 1:maxDepth
    model = decisionTree_infoGain(Xtest, ytest, depth)
    yhat = model.predict(Xtest)
    testError = sum(yhat .!= ytest) / t
    @printf("Test error with depth-%d decision tree: %.3f\n", depth, testError)
end

scatter(1:15, [0.33, 0.24, 0.15, 0.12, 0.08, 0.05, 0.02, 0.01, 0, 0, 0, 0, 0, 0, 0], markersize=3, markercolor=:red, label="Training Error")
scatter!(1:15, [0.333, 0.286, 0.177, 0.183, 0.127, 0.119, 0.103, 0.089, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095], markersize=3, markercolor=:green, label="Test Error")