using Printf

# Load X and y variable
using JLD
A = load("citiesSmall.jld", "X")
X = zeros(200, 2)
for i in 1:200
    for j in 1:2
        X[i, j] = A[i, j]
    end
end
y = load("citiesSmall.jld", "y")
y = y[1:200]

n = size(X, 1)

# Maximum depth we will plot
maxDepth = 15

# Training Error
include("decisionTree_infoGain.jl")
for depth in 1:maxDepth
    model = decisionTree_infoGain(X, y, depth)
    yhat = model.predict(X)
    trainError = sum(yhat .!= y) / n
    @printf("Training error with depth-%d infogain-based decision tree: %.2f\n", depth, trainError)
end

# Validation Error
B = load("citiesSmall.jld", "X")
X = zeros(200, 2)
for i in 201:400
    for j in 1:2
        X[i-200, j] = B[i, j]
    end
end
y = load("citiesSmall.jld", "y")
y = y[201:400]
t = size(X, 1)

for depth in 1:maxDepth
    model = decisionTree_infoGain(X, y, depth)
    yhat = model.predict(X)
    testError = sum(yhat .!= y) / t
    @printf("VALIDATION error with depth-%d infogain-based decision tree: %.2f\n", depth, testError)
end

