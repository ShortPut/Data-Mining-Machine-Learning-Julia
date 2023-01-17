using Printf
using Statistics

# Load X and y variable
using JLD
data = load("basisData.jld")
(X, y, Xtest, ytest) = (data["X"], data["y"], data["Xtest"], data["ytest"])

#= # Fit a least squares model
include("leastSquares.jl")
model = leastSquares(X, y)

# Evaluate training error
yhat = model.predict(X)
trainError = mean((yhat - y) .^ 2)
@printf("Squared train Error with least squares: %.3f\n", trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest) .^ 2)
@printf("Squared test Error with least squares: %.3f\n", testError) =#

######################################################

#= # Fit a least squares BIAS model
include("leastSquaresBias.jl")
model = leastSquaresBias(X, y)

# Evaluate training error
yhat = model.predict(X)
trainErrorBias = mean((yhat - y) .^ 2)
@printf("Squared train Error with least squares BIAS: %.3f\n", trainErrorBias)

# Evaluate test error
yhat = model.predict(Xtest)
testErrorBias = mean((yhat - ytest) .^ 2)
@printf("Squared test Error with least squares BIAS: %.3f\n", testErrorBias) =#

######################################################

#= # Fit a least squares BASIS model
p = 10
include("leastSquaresBasis.jl")
model = leastSquaresBasis(X, y, p)

# Evaluate training error
yhat = model.predict(X, p)
trainErrorPoly = mean((yhat - y) .^ 2)
@printf("Squared train Error with least squares POLY, p=%.f: %.3f\n", p, trainErrorPoly)

# Evaluate test error
yhat = model.predict(Xtest, p)
testErrorPoly = mean((yhat - ytest) .^ 2)
@printf("Squared test Error with least squares POLY, p=%.f: %.3f\n", p, testErrorPoly) =#

# Fit a least squares Gaussian RBF model
s = 1
include("leastSquaresGauss.jl")
model = leastSquaresGauss(X, y, s)

# Evaluate training error
yhat = model.predict(X, Xhat, s)
trainErrorGauss = mean((yhat - y[3:200]) .^ 2)
@printf("Squared train Error with least squares Gauss, s=%.f: %.3f\n", s, trainErrorGauss)

# Evaluate test error
yhat = model.predict(X, Xtest, s)
testErrorGauss = mean((yhat - ytest) .^ 2)
@printf("Squared test Error with least squares Gauss, s=%.f: %.3f\n", s, testErrorGauss)

# Plot model
using Plots
scatter(X, y, legend=false, linestyle=:dot)
Xhat = minimum(X):0.1:maximum(X)
yhat = model.predict(X, Xhat, s)
plot!(Xhat, yhat, legend=false)
savefig("leastSqGauss.png")
gui()
