using Printf
using Statistics
include("misc.jl")

# Load X and y variable
using JLD
data = load("basisData.jld")
(X, y, Xtest, ytest) = (data["X"], data["y"], data["Xtest"], data["ytest"])

# # Fit least squares with bias via gradient descent
# n = size(X, 1)
# Z = [ones(n, 1) X]
# lambda = 1
# v = zeros(2, 1)
# alpha = 1 / (norm(Z, 2)^2 + lambda)
# for t in 1:500
#     global v -= alpha * (Z' * (Z * v - y) + lambda * v)
#     @show (1 / 2)norm(Z * v - y)^2 + lambda * norm(v)^2
# end
# predict(Xhat) = [ones(size(Xhat, 1), 1) Xhat] * v
# model = LinearModel(predict, v)

# # Evaluate training error
# yhat = model.predict(X)
# trainError = mean((yhat - y) .^ 2)
# @printf("Squared train Error with least squares: %.3f\n", trainError)

# # Evaluate test error
# yhat = model.predict(Xtest)
# testError = mean((yhat - ytest) .^ 2)
# @printf("Squared test Error with least squares: %.3f\n", testError)

# # Plot model
# using Plots
# scatter(X, y, legend=false, linestyle=:dot)
# scatter!(Xtest, ytest, legend=false, linestyle=:dot)
# Xhat = minimum(X):0.1:maximum(X)
# Xhat = reshape(Xhat, length(Xhat), 1) # Make into an n by 1 matrix
# yhat = model.predict(Xhat)
# plot!(Xhat, yhat, legend=false)


# # Fit least squares with bias via STOCHASTIC gradient descent
# n = size(X, 1)
# Z = [ones(n, 1) X]
# lambda = 1
# v = zeros(2, 1)
# alpha = 1 / (norm(Z, 2)^2 + lambda)
# for t in 1:500
#     i = rand(1:n)
#     # println("Z", Z[i, :, :])
#     # println("Zt", Z[i, :]')
#     # println("v", v)
#     # println("y", y[i, :, :])

#     # println("Z size", size(Z[i, :, :]))
#     # println("Ztsize", size(Z[i, :, :]'))
#     # println("ysize", size(y[i, :, :]))
#     # println("vsize", size(v))
#     # println(Z[i, :] * (Z[i, :, :]' * v - y[i, :, :]))

#     # global v -= alpha * ((Z[i, :] * ((Z[i, :, :]' * v) - y[i, :, :])) + (lambda / n) * v)
#     @show (1 / 2)norm(Z * v - y)^2 + lambda * norm(v)^2
# end
# predict(Xhat) = [ones(size(Xhat, 1), 1) Xhat] * v
# model = LinearModel(predict, v)

# # Evaluate training error
# yhat = model.predict(X)
# trainError = mean((yhat - y) .^ 2)
# @printf("Squared train Error with least squares: %.3f\n", trainError)

# # Evaluate test error
# yhat = model.predict(Xtest)
# testError = mean((yhat - ytest) .^ 2)
# @printf("Squared test Error with least squares: %.3f\n", testError)

# # Plot model
# using Plots
# scatter(X, y, legend=false, linestyle=:dot)
# scatter!(Xtest, ytest, legend=false, linestyle=:dot)
# Xhat = minimum(X):0.1:maximum(X)
# Xhat = reshape(Xhat, length(Xhat), 1) # Make into an n by 1 matrix
# yhat = model.predict(Xhat)
# plot!(Xhat, yhat, legend=false)


# Fit least squares with bias via STOCHASTIC gradient descent MODIFIED STEPS
n = size(X, 1)
Z = [ones(n, 1) X]
lambda = 1
v = zeros(2, 1)
alpha = 1 / (norm(Z, 2)^2 + lambda)
for t in 1:500
    gamma = 0.1
    alpha = gamma
    i = rand(1:n)
    global v -= alpha * ((Z[i, :] * ((Z[i, :, :]' * v) - y[i, :, :])) + (lambda / n) * v)
    # @show (1 / 2)norm(Z * v - y)^2 + lambda * norm(v)^2
end
predict(Xhat) = [ones(size(Xhat, 1), 1) Xhat] * v
model = LinearModel(predict, v)

# Evaluate training error
yhat = model.predict(X)
trainError = mean((yhat - y) .^ 2)
@printf("Squared train Error with least squares: %.3f\n", trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest) .^ 2)
@printf("Squared test Error with least squares: %.3f\n", testError)

# Plot model
using Plots
scatter(X, y, legend=false, linestyle=:dot)
scatter!(Xtest, ytest, legend=false, linestyle=:dot)
Xhat = minimum(X):0.1:maximum(X)
Xhat = reshape(Xhat, length(Xhat), 1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot!(Xhat, yhat, legend=false)