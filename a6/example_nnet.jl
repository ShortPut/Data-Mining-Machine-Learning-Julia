using Printf

# Load X and y variable
using JLD
using Plots
data = load("basisData.jld")
(X, y) = (data["X"], data["y"])
XPlot = X
(n, d) = size(X)

# # Choose network structure and randomly initialize weights
# include("NeuralNet.jl")
# nHidden = [3]
# nParams = NeuralNet_nParams(d, nHidden)
# w = randn(nParams, 1)

# # Train with stochastic gradient
# maxIter = 10000
# stepSize = 1e-4
# for t in 1:maxIter

#     # The stochastic gradient update:
#     i = rand(1:n)
#     (f, g) = NeuralNet_backprop(w, X[i, :], y[i], nHidden)
#     global w = w - stepSize * g

#     # Every few iterations, plot the data/model:
#     if (mod(t - 1, round(maxIter / 50)) == 0)
#         @printf("Training iteration = %d\n", t - 1)
#         xVals = -10:0.05:10
#         Xhat = zeros(length(xVals), 1)
#         Xhat[:] .= xVals
#         yhat = NeuralNet_predict(w, Xhat, nHidden)
#         scatter(X, y, legend=false, linestyle=:dot)
#         plot!(Xhat, yhat, legend=false)
#         gui()
#         sleep(0.1)
#     end
# end
# plot!()

##### Improved Version #####
# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
X = [ones(n, 1) X]
(n, d) = size(X)
nHidden = [15]
nParams = NeuralNet_nParams(d, nHidden)
w = randn(nParams, 1)

# Train with stochastic gradient
maxIter = 100000
stepSize = 1e-5
lambda = 0.1    # Added lambda for regularization
for t in 1:maxIter

    # The stochastic gradient update:
    i = rand(1:n)
    (f, g) = NeuralNet_backprop(w, X[i, :], y[i], nHidden)
    f = f + ((lambda / 2) * sum(w .^ 2))
    g = g + (lambda * w)
    global w = w - stepSize * (g + ((lambda / n) * w))

    # Every few iterations, plot the data/model:
    if (mod(t - 1, round(maxIter / 50)) == 0)
        @printf("Training iteration = %d\n", t - 1)
        xVals = -10:0.05:10
        Xhat = zeros(length(xVals), 1)
        Xhat[:] .= xVals
        XhatPlot = Xhat
        l = length(Xhat)
        oneCol = ones(l, 1)
        Xhat = hcat(oneCol, Xhat)
        yhat = NeuralNet_predict(w, Xhat, nHidden)
        scatter(XPlot, y, legend=false, linestyle=:dot)
        plot!(XhatPlot, yhat, legend=false)
        gui()
        sleep(0.1)
    end
end
plot!()

