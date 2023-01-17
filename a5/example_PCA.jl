using DelimitedFiles

### HEATMAP
# # Load data
# dataTable = readdlm("animals.csv",',')
# X = float(real(dataTable[2:end,2:end]))
# (n,d) = size(X)

# # Plot matrix as image
# using Plots
# heatmap(X)

### SCATTER PLOT
# Load data
dataTable = readdlm("animals.csv", ',')
include("PCA.jl")
X = float(real(dataTable[2:end, 2:end]))
(n, d) = size(X)
k = 12
model = PCA(X, k)
W = model.W
XW = model.compress(X)
Z = XW * inv(W * W')

# # Plot matrix as image
# using Plots
# scatter(Z[:, 1], Z[:, 2], legend=false)
# for i in 1:n
#     animal = Z[i, :]
#     annotate!(Z[i, :][1], Z[i, :][2], dataTable[2:end, 1][i], annotationfontsize=8)
# end
# savefig("animals.png")


### Compute Explained Variance
(n, d) = size(X)
mu = mean(X, dims=1)
X -= repeat(mu, n, 1)
var = ((norm(Z * W - X))^2) / (norm(X))^2
println(var)