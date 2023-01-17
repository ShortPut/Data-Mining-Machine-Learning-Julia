# Load data
using JLD
X = load("clusterData2.jld", "X")

# K-means clustering
k = 4
include("kMedians.jl")
model = kMedians(X, k, doPlot=false)
y = model.predict(X)
b = [5595, 1627, 958, 906, 887, 882, 870, 871, 868, 868]
include("clustering2Dplot.jl")

#clustering2Dplot(X, y, model.W)
#plot(1:10, b, xlabel="k", ylabel="Error", label="Error")
#savefig("plot.png")

#clustering2Dplot(X, y, model.W)
#savefig("plot2.png")
#c = [87353, 54326, 44324, 35065, 25150, 17693, 9660, 1854, 1626, 1284]
#plot(1:10, c, xlabel="k", ylabel="Error", label="Error")

#d = [4795, 4600, 3827, 1166, 1061, 970, 864, 810, 735, 662]
#plot(1:10, d, xlabel="k", ylabel="Error", label="Error")

clustering2Dplot(X, y, model.W)
savefig("plot3.png")