using Images, Plots
include("kMeans.jl")

function quantizeImage(imageName, b)
    I = load(imageName)
    (nRows, nCols) = size(I)

    R = permutedims(channelview(I), [2, 3, 1])
    X = reshape(float64.(R), (nRows * nCols, 3))

    cluster = kMeans(X, 2^b)
    W = cluster.W
    yhat = cluster.predict(X)

    deQuantizeImage(yhat, W, nRows, nCols)

    return yhat, W, nRows, nCols

end

function deQuantizeImage(yhat, W, nRows, nCols)
    meanColors = zeros(nRows * nCols, 3)
    for i in 1:nRows*nCols
        for j in 1:3
            meanColors[i, j] = W[yhat[i], j]
        end
    end

    R2 = reshape(float64.(meanColors), (nRows, nCols, 3))
    I2 = colorview(RGB, permutedims(R2, [3, 1, 2]))

    save("qDog6.png", I2)
    #=     P = load("qDog.png")
        plot(P)
        gui() =#
end

quantizeImage("dog.png", 6)