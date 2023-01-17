using Printf
using Statistics
include("misc.jl")
include("randomTree.jl")

function randomForest(X, y, depth, nTrees)

    subModels = Array{GenericModel}(undef, nTrees)
    for i in 1:nTrees
        subModels[i] = randomTree(X, y, depth)
    end

    # Make a predict function
    function predict(Xhat)
        (t, d) = size(Xhat)
        a = length(subModels)
        yhat = zeros(t)
        predictions = zeros(t, nTrees)

        for i in 1:a
            pred = subModels[i].predict(Xhat)
            predictions[:, i] = pred
        end

        for i in 1:t
            yhat[i] = mode(predictions[i, :])
        end
        return yhat
    end
    return GenericModel(predict)
end