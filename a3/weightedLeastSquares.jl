include("misc.jl")

function weightedLeastSquares(X, y, v)
    V = Diagonal(v)
    w = ((X'V) * X) \ ((X'V) * y)

    predict(Xhat) = Xhat * w

    return GenericModel(predict)
end