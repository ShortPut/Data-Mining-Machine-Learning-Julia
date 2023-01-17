include("misc.jl")

function leastSquaresBias(X, y)

    (n, d) = size(X)
    oneCol = ones(n, 1)
    Z = hcat(oneCol, X)

    # Find regression weights minimizing squared error
    w = (Z'Z) \ (Z'y)

    # Make linear prediction function
    function predict(Xhat)
        l = length(Xhat)
        oneCol = ones(l, 1)
        Z = hcat(oneCol, Xhat)
        return Z * w
    end

    # Return model
    return GenericModel(predict)
end