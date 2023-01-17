using Printf
include("misc.jl")
include("findMin.jl")

function robustRegression(X, y)

    (n, d) = size(X)

    # Initial guess
    w = zeros(d, 1)

    # Function we're going to minimize (and that computes gradient)
    funObj(w) = robustRegressionObj(w, X, y)

    # This is how you compute the function and gradient:
    (f, g) = funObj(w)

    # Derivative check that the gradient code is correct:
    g2 = numGrad(funObj, w)

    if maximum(abs.(g - g2)) > 1e-4
        @printf("User and numerical derivatives differ:\n")
        @show([g g2])
    else
        @printf("User and numerical derivatives agree\n")
    end

    # Solve least squares problem
    w = findMin(funObj, w)

    # Make linear prediction function
    predict(Xhat) = Xhat * w

    # Return model
    return GenericModel(predict)
end

function robustRegressionObj(w, X, y)
    f = 0
    g = zeros(size(w))
    e = 1

    n, d = size(X)
    for i in 1:n
        yihat = X[i] * w
        r = yihat .- y[i]
        r = r[1]

        if abs(r) <= e
            f += 0.5 * (r^2)
            g[1] += (r) * X[i]
        else
            f += e * (abs(r) - (0.5 * e))
            g[1] += e * (sign(r)) * X[i]
        end
    end
    return (f, g)
end

