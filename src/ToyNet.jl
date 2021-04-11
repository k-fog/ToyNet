export ToyNet

module ToyNet

using LinearAlgebra
export NN, NN3, predict, loss, accuracy, numerical_gradient
export onehot

# neural network structure
mutable struct NN
    size::Int         # size of this network
    w::Vector{Matrix} # weights
    b::Vector{Vector} # biases
end


# init neural network
# num of input, num of hidden nodes, num of output
function NN3(i::Int, h::Int, o::Int)
    w = [rand(h, i), rand(o, h)]
    b = [rand(h), rand(o)]
    return NN(3, w, b)
end


# feed forward
# network, layer num, activation function, input
function forward(net::NN, i::Int, h, x)
    return h(net.w[i] * x .+ net.b[i])
end


# execute network
# network, af of hidden layer, af of output layer, input, layer num
function predict(net::NN, h::Function, σ::Function, x, i::Int=1)
    if i == net.size
        return forward(net, i, σ, x)
    else
        return predict(net, h, σ, forward(net, i, h, x), i + 1)
    end
end

function predict(net::NN, x)
    return predict(net, sigmoid, sigmoid, x)
end


# loss
function loss(net::NN, x, t)
    return cross_entropy_error(predict(net, x), t)
end


function numerical_gradient(f::Function, x)
    Δh = 1e-4
    grad = zeros(size(x))
    for idx in 1:length(x)
        tmp = x[idx]
        x[idx] = tmp + Δh
        fxh1 = f(x)
        x[idx] = tmp - Δh
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * Δh)
        x[idx] = tmp
    end
    return grad
end


function numerical_gradient(net::NN, x, t)
    loss_w = w -> loss(net, x, t)
    grads = Dict(
        "w1" => numerical_gradient(loss_w, net.w[1]),
        "b1" => numerical_gradient(loss_w, net.b[1]),
        "w2" => numerical_gradient(loss_w, net.w[2]),
        "b2" => numerical_gradient(loss_w, net.b[2]),
    )
    return grads
end


function sigmoid(x)
    return @. 1 / (1 + exp(-x))
end


function softmax(a)
    c = maximum(a)
    exp_a = exp.(a .- c)
    return exp_a ./ sum(exp_a)
end


function cross_entropy_error(y, t)
    batch_size = size(y, 2)
    δ = 1e-7
    @show size(y)
    @show size(t)
    return -sum(t .* log.(y .+ δ)) / batch_size
end


function onehot(len::Int, ans::Vector)
    r = zeros(len, length(ans))
    for i in 1:length(ans)
        r[ans[i] + 1, i] = 1
    end
    return r
end

end # module