module ToyNet

using LinearAlgebra
export NN, NN2, predict, loss, accuracy, numerical_gradient, onehot, addlayer!, addlastlayer!, gradient

abstract type AbstractLayer end

# neural network structure
mutable struct NN
    size::Tuple       # size of this network
    w::AbstractArray
    b::AbstractArray
    layer::Vector{AbstractLayer}
    lastlayer::AbstractLayer
    NN(s, w, b) = new(s, w, b, [])
end

include("layer.jl")

# init neural network
# num of input, num of hidden nodes, num of output
function NN2(i::Int, h::Int, o::Int, weight_init_std=0.01)
    s = (i, h ,o)
    w = [rand(h, i), rand(o, h)] .* weight_init_std
    b = [zeros(h), zeros(o)] .* weight_init_std
    return NN(s, w, b)
end


#= add layer to network
function sequential!(net::NN, layer::Function...)
    net.layer = [layer[i](net, i) for i in 1:length(layer)]
end
=#


# add layer to network
function addlayer!(net::NN, layer::T) where T <: AbstractLayer
    push!(net.layer, layer)
end


function addlastlayer!(net::NN, layer::T) where T <: AbstractLayer
    net.lastlayer = layer
end


# network, af of hidden layer, af of output layer, input, layer num
function predict(net::NN, x)
    for layer in net.layer
        x = forward!(layer, x)
    end
    return x
end


# loss
function loss(net::NN, x, t)
    y = predict(net, x)
    return forward!(net.lastlayer, y, t)
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


function gradient(net::NN, x, t)
    loss(net, x, t)
    dout = 1
    dout = backward!(net.lastlayer, dout)
    layers = reverse(net.layer)
    for layer in layers
        dout = backward!(layer, dout)
    end
    return Dict(
        "w1" => net.w[1],
        "b1" => net.b[1],
        "w2" => net.w[2],
        "b2" => net.b[2],
    )
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
