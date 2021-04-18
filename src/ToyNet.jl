module ToyNet

using LinearAlgebra
export NN, NN2, predict, loss, accuracy, numerical_gradient, onehot, addlayer!, addlastlayer!, gradient

abstract type AbstractLayer end

# neural network structure
mutable struct NN
    size::Tuple       # size of this network
    params::Dict{String, AbstractArray}
    layers::Vector{AbstractLayer}
    lastlayer::AbstractLayer
    NN(s, w1, b1, w2, b2) = new(s, Dict("w1"=>w1,"b1"=>b1,"w2"=>w2,"b2"=>b2), [])
end

include("layer.jl")

# init neural network
# num of input, num of hidden nodes, num of output
function NN2(i::Int, h::Int, o::Int, weight_init_std=0.01)
    s = (i, h ,o)
    w1 = randn(h, i) * weight_init_std
    b1 = zeros(h)
    w2 = randn(o, h) * weight_init_std
    b2 = zeros(o)
    return NN(s, w1, b1, w2, b2)
end


#= add layer to network
function sequential!(net::NN, layer::Function...)
net.layer = [layer[i](net, i) for i in 1:length(layer)]
end
=#


# add layer to network
function addlayer!(net::NN, layer::AbstractLayer)
    push!(net.layers, layer)
end


function addlastlayer!(net::NN, layer::AbstractLayer)
    net.lastlayer = layer
end


# network, af of hidden layer, af of output layer, input, layer num
function predict(net::NN, x)
    for layer in net.layers
        x = forward!(layer, x)
    end
    return x
end


# loss
# input data, teaching data
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


function accuracy(net::NN, x, t)
    y = predict(net, x)
    y = mapslices(argmax, y, dims=1)
    if ndims(t) != 1
        t = mapslices(argmax, t, dims=1)
    end
    accuracy = sum(y .== t) / float(size(x, 2))
    return accuracy
end

function gradient(net::NN, x, t)
    # forward
    loss(net, x, t)
    # backbard
    dout = 1
    dout = backward!(net.lastlayer, dout)
    layers = reverse(net.layers)
    for layer in layers
        dout = backward!(layer, dout)
    end
    return Dict(
        "w1" => net.layers[1].dw,
        "b1" => net.layers[1].db,
        "w2" => net.layers[3].dw,
        "b2" => net.layers[3].db,
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
