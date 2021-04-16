export ReluLayer, SigmoidLayer, AffineLayer, SoftmaxWithLossLayer, forward!, backward!

#relu layer
mutable struct ReluLayer <: AbstractLayer
    mask::AbstractArray
    ReluLayer(x...) = new()
end

function forward!(layer::ReluLayer, x)
    layer.mask = (x .<= 0)
    out = copy(x)
    out[layer.mask] .= 0
    return out
end

function backward!(layer::ReluLayer, dout)
    dout[layer.mask] .= 0
    dx = dout
    return dx
end


# sigmoid layer
mutable struct SigmoidLayer <: AbstractLayer
    out::AbstractArray
    SigmoidLayer() = new()
    SigmoidLayer(net::NN, n::Int) = new()
end

function forward!(layer::SigmoidLayer, x)
    out = 1 ./ (1 .+ exp.(-x))
    layer.out = out
    return out
end

function backward!(layer::SigmoidLayer, dout)
    dx = dout .* (1.0 .- layer.out) .* layer.out
end


# affine layer
mutable struct AffineLayer <: AbstractLayer
    w::AbstractArray
    b::AbstractArray
    x::AbstractArray
    dw::AbstractArray
    db::AbstractArray
    AffineLayer() = new()
    AffineLayer(net::NN, n::Int) = new(net.w[n], net.b[n])
    AffineLayer(w, b) = new(w, b)
end

function forward!(layer::AffineLayer, x)
    layer.x = x
    out = layer.w * x .+ layer.b
    return out
end

function backward!(layer::AffineLayer, dout)
    dx = layer.w' * dout
    layer.dw = dout * layer.x'
    layer.db = sum.(dout)
    return dx
end


#softmax with loss layer
mutable struct SoftmaxWithLossLayer <: AbstractLayer
    loss::Float64
    y::AbstractArray
    t::AbstractArray
    SoftmaxWithLossLayer() = new()
end

function forward!(layer::SoftmaxWithLossLayer, y, t)
    layer.t = t
    layer.y = y
    layer.loss = cross_entropy_error(layer.y, layer.t)
    return layer.loss
end

function backward!(layer::SoftmaxWithLossLayer, dout=1)
    batch_size = size(layer.t, 2)
    dx = (layer.y .- layer.t) ./ batch_size
    return dx
end
