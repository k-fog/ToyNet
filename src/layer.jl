export SigmoidLayer, AffineLayer, SoftmaxWithLossLayer, forward!, backward!

# sigmoid layer
mutable struct SigmoidLayer <: AbstractLayer
    out::AbstractArray
end

SigmoidLayer() = SigmoidLayer(undef)
SigmoidLayer(net::NN, n::Int) = SigmoidLayer(undef)

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
end

AffineLayer() = AffineLayer(undef, undef, undef, undef, undef)
AffineLayer(net::NN, n::Int) = AffineLayer(net.w[n], net.b[n], undef, undef, undef)
AffineLayer(w, b) = AffineLayer(w, b, undef, undef, undef)

function forward!(layer::AffineLayer, x)
    layer.x = x
    out = layer.w * x .+ b
    return out
end

function backward!(layer::AffineLayer, dout)
    dx = layer.w' * dout
    layer.dw = dout * layer.x'
    layer.db = sum(dout)
    return dx
end


#softmax with loss layer
mutable struct SoftmaxWithLossLayer <: AbstractLayer
    loss::AbstractArray
    y::AbstractArray
    t::AbstractArray
end

SoftmaxWithLossLayer() = SoftmaxWithLossLayer(undef, undef, undef)
SoftmaxWithLossLayer(net::NN, n::Int) = SoftmaxWithLossLayer()

function forward!(layer::SoftmaxWithLossLayer, x, t)
    layer.t = t
    layer.y = y
    layer.loss = cross_entropy_error(layer.y, layer.t)
    return layer.loss
end

function backward!(layer::SoftmaxWithLossLayer, dout=1)
    batch_size = size(layer.t, 3)
    dx = (layer.y .- layer.t) ./ batch_size
    return dx
end
