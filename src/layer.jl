abstract type AbstractLayer end

# sigmoid layer
mutable struct SigmoidLyr <: AbstractLayer
    out::AbstractArray
end

function forward!(lyr::SigmoidLyr, x)
    out = 1 ./ (1 . exp(-x))
    lyr.out = out
    return out
end


function backward!(lyr::SigmoidLyr, dout)
    dx = dout .* (1.0 .- lyr.out) .* lyr.out
end


# affine layer
mutable struct AffineLayer <: AbstractLayer
    w::AbstractArray
    b::AbstractArray
    x::AbstractArray
    dw::AbstractArray
    db::AbstractArray
end

function forward!(lyr::AffineLayer, x)
    lyr.x = x
    out = lyr.w * x .+ b
    return out
end

function backward!(lyr::AffineLayer, dout)
    dx = lyr.w' * dout
    lyr.dw = dout * lyr.x'
    lyr.db = sum(dout)
    return dx
end
