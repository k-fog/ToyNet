mutable struct SGD <: AbstractOptimizer
    learning_rate::Float64
    SGD(lr=0.01) = new(lr)
end

function update!(opt::SGD, params, grads)
    for key in keys(params)
        params[key] .-= opt.learning_rate * grads[key]
    end
end


mutable struct Momentum <: AbstractOptimizer
    learning_rate::Float64
    momemtum::Float64
    v::Dict
    Momentum(lr=0.01, momemtum=0.9) = new(lr, momemtum)
end

function update!(opt::Momentum, params, grads)
    if !isdefined(opt, :v)
        opt.v = Dict()
        for (key, val) in params
            opt.v[key] = zero(val)
        end
    end
    for key in keys(params)
        opt.v[key] = opt.momemtum * opt.v[key] - opt.learning_rate * grads[key]
        params[key] += opt.v[key]
    end
end
