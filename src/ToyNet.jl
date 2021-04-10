module toynet

mutable struct ToyNet
    size::Vector{Int}
    w1::Matrix{Float64}
    b1::Vector{Float64}
    w2::Matrix{Float64}
    b2::Vector{Float64}
end

function ToyNet(i::Int, h::Int, o::Int)
    size = [i, h, o]
    w1 = rand(i, h)
    b1 = zeros(h)
    w2 = rand(h, o)
    b2 = zeros(o)
    return ToyNet(size, w1, b1, w2, b2)
end

function predict(net::ToyNet, input)
    w1, w2 = net.w1, net.w2
    b1, b2 = net.b1, net.b2

    a1 = input * w1 .+ b1
    z1 = sigmoid(a1)
    a2 = z1 * w2 .+ b2
    y = softmax(a2)

    return y
end

function sigmoid(x)
    return @. 1 / (1 + exp(-x))
end

function softmax(a)
    c = maximum(a)
    exp_a = exp.(a .- c)
    sum_exp_a = sum(exp_a)
    return exp_a / sum_exp_a
end

function main()
    network = ToyNet(3, 2, 1)
    print(softmax([0.3, 2.9, 4.0]))
end

main()
end # module
