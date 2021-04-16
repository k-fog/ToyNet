include("./ToyNet.jl")

using MLDatasets
using .ToyNet

function main()
    train_x, train_t = MNIST.traindata()
    # test_x,  test_y  = MNIST.testdata()
    losslist = []

    iters_num = 10
    train_size = size(train_x)[3]
    batch_size = 100
    learning_rate = 0.1

    network = NN2(784, 100, 10)
    addlayer!(network, ReluLayer(network, 1))
    addlayer!(network, AffineLayer(network, 1))
    addlayer!(network, ReluLayer(network, 2))
    addlayer!(network, AffineLayer(network, 2))
    addlastlayer!(network, SoftmaxWithLossLayer())

    for _ in 1:iters_num
        batch_mask = rand(1:train_size, batch_size)
        batch_x = reshape(train_x[:,:,batch_mask], (:, batch_size))
        batch_t = onehot(10, train_t[batch_mask])
        # @show predict(network, batch_x)
        grad = gradient(network, batch_x, batch_t)

        for (key, value) in grad
            if key[1] == 'w'
                network.w[parse(Int, key[2])] -= learning_rate * value
            else
                network.b[parse(Int, key[2])] -= learning_rate * value
            end
        end

        l = loss(network, batch_x, batch_t)
        println(l)
        push!(losslist, l)
    end
end

@time main()
