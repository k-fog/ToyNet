include("./ToyNet.jl")

using MLDatasets
using Plots
using .ToyNet

function main()
    train_x, train_t = MNIST.traindata()
    test_x,  test_t  = MNIST.testdata()
    losslist = []
    accuracylist = []

    iters_num = 10000
    train_size = size(train_x, 3)
    batch_size = 100
    learning_rate = 0.1

    network = NN2(784, 100, 10, optimizer=Momentum())
    addlayer!(network, AffineLayer(network.params["w1"], network.params["b1"]))
    addlayer!(network, ReluLayer())
    addlayer!(network, AffineLayer(network.params["w2"], network.params["b2"]))
    addlayer!(network, ReluLayer())
    addlastlayer!(network, SoftmaxWithLossLayer())

    for i in 0:iters_num
        batch_mask = rand(1:train_size, batch_size)
        batch_x = reshape(train_x[:,:,batch_mask], (:, batch_size))
        batch_t = onehot(10, train_t[batch_mask])
        grad = gradient(network, batch_x, batch_t)

        update!(network.optimizer, network.params, grad)
        l = loss(network, batch_x, batch_t)

        i % 1000 == 0 && println("loss: $l($i)")
        i % 100 == 0 && begin
            a1 = accuracy(network, batch_x, batch_t);
            println("train-accuracy: $a1($i)")
            a2 = accuracy(network, reshape(test_x, (28 ^ 2, :)), onehot(10, test_t))
            println("test-accuracy: $a2($i)")
            push!(accuracylist, a2)
        end

        push!(losslist, l)
    end
    f1 = plot(losslist, label="loss")
    savefig(f1, "fig-loss")
    f2 = plot(accuracylist, label="accuracy")
    savefig(f2, "fig-accuracy")
end

@time main()

