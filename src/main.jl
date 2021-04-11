include("./ToyNet.jl")

using MLDatasets
using .ToyNet

function main()
    train_x, train_t = MNIST.traindata()
    # test_x,  test_y  = MNIST.testdata()
    losslist = []

    iters_num = 1
    train_size = size(train_x)[3]
    batch_size = 100
    learning_rate = 0.1

    network = NN3(784, 100, 10)
    for _ in 1:iters_num
        batch_mask = rand(1:train_size, batch_size)
        batch_x = reshape(train_x[:,:,batch_mask], (:, batch_size))
        batch_t = onehot(10, train_t[batch_mask])
        @show size(predict(network, batch_x))
        grad = numerical_gradient(network, batch_x, batch_t)
        # @show grad
    end
end

main()
