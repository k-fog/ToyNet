using ToyNet, Flux
using Test

input = rand(10)
@test ToyNet.sigmoid(input) === NNlib.sigmoid(input)

