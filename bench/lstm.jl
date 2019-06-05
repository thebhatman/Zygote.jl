using Zygote, Flux, BenchmarkTools, Test, DataFrames, GLM, Statistics
using Plots
import Flux: gate

include("fakearray.jl")

struct LSTMCell{A,V}
  Wi::A
  Wh::A
  b::V
  h::V
  c::V
end

function LSTMCell(in::Integer, out::Integer; init = randn, batch = 1)
  cell = LSTMCell(init(out*4, in), init(out*4, out), init(out*4),
                  init(out), init(out))
  return cell
end

function (m::LSTMCell)(h, c, x)
  b, o = m.b, size(h, 1)
  g = m.Wi*x .+ m.Wh*h .+ b
  input = σ.(gate(g, o, 1))
  forget = σ.(gate(g, o, 2))
  cell = tanh.(gate(g, o, 3))
  output = σ.(gate(g, o, 4))
  c = forget .* c .+ input .* cell
  h′ = output .* tanh.(c)
  return h′, c, h′
end

m = LSTMCell(10, 10, init = FakeArray)
x = FakeArray(10, 5)
m(m.h, m.c, x)

function test(m::LSTMCell, x, len)
  h = m.h
  c = m.c
  i = 0
  while (i += 1) <= len
    h, c, x = m(h, c, x)
  end
  return sum(x)
end

gtest(m, x, len) = gradient(test, m, x, len)

const N = 100
const len = 10

# m = LSTMCell(N, N, batch = 0)
# x = randn(Float32, N, 0)
# m(m.h, m.c, x)
# @benchmark $m($m.h, $m.c, $x)

@profiler for i = 1:1000
  gtest(m, x, len)
end

function timing(batch)
  model = LSTMCell(N, N, batch = batch)
  x = randn(Float32, N, batch)
  minimum(@benchmark $test($model, $x, $len)).time
end

timing(0)

3200 / 424

@profiler for i = 1:1000
  test(m, x, len)
end

@benchmark $(sin.(rand(10, 0)))

data = DataFrame(X=batches, Y=timings)
ols = lm(@formula(Y ~ X), data)
overhead = coeftable(ols).cols[1][1]

lstm_ops = 424
samples = values(overhead_estimates)./lstm_ops
println("Mean overhead: $(mean(samples))ns ± $(std(samples))")
