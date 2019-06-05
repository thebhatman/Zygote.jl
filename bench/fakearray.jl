struct FakeArray{T,N} <: AbstractArray{T,N}
  size::NTuple{N,Int}
end

FakeArray(sz::Vararg{Int,N}) where N = FakeArray{Float64,N}(sz)

Base.size(x::FakeArray) = x.size
Base.getindex(x::FakeArray, i...) = error("FakeArray has no values.")
Base.show(io::IO, x::FakeArray) = print(io, typeof(x), "(", size(x), ")")
Base.print_array(io::IO, x::FakeArray) = println("(no data)")

function Base.getindex(x::FakeArray, i...)

end
