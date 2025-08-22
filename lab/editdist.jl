
function editdist(a::AbstractString, b::AbstractString)
    m, n = length(a), length(b)
    dist = Array{Int, 2}(undef, m+1, n+1)
    dist[:, n+1] = m:-1:0
    dist[m+1, :] = n:-1:0
    for j in n:-1:1
        for i in m:-1:1
            dist[i, j] = min(dist[i+1, j+1] + (a[i]!=b[j]), 
                             dist[i+1, j] + 1, 
                             dist[i, j+1] + 1)
        end
    end
    println(a, " ", b)
    display(dist)
    println()
    return dist[1, 1]
end

editdist("rake", "sake")
editdist("abc", "def")
editdist("abcdef", "bcdefg")

editdist("ake", "sake")
editdist("abc", "def")
editdist("abcdef", "bcdefg")

editdist("abcdef", "acxydf")

editdist("abcdevwxyz", "acdemnvwxz")

using Random
@show c = randstring(20)
@show a = randstring(10) * c
@show b = c * randstring(10)
@show editdist(a, b)