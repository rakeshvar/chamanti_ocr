
function editdist(a::AbstractString, b::AbstractString)
    m, n = length(a), length(b)
    simi = zeros(Int, m, n)
    for i in 1:m
        for j in 1:n
            simi[i, j] = a[i] != b[j]        end    end
    
    dist = fill(-1, m+1, n+1)
    dist[:, n+1] = m:-1:0
    dist[m+1, :] = n:-1:0
    for d in 0:2(m-1)
        for k in 0:d
            i, j = m-(d-k), m-k
            i*j ≤ 0 && continue
            dist[i, j] = min(simi[i, j]+dist[i+1, j+1], 1+dist[i+1, j], 1+dist[i, j+1])
        end
    end
    println(a, " ", b)
    display(simi)
    println()
    display(dist)
    println()
end

editdist("rake", "sake")
editdist("abc", "def")
editdist("abcdef", "bcdefg")

editdist("ake", "sake")
editdist("abc", "def")
editdist("abcdef", "bcdefg")

