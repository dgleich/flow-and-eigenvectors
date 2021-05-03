using Plots
function graph_lines(A::SparseMatrixCSC, xy)
    ei,ej = findnz(triu(A,1))[1:2]
    # find the line segments
    lx = zeros(0)
    ly = zeros(0)
    for nz=1:length(ei)
        src = ei[nz]
        dst = ej[nz]
        push!(lx, xy[src,1])
        push!(lx, xy[dst,1])
        push!(lx, Inf)

        push!(ly, xy[src,2])
        push!(ly, xy[dst,2])
        push!(ly, Inf)
    end
    return lx, ly
end
