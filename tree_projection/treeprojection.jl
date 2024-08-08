using FileIO
using LasIO
using LinearAlgebra
using GLMakie
using Makie
using Distributed
using StaticArrays
using Base.Threads: @threads


las2jl(p, header) = @SVector [xcoord(p, header), ycoord(p, header), zcoord(p, header)]

findhighestpoint(points) = points[argmax(getindex.(points, 3))]

function hist2d(points, xbins, ybins)
	output = zeros(Int32, length(ybins), length(xbins))
	@threads for p in points
		x = floor(Int, (p[1] - minimum(xbins)) / step(xbins)) + 1
		y = floor(Int, (p[2] - minimum(ybins)) / step(ybins)) + 1
		output[y, x] += 1
	end
	return output
end

function offset(points)
	highestpoint = findhighestpoint(points)
	return @SVector [highestpoint[1], highestpoint[2], 0]
end

function get_plane(angle)
	u = @SVector [cos(angle), sin(angle), 0]
	v = @SVector [0., 0., 1.]
	n = @SVector [-sin(angle), cos(angle), 0]
	return u, v, n
end

function project_to_plane(p, u, v)
	return @SVector [dot(p, u), dot(p,v)]
end

function project_points(points, u, v)
	out = reinterpret(SVector{2, Float32}, zeros(Float32, length(points) * 2))
	@threads for i in 1:length(points)
		out[i] = project_to_plane(points[i], u, v)
	end
	return out
end

function add_random_offsets!(points, max_offset)
	@threads for i in 1:length(points)
		points[i] += @SVector [rand() * max_offset, rand() * max_offset, rand() * max_offset]
	end
	return points
end

function process_tree(input_file, output_folder)

	header, laspoints = load(input_file)

	jlpoints = [las2jl(p, header) for p in laspoints]
	add_random_offsets!(jlpoints, 0.01)
	o = offset(jlpoints)
	shifted_points = [p - o for p in jlpoints]

	for angle in (0, 45, 90, 135)
		filename = split(basename(input_file), '.')[1]
		
		u,v,n = get_plane(deg2rad(angle))
	
		projected_points = project_points(shifted_points, u, v)
	
		xext = extrema(getindex.(projected_points, 1))
		yext = extrema(getindex.(projected_points, 2))
		xl = xext[2] - xext[1]
		yl = yext[2] - yext[1]
		ypx = 800
		xpx = round(Int, (ypx/(yl/xl)))
		pd = sqrt((xl*yl) / length(projected_points))
		xr = range(;start=xext[1], step=pd, stop=xext[2])
		yr = range(;start=yext[1], step=pd, stop=yext[2])
	
		arr = hist2d(projected_points, xr, yr)
		larr = log.(arr)

		set_theme!(figure_padding = 0)
		fig = Figure(;size=(ypx * 3 ÷ 4, ypx))
		ax = Axis(fig[1, 1], aspect = xpx/ypx)
		hidedecorations!(ax)
		hidespines!(ax)
		image!(ax, larr'; colormap=:binary, colorrange=(-1, maximum(larr)*0.5), interpolate=false)
		
		filepath = joinpath(output_folder, "$(filename)_$(angle)_$(round(yl/ypx*100; digits=3))cm-px.png")
		save(filepath, fig; px_per_unit=1)
	end
	return nothing
end

function main(args)
    path = args[1]
    species_folders = filter(isdir, readdir(path, join=true))
    output_folder = args[2]
    nfolder = length(species_folders)
    map(1:nfolder) do i
        output_folder_name = basename(species_folders[i])
		this_species_output_folder = joinpath(output_folder, output_folder_name)
		mkpath(this_species_output_folder)
        println("processing $output_folder_name")
        files = readdir(species_folders[i], join=true)
        for file in files
            println("processing $file")
            process_tree(file, this_species_output_folder)
        end
    end
end

@profview main([".", "/tmp"])
