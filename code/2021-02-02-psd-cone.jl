"""
Positive definite matrix (covariance matrix) distances.
This script compares arithmetic mean interpolation and geometric mean interpolation between two cov mats.
Lyndon Duong, Jan 2021
"""
##
using Plots
pyplot()
using LinearAlgebra

@userplot CovPlot
"""Plots 2D covariance matrix as ellipse"""
@recipe function f(cp::CovPlot)
    C, col = cp.args
    dx = 2π/51
    x = (0:dx:(2π))
    dots = [cos.(x)'; 
            sin.(x)']
    aspect_ratio --> 1
    label --> false
    linewidth --> 3
    xlims --> (-2, 2)
    ylims --> (-2, 2)
    seriescolor --> col
    ticks --> false
    framestyle --> :none

    ellipse = C * dots
    ellipse[1,:], ellipse[2,:]
end

##

""" Weighted geometric mean between two covariances
Eq 6.11 Bhatia, Positive Definite Matrices (2007)
"""
function geomsum(C1::Matrix, C2::Matrix, t::Float64)
    Ct = C1^.5 * (C1^-.5 * C2 * C1^-.5)^t * C1^.5
    real(Ct)
end

"""Arithmetic weighted mean between two covs"""
function arisum(C1::Matrix, C2::Matrix, t::Float64)
    Ct = (1-t)*C1 + (t) * C2
end

"""Rotates matrix C by theta"""
function rot(C::Matrix, theta::Float64)
    R = [cos(theta) -sin(theta); 
        sin(theta)  cos(theta)]
    return R'*C*R
end

# Create skinny covariance matrix and rotate it
n = 60
cola = range(colorant"red", stop=colorant"yellow", length=n)
colg = range(colorant"blue", stop=colorant"pink", length=n)
C0 = [2. 0.; 0. 1/3]
C1 = rot(C0, -π/4)
C2 = rot(C0, π/4)

ts = range(0, 1, length=n)
Cg = map(t->geomsum(C1,C2,t), ts)
Ca = map(t->arisum(C1,C2,t), ts)

p = plot(covplot(C1, cola[1]), covplot(C1, colg[1]))
anim = @animate for (i, cc) in enumerate(zip(Ca, Cg))
    ca, cg = cc
    p = plot(covplot(ca, cola[i,:]), covplot(cg, colg[i,:]))
end

gif(anim, "anim.gif", fps = 10)

##  plot area of ellipse at each step of iteration

fig = plot(map(det, Ca), color=cola, linewidth=3, ylabel="area of ellipse", label="arithmetic mean", xlabel="step")
plot!(map(det, Cg), color=colg, linewidth=3, label="geometric mean")

savefig(fig, "area.png")

## plot positive definite cone and trajectories

"""xyz coordinates in this 3D spare are sigma_x, sigma_y, and sigma_xy
"""

pyplot()
x = 0:.05:2

x = x' .* ones(length(x))
y = x'

f(x, y) = sqrt.(x.*y)
colmap = range(colorant"black", colorant"grey", length=20)

# plot cone
surface(x, y, f(x,y), color=colormap("grays"))
surface!(x,y, -f(x,y), color=colormap("grays"))
plot!(legend=false, colorbar=false, ticks=false)
for t=0:0.2:1  # plot lines radiating from origin
    plot!([0,2],[0,2*t^2],[0,2*t], color="black")
    plot!([0,2*t^2],[0,2],[0,2*t], color="black")
    plot!([0,2],[0,2*t^2],-[0,2*t], color="black")
    plot!([0,2*t^2],[0,2],-[0,2*t], color="black")
end

# plot trajectories of arithmetic and geometric 
ptg = reduce(hcat, [[c[1], c[4], c[3]] for c in Cg])
pta = reduce(hcat, [[c[1], c[4], c[3]] for c in Ca])
plot!(pta[1,:], pta[2,:], pta[3,:], color=cola, linewidth=4)
plot!(ptg[1,:], ptg[2,:], ptg[3,:], color=colg, linewidth=4)

animcone = @animate for i in vcat(45:10:175, 175:-10:45)
    plot!(camera=(i, 20))
end

gif(animcone, "animcone.gif", fps=5)