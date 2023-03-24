using Plots
using PGFPlotsX
using DelimitedFiles

t = 800
x           = collect(1:t)
ytest       = Matrix(readdlm("benchmark data/ytest_kukainv.csv",','))[1:t,1]
yfull       = Matrix(readdlm("benchmark data/mstar_full.csv",','))[1:t,1]
ybf         = Matrix(readdlm("benchmark data/m_bf.csv",','))[1:t,1]
your         = Matrix(readdlm("benchmark data/m_our.csv",','))[1:t,1]

figure = @pgf Axis(
    {
        xlabel = "Time [s]",
        ylabel = "Motor torque [Nm]",
        no_markers
    },
    Plot(Table([x, ytest])),
    LegendEntry("Validation data"),
    Plot(Table([x, yfull])),
    LegendEntry("Full GP"),
    Plot(Table([x, ybf])),
    LegendEntry("Hilbert-GP"),
    Plot(Table([x, your])),
    LegendEntry("Algorithm ...")
)

pgfsave("inversedynamics.tex", figure; include_preamble=false, dpi = 150)