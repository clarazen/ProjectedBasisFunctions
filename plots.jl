using Plots
using LinearAlgebra
using Statistics
using PGFPlotsX

err_gp      =   [   1.08261   1.02276   0.960662  0.841045  0.568106  0.820469  0.749959  0.896886  0.931924  0.674067;
                    0.388433  0.425909  0.390408  0.376836  0.409963  0.416612  0.367794  0.384706  0.378804  0.365581;
                    0.330288  0.31443   0.305569  0.315485  0.327945  0.306818  0.274255  0.319312  0.318528  0.318045;
                    0.284498  0.286041  0.285571  0.276299  0.277106  0.29693   0.281338  0.267997  0.266632  0.269538
                ]

err_rr      =   [   0.979795  1.0013    0.989184  0.878741  0.938945  0.995195  1.00421   0.998501  0.997925  1.0025;
                    0.678236  0.711248  0.621615  0.614855  0.654717  0.587747  0.6481    0.659975  0.629814  0.597431;
                    0.391158  0.395178  0.401342  0.366606  0.420159  0.365058  0.354231  0.445032  0.427749  0.394462;
                    0.282535  0.284268  0.28456   0.275352  0.276527  0.294198  0.278822  0.265083  0.266603  0.270121
                ]

err_tt      =   [   0.0305079  0.034008  0.0337478  0.0349449  0.0377734  0.0316733  0.0399313  0.0301216  0.0357206  0.0359804;
                    0.123497   0.138137  0.119403   0.121629   0.114529   0.125899   0.117704   0.110313   0.115962   0.121477;
                    0.263443   0.250774  0.236771   0.249338   0.273479   0.234157   0.21217    0.247734   0.245583   0.248505;
                    0.282535   0.284268  0.28456    0.275352   0.276527   0.294198   0.278822   0.265083   0.266603   0.270121
                ]

m_err_gp = mean.(eachrow(err_gp))
m_err_rr = mean.(eachrow(err_rr))
m_err_tt = mean.(eachrow(err_tt))
s_err_gp = std.(eachrow(err_gp))
s_err_rr = std.(eachrow(err_rr))
s_err_tt = std.(eachrow(err_tt))


#plot(1:4,mean.(eachrow(err_gp)), ribbon = (var.(eachrow(err_gp)),var.(eachrow(err_gp))))
#plot!(1:4,mean.(eachrow(err_rr)), ribbon = (var.(eachrow(err_rr)),var.(eachrow(err_rr))))
#plot!(1:4,mean.(eachrow(err_tt)), ribbon = (var.(eachrow(err_tt)),var.(eachrow(err_tt))))


figure = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "TT-ranks",
        ylabel = "Relative error",
        xtick = [1,2,3,4],
        xticklabels = ["1","5","10","20"]
    },
    Plot(
        {
            color = "red",
            mark  = "x",
            "error bars/y dir=both",
            "error bars/y explicit"
        },
        Coordinates(collect(1:4),m_err_gp,; yerror = s_err_gp)
    ),
    Plot(
        {
            color = "blue",
            mark  = "x",
            "error bars/y dir=both",
            "error bars/y explicit"
        },
        Coordinates(collect(1:4),m_err_rr,; yerror = s_err_rr)
    ),
    Plot(
        {
            color = "green",
            mark  = "x",
            "error bars/y dir=both",
            "error bars/y explicit"
        },
        Coordinates(collect(1:4),m_err_tt,; yerror = s_err_tt)
    ),
    Legend(["full GP","reduced-rank","Our method"])
)

pgfsave("rel_err.tex", figure; include_preamble= false, dpi = 150)

# plot for optimize hyper parameters
err_gp      =   [   0.291788  0.284608  0.280415  0.299321  0.282108  0.275978  0.283596  0.295266  0.287585  0.267737;
                    0.279367  0.269456  0.269156  0.287429  0.298933  0.299645  0.294799  0.29485   0.267047  0.271708;
                    0.272312  0.277661  0.287266  0.289294  0.268445  0.288561  0.295568  0.283899  0.27443   0.256575;
                    0.277481  0.266614  0.275623  0.290372  0.285319  0.28728   0.298917  0.28919   0.265865  0.261447

                ]

err_rr      =   [    0.962468  0.975077  0.980473  0.988316  0.976113  0.966683  0.970651  0.967451  0.974081  0.963892;
                    0.661883  0.675064  0.679449  0.685405  0.692024  0.690734  0.701451  0.695111  0.642895  0.618922;
                    0.374856  0.377687  0.383167  0.377834  0.370828  0.385334  0.397879  0.388552  0.362615  0.344365;
                    0.276253  0.265549  0.274123  0.288281  0.284112  0.286392  0.297425  0.287352  0.263944  0.260349

                ]

err_tt      =   [    0.145952  0.150135  0.144333  0.155077  0.153566  0.14519   0.147718  0.1519    0.159255  0.1496;
                    0.148586  0.148375  0.151129  0.157561  0.170239  0.166994  0.159816  0.153618  0.145199  0.138842;
                    0.241124  0.235928  0.248598  0.251591  0.236998  0.246233  0.260087  0.255588  0.238605  0.222133;
                    0.276253  0.265549  0.274123  0.288281  0.284112  0.286392  0.297425  0.287352  0.263944  0.260349

                ]

m_err_gp = mean.(eachrow(err_gp))
m_err_rr = mean.(eachrow(err_rr))
m_err_tt = mean.(eachrow(err_tt))
s_err_gp = std.(eachrow(err_gp))
s_err_rr = std.(eachrow(err_rr))
s_err_tt = std.(eachrow(err_tt))

σ_f = [0.001008464050168852, 0.053819928190294415, 0.300429143960008, 1.0825492540674233]

figure = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "TT-ranks",
        ylabel = "Relative error",
        xtick = [1,2,3,4],
        xticklabels = ["1","5","10","20"],
        legend_pos  = "north center"
    },
    Plot(
        {
            color = "red",
            mark  = "x",
            "error bars/y dir=both",
            "error bars/y explicit"
        },
        Coordinates(collect(1:4),m_err_gp,; yerror = s_err_gp)
    ),
    Plot(
        {
            color = "blue",
            mark  = "x",
            "error bars/y dir=both",
            "error bars/y explicit"
        },
        Coordinates(collect(1:4),m_err_rr,; yerror = s_err_rr)
    ),
    Plot(
        {
            color = "green",
            mark  = "x",
            "error bars/y dir=both",
            "error bars/y explicit"
        },
        Coordinates(collect(1:4),m_err_tt,; yerror = s_err_tt)
    ),
    Plot(
        {
            color = "orange",
            mark  = "*",
        },
        Coordinates(collect(1:4),σ_f)
    ),
    Legend(["full GP","reduced-rank","Our method","optimzied sigma"])
)

pgfsave("rel_err_hyp.tex", figure; include_preamble= false, dpi = 150)

## MSLL
err_gp      =   [    420.265  392.433  367.545  403.51   391.28   403.386  451.975  442.31   406.923  391.053;
                    373.213  368.459  373.669  403.458  405.078  404.417  416.41   416.527  371.52   381.395;
                    379.871  389.484  400.981  421.34   398.614  411.925  419.646  410.51   417.659  376.441;
                    394.536  392.652  410.714  442.416  423.956  433.922  450.953  432.686  395.631  379.232
                ]

err_rr      =   [     8705.79   8877.11   9185.24   8880.91   8982.89   9200.97   9875.71   9254.3    8501.38   8975.31;
                    3959.61   4383.88   4665.16   4486.86   4147.32   4204.59   4585.55   4525.43   4068.37   3718.65;
                    1004.2    1015.33   1028.03   1024.51   1073.56   1034.97   1070.12   1081.98   1031.99    945.598;
                    400.854   398.713   418.185   448.879   432.503   444.451   458.932   439.887   402.083   386.509
                ]

err_tt      =   [    196.545  206.668  195.197  214.836  218.583  203.915  225.037  224.395  223.662  212.619;
                    195.338  207.608  224.844  231.178  243.665  237.422  233.656  216.448  203.713  182.657;
                    404.704  380.495  414.257  441.259  419.123  413.482  442.816  446.536  417.601  371.404;
                    400.854  398.713  418.185  448.879  432.503  444.451  458.932  439.887  402.083  386.509
                ]

m_err_gp = mean.(eachrow(err_gp))
m_err_rr = mean.(eachrow(err_rr))
m_err_tt = mean.(eachrow(err_tt))
s_err_gp = std.(eachrow(err_gp))
s_err_rr = std.(eachrow(err_rr))
s_err_tt = std.(eachrow(err_tt))


figure = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "TT-ranks",
        ylabel = "MSLL",
        xtick = [1,2,3,4],
        xticklabels = ["1","5","10","20"]
    },
    Plot(
        {
            color = "red",
            mark  = "x",
            "error bars/y dir=both",
            "error bars/y explicit"
        },
        Coordinates(collect(1:4),m_err_gp,; yerror = s_err_gp)
    ),
    Plot(
        {
            color = "blue",
            mark  = "x",
            "error bars/y dir=both",
            "error bars/y explicit"
        },
        Coordinates(collect(1:4),m_err_rr,; yerror = s_err_rr)
    ),
    Plot(
        {
            color = "green",
            mark  = "x",
            "error bars/y dir=both",
            "error bars/y explicit"
        },
        Coordinates(collect(1:4),m_err_tt,; yerror = s_err_tt)
    ),
    Legend(["Full GP","Hilbert-GP","Algorithm 1"])
)

pgfsave("rel_err_MSLL.tex", figure; include_preamble= false, dpi = 150)