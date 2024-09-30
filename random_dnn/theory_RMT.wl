<<"utils.wl"

RNN[\[Alpha]_, g_, \[Phi]_, nNeurons_, nIters_, dt_:1*^-3] :=
    With[{M = g nNeurons ^ (-1 / \[Alpha]) RandomVariate[StableDist[\[Alpha]
        ], {nNeurons, nNeurons}]},
        Module[{h = RandomReal[1, nNeurons]},
            Do[h += dt (-h + M . \[Phi][h]), nIters];
            h
        ]
    ]

JacobianEigCDF[\[Alpha]_?NumericQ, g_?NumericQ, rList_ ? (VectorQ[#, 
    NumericQ]&), \[Phi]_, stableSamples_, fpStable_:None, SDistSamples_:None
    ] :=
    With[{
        hList =
            If[IntegerQ @ stableSamples,
                RandomVariate[StableDist[\[Alpha]], stableSamples]
                ,
                stableSamples
            ]
    },
        With[{
            \[Chi]Samples =
                g
                    \[Phi]'[
                        If[fpStable === None,
                                    FPStable[\[Alpha], g, 0, \[Phi], 
                                        hList]
                                    ,
                                    fpStable
                                ] ^ (1 / \[Alpha]) hList
                    ]
        },
            EigCDF[\[Alpha], rList, \[Chi]Samples, SDistSamples]
        ]
    ]

JacobianEigCDF[\[Alpha]_?NumericQ, g_?NumericQ, r_?NumericQ, \[Phi]_,
     stableSamples_, fpStable_:None, SDistSamples_:None] :=
    With[{
        hList =
            If[IntegerQ @ stableSamples,
                RandomVariate[StableDist[\[Alpha]], stableSamples]
                ,
                stableSamples
            ]
    },
        With[{
            \[Chi]Samples =
                g
                    \[Phi]'[
                        If[fpStable === None,
                                    FPStable[\[Alpha], g, 0, \[Phi], 
                                        hList]
                                    ,
                                    fpStable
                                ] ^ (1 / \[Alpha]) hList
                    ]
        },
            EigCDF[\[Alpha], r, \[Chi]Samples, SDistSamples]
        ]
    ]

JacobianEigCDF[\[Alpha]_?NumericQ, g_?NumericQ, r_?NumericQ, \[Phi]_] :=
    EigCDF[\[Alpha], r, StableDist[\[Alpha]], g \[Phi]'[# FPStable[\[Alpha],
         g, 0, \[Phi]] ^ (1 / \[Alpha])]&]

(*vectorised fast version:*)

EigCDF[\[Alpha]_?NumericQ, rList_ ? (VectorQ[#, NumericQ]&), \[Chi]Samples_
     ? (VectorQ[#, NumericQ]&), SDistSamples_:None] :=
    With[{
        SList =
            If[SDistSamples === None,
                RandomVariate[SDist[\[Alpha]], Length @ \[Chi]Samples
                    ]
                ,
                SDistSamples
            ]
    },
        With[{\[Chi]2S = \[Chi]Samples^2 SList, Sp = RotateRight @ SList
            },
            Table[
                With[{y = Abs[y] /. FindRoot[Abs[y] ^ \[Alpha] - Mean[
                    (\[Chi]2S / ((r / Abs @ y) ^ 2 + \[Chi]2S Sp)) ^ (\[Alpha] / 2)], {y,
                     1, 0.5}]},
                    Mean[(r / y) ^ 2 / ((r / y) ^ 2 + \[Chi]2S Sp)]
                ]
                ,
                {r, rList}
            ]
        ]
    ]

(*single val fast version:*)

EigCDF[\[Alpha]_?NumericQ, r_?NumericQ, \[Chi]Samples_ ? (VectorQ[#, 
    NumericQ]&), SDistSamples_:None] :=
    With[{
        SList =
            If[SDistSamples === None,
                RandomVariate[SDist[\[Alpha]], Length @ \[Chi]Samples
                    ]
                ,
                SDistSamples
            ]
    },
        With[{\[Chi]2S = \[Chi]Samples^2 SList, Sp = RotateRight @ SList
            },
            With[{y = Abs[y] /. FindRoot[Abs[y] ^ \[Alpha] - Mean[(\[Chi]2S
                 / ((r / Abs @ y) ^ 2 + \[Chi]2S Sp)) ^ (\[Alpha] / 2)], {y, 1, 0.5}]
                },
                Mean[(r / y) ^ 2 / ((r / y) ^ 2 + \[Chi]2S Sp)]
            ]
        ]
    ]

(*exact version, never finishes (even if you manually MC sample y \
with Mean[])*)

EigCDF[\[Alpha]_?NumericQ, r_?NumericQ, \[Chi]Dist_?DistributionParameterQ,
     \[Chi]Fn_] :=
    With[{y = Abs[y] /. FindRoot[Abs[y] ^ \[Alpha] - NExpectation[((\[Chi]Fn[
        \[Chi]] ^ 2 S) / ((r / Abs @ y) ^ 2 + \[Chi]Fn[\[Chi]] ^ 2 S Sp)) ^ (
        \[Alpha] / 2), {S \[Distributed] SDist[\[Alpha]], Sp \[Distributed] SDist[
        \[Alpha]], \[Chi] \[Distributed] \[Chi]Dist}, Method -> "MonteCarlo"],
         {y, 1, 0.5}]},
        NExpectation[(r / y) ^ 2 / ((r / y) ^ 2 + \[Chi]Fn[\[Chi]] ^ 
            2 S Sp), {S \[Distributed] SDist[\[Alpha]], Sp \[Distributed] SDist[\[Alpha]
            ], \[Chi] \[Distributed] \[Chi]Dist}, Method -> "MonteCarlo"]
    ]

(* JacobianEigInverseCDF[\[Alpha]_?NumericQ, g_?NumericQ, cdf_?NumericQ,
     \[Phi]_, stableSamples_, SDistSamples_:None] :=
    With[{
        hList =
            If[IntegerQ @ stableSamples,
                RandomVariate[StableDist[\[Alpha]], stableSamples]
                ,
                stableSamples
            ]
    },
        With[{\[Chi]Samples = g \[Phi]'[FPStable[\[Alpha], g, 0, \[Phi],
             stableSamples] ^ (1 / \[Alpha]) hList]},
            EigInverseCDF[\[Alpha], cdf, \[Chi]Samples, SDistSamples]
                
        ]
    ]

EigInverseCDF[\[Alpha]_?NumericQ, cdf_?NumericQ, \[Chi]Samples_ ? (VectorQ[
    #, NumericQ]&), SDistSamples_:None] :=
    With[{
        SList =
            If[SDistSamples === None,
                RandomVariate[SDist[\[Alpha]], Length @ \[Chi]Samples
                    ]
                ,
                SDistSamples
            ]
    },
        With[{\[Chi]2S = \[Chi]Samples^2 SList, Sp = RotateRight @ SList
            },
            With[{rym1 = Abs[rym1] /. FindRoot[Mean[Abs[rym1] ^ 2 / (
                Abs[rym1] ^ 2 + \[Chi]2S Sp)] - cdf, {rym1, 1*^-2, 1*^3}]},
                rym1 Mean[(\[Chi]2S / (rym1^2 + \[Chi]2S Sp)) ^ (\[Alpha]
                     / 2)] ^ (1 / \[Alpha])
            ]
        ]
    ] *)

ClearAll @ JacobianEigInverseCDF;

JacobianEigInverseCDF[\[Alpha]_?NumericQ, g_?NumericQ, cdf_?NumericQ,
     \[Phi]_, stableSamples_, SDistSamples_:None] :=
    With[{
        hList =
            If[IntegerQ @ stableSamples,
                RandomVariate[StableDist[\[Alpha]], stableSamples]
                ,
                stableSamples
            ]
    },
        With[{\[Chi]Samples = g \[Phi]'[FPStable[\[Alpha], g, 0, \[Phi],
             stableSamples] ^ (1 / \[Alpha]) hList]},
            EigInverseCDF[\[Alpha], cdf, \[Chi]Samples, SDistSamples]
                
        ]
    ]

JacobianEigInverseCDF[\[Alpha]_?NumericQ, g_?NumericQ, cdfList_ ? (VectorQ[
    #, NumericQ]&), \[Phi]_, stableSamples_, SDistSamples_:None] :=
    With[{
        hList =
            If[IntegerQ @ stableSamples,
                RandomVariate[StableDist[\[Alpha]], stableSamples]
                ,
                stableSamples
            ]
    },
        With[{\[Chi]Samples = g \[Phi]'[FPStable[\[Alpha], g, 0, \[Phi],
             stableSamples] ^ (1 / \[Alpha]) hList]},
            EigInverseCDF[\[Alpha], cdfList, \[Chi]Samples, SDistSamples
                ]
        ]
    ]

ClearAll @ EigInverseCDF;

EigInverseCDF[\[Alpha]_?NumericQ, cdf_?NumericQ, \[Chi]Samples_ ? (VectorQ[
    #, NumericQ]&), SDistSamples_:None] :=
    With[{
        SList =
            If[SDistSamples === None,
                RandomVariate[SDist[\[Alpha]], Length @ \[Chi]Samples
                    ]
                ,
                SDistSamples
            ]
    },
        With[{\[Chi]2S = \[Chi]Samples^2 SList, Sp = RotateRight @ SList
            },
            With[{rym1 = Exp @ LogRym1 /. FindRoot[Mean[Exp[2 LogRym1
                ] / (Exp[2 LogRym1] + \[Chi]2S Sp)] - cdf, {LogRym1, 0}]},
                rym1 Mean[(\[Chi]2S / (rym1^2 + \[Chi]2S Sp)) ^ (\[Alpha]
                     / 2)] ^ (1 / \[Alpha])
            ]
        ]
    ]

EigInverseCDF[\[Alpha]_?NumericQ, cdfList_ ? (VectorQ[#, NumericQ]&),
     \[Chi]Samples_ ? (VectorQ[#, NumericQ]&), SDistSamples_:None] :=
    With[{
        SList =
            If[SDistSamples === None,
                RandomVariate[SDist[\[Alpha]], Length @ \[Chi]Samples
                    ]
                ,
                SDistSamples
            ]
    },
        With[{\[Chi]2S = \[Chi]Samples^2 SList, Sp = RotateRight @ SList
            },
            Table[
                With[{rym1 = Exp @ LogRym1 /. FindRoot[Mean[Exp[2 LogRym1
                    ] / (Exp[2 LogRym1] + \[Chi]2S Sp)] - cdf, {LogRym1, 0}]},
                    rym1 Mean[(\[Chi]2S / (rym1^2 + \[Chi]2S Sp)) ^ (
                        \[Alpha] / 2)] ^ (1 / \[Alpha])
                ]
                ,
                {cdf, cdfList}
            ]
        ]
    ]

(* Getting the CDFs on a grid in \[Alpha],g *)

(* GetJacobianEigCDFs[] :=
    With[{fpStable = Import["fig/fpStable.wxf"], rList = Range[0, 10,
         0.01]},
            <|"gList" -> fpStable["\[Sigma]wList"], "\[Alpha]List" ->
                 fpStable["\[Alpha]List"], "rList" -> rList, "CDF" -> ParallelOuterWithData[
                 @ Quiet @ JacobianEigCDF[#1, #2, rList, Tanh, 10000, #3]&, fpStable[
                "data"], fpStable["\[Alpha]List"], fpStable["\[Sigma]wList"]]|>
        ] // Export["fig/jacobianEigCDFs.wxf", #]& *)

(* Getting the effective spectral radius *)

(* GetJacobianEigInverseCDF[] :=
    ParallelOuter[Quiet @ JacobianEigInverseCDF[#1, #2, .99, Tanh, 10000
        ]&, Range[1, 2, .01], Range[0, 5, .01]] // ListDensityPlot[Transpose 
        @ #, PlotLegends -> Automatic]& *)

PlotLogMeasure[] :=
    ParallelOuter[Mean @ Table[Quiet @ Log @ JacobianEigInverseCDF[#1,
         #2, u, Tanh, 1000], {u, Subdivide[0, 1, 100]}]&, Range[1, 2, .1], Rest
         @ Range[0, 3, .1]] //
    Transpose //
    ListDensityPlot //
    Export["fig/log.png", #]&

ClearAll @ LogJacobianEigInverseCDF;

LogJacobianEigInverseCDF[\[Alpha]_?NumericQ, g_?NumericQ, cdf_?NumericQ,
     \[Phi]_, stableSamples_, SDistSamples_:None] :=
    With[{
        hList =
            If[IntegerQ @ stableSamples,
                RandomVariate[StableDist[\[Alpha]], stableSamples]
                ,
                stableSamples
            ]
    },
        With[{\[Chi]Samples = g \[Phi]'[FPStable[\[Alpha], g, 0, \[Phi],
             stableSamples] ^ (1 / \[Alpha]) hList]},
            Log @ EigInverseCDF[\[Alpha], cdf, \[Chi]Samples, SDistSamples
                ]
        ]
    ]

LogJacobianEigInverseCDF[\[Alpha]_?NumericQ, g_?NumericQ, cdfList_ ? 
    (VectorQ[#, NumericQ]&), \[Phi]_, stableSamples_, SDistSamples_:None] :=
    With[{
        hList =
            If[IntegerQ @ stableSamples,
                RandomVariate[StableDist[\[Alpha]], stableSamples]
                ,
                stableSamples
            ]
    },
        With[{\[Chi]Samples = g \[Phi]'[FPStable[\[Alpha], g, 0, \[Phi],
             stableSamples] ^ (1 / \[Alpha]) hList]},
            Log @ EigInverseCDF[\[Alpha], cdfList, \[Chi]Samples, SDistSamples
                ]
        ]
    ]

ClearAll @ LogEigInverseCDF;

LogEigInverseCDF[\[Alpha]_?NumericQ, cdf_?NumericQ, \[Chi]Samples_ ? 
    (VectorQ[#, NumericQ]&), SDistSamples_:None] :=
    With[{
        SList =
            If[SDistSamples === None,
                RandomVariate[SDist[\[Alpha]], Length @ \[Chi]Samples
                    ]
                ,
                SDistSamples
            ]
    },
        With[{\[Chi]2S = \[Chi]Samples^2 SList, Sp = RotateRight @ SList
            },
            With[{LogRym1 = LogRym1 /. FindRoot[Mean[Exp[2 LogRym1] /
                 (Exp[2 LogRym1] + \[Chi]2S Sp)] - cdf, {LogRym1, 0}]},
                LogRym1 + Log @ Mean[(\[Chi]2S / (Exp[2 LogRym1] + \[Chi]2S
                     Sp)) ^ (\[Alpha] / 2)] ^ (1 / \[Alpha])
            ]
        ]
    ]

LogEigInverseCDF[\[Alpha]_?NumericQ, cdfList_ ? (VectorQ[#, NumericQ]&
    ), \[Chi]Samples_ ? (VectorQ[#, NumericQ]&), SDistSamples_:None] :=
    With[{
        SList =
            If[SDistSamples === None,
                RandomVariate[SDist[\[Alpha]], Length @ \[Chi]Samples
                    ]
                ,
                SDistSamples
            ]
    },
        With[{\[Chi]2S = \[Chi]Samples^2 SList, Sp = RotateRight @ SList
            },
            Table[
                With[{LogRym1 = LogRym1 /. FindRoot[Mean[Exp[2 LogRym1
                    ] / (Exp[2 LogRym1] + \[Chi]2S Sp)] - cdf, {LogRym1, 0}]},
                    LogRym1 + Log @ Mean[(\[Chi]2S / (Exp[2 LogRym1] 
                        + \[Chi]2S Sp)) ^ (\[Alpha] / 2)] / \[Alpha]
                ]
                ,
                {cdf, cdfList}
            ]
        ]
    ]

GetJacobianEigInverseCDF[] :=
    With[{\[Alpha]List = Range[1, 2, .1], gList = Rest @ Range[0, 3, 
        .1], cdfList = Subdivide[0, 1, 100]},
            <|"\[Alpha]List" -> \[Alpha]List, "gList" -> gList, "LogInvCDF"
                 -> ParallelOuter[Quiet @ LogJacobianEigInverseCDF[#1, #2, cdfList, Tanh,
                 10000]&, \[Alpha]List, gList]|>
        ] // Export["fig/LogInvCDF.wxf", #]&

c = (Print @* EchoTiming @* ToExpression) /@ $ScriptCommandLine[[2 ;; ]]
