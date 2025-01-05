(* Functions for the DNN paper.

Notes:
- We work in log space to avoid numerical instabilities.
- For the inverse CDF we use two forms: one for cdf<0.5 (that works well close to 0), and one for cdf > 0.5 (for close to 1).

*)

EvaluatePreviousCell = ResourceFunction["EvaluatePreviousCell"];

ParallelOuter[f_, args__, opts : OptionsPattern[ParallelMap]] :=
    With[{fullData = Map[Inactive[Identity], Outer[List, args], {Length
         @ {args}}]},
        Activate @ ArrayReshape[ParallelMap[Inactive[Identity] @* Apply[
            f] @* Activate, Flatten @ fullData, opts], Dimensions @ fullData]
    ]

ParallelOuterWithData[f_, data_, args__, opts : OptionsPattern[ParallelMap
    ]] := (* TODO: rewrite based on ParallelOuter's rewrite *)With[{fullData
     = MapThread[Inactive[Identity] @* Prepend, {Outer[List, args], data},
     2]},
        Activate @ ArrayReshape[ParallelMap[Inactive[Identity] @* Apply[
            f], Activate @ Flatten @ fullData, {Length @ Dimensions @ fullData - 
            1}, opts], Dimensions @ fullData]
    ]

LogSumExp = ResourceFunction["LogSumExp"]

LogAvgExp[x_] :=
    LogSumExp[x] - Log @ Length @ x

StableDist[\[Alpha]_] :=
    StableDistribution[\[Alpha], 0, 0, 2 ^ (-1 / \[Alpha])]

(* With[{
        c =
            Function[\[Alpha]1,
                Gamma[1 + \[Alpha]1] Sin[\[Pi] \[Alpha]1 / 2] / \[Pi]
                    
            ]
    },
        (c[\[Alpha]] / (4 c[\[Alpha] / 2])) ^ (2 / \[Alpha])
    ] // FullSimplify *)

SDist[\[Alpha]_?NumericQ] :=
    If[\[Alpha] < 2,
        StableDistribution[\[Alpha] / 2, 1, 0, (4 \[Pi]) ^ (-1 / \[Alpha]
            ) (2^\[Alpha] Cos[(\[Pi] \[Alpha]) / 4] Gamma[(1 + \[Alpha]) / 2]) ^ 
            (2 / \[Alpha])]
        ,
        TransformedDistribution[1, {x \[Distributed] NormalDistribution[
            ]}]
    ]

LogFPStable[\[Alpha]_?NumericQ, Log\[Sigma]w_?NumericQ, Log\[Sigma]b_
    ?NumericQ, \[Phi]_, stableSamples_] :=
    If[Log\[Sigma]w == -\[Infinity],
        \[Alpha] Log\[Sigma]b
        ,
        With[{
            z =
                If[IntegerQ @ stableSamples,
                    RandomVariate[StableDist[\[Alpha]], stableSamples
                        ]
                    ,
                    stableSamples
                ]
        },
            LogSumExp[{Logq2, \[Alpha] Log\[Sigma]b}] /. FindRoot[Logq2
                 - \[Alpha] Log\[Sigma]w - LogAvgExp[\[Alpha] Log @ Abs[\[Phi][z Exp[
                LogSumExp[{Logq2, \[Alpha] Log\[Sigma]b}] / \[Alpha]]]]], {Logq2, 1}]
                
        ]
    ]

LogFPStable[\[Alpha]_?NumericQ, Log\[Sigma]w_?NumericQ, -\[Infinity],
     \[Phi]_, stableSamples_] :=
    If[Log\[Sigma]w == -\[Infinity],
        -\[Infinity]
        ,
        With[{
            z =
                If[IntegerQ @ stableSamples,
                    RandomVariate[StableDist[\[Alpha]], stableSamples
                        ]
                    ,
                    stableSamples
                ]
        },
            Logq2 /. FindRoot[Logq2 - \[Alpha] Log\[Sigma]w - LogAvgExp[
                \[Alpha] Log @ Abs[\[Phi][z Exp[Logq2 / \[Alpha]]]]], {Logq2, 1}]
        ]
    ]

LogFPStable[\[Alpha]_?NumericQ, Log\[Sigma]w_?NumericQ, Log\[Sigma]b_
    ?NumericQ, \[Phi]_] :=
    If[Log\[Sigma]w == -\[Infinity],
        \[Alpha] Log\[Sigma]b
        ,
        LogSumExp[{Logq2, \[Alpha] Log\[Sigma]b}] /. FindRoot[Logq2 -
             \[Alpha] Log\[Sigma]w - Log @ NExpectation[Exp[\[Alpha] Log @ Abs[\[Phi][
            z Exp[LogSumExp[{Logq2, \[Alpha] Log\[Sigma]b}] / \[Alpha]]]]], {z \[Distributed]
             StableDist[\[Alpha]]}], {Logq2, 1}]
    ]

LogFPStable[\[Alpha]_?NumericQ, Log\[Sigma]w_?NumericQ, -\[Infinity],
     \[Phi]_] :=
    If[Log\[Sigma]w == -\[Infinity],
        -\[Infinity]
        ,
        Logq2 /. FindRoot[Logq2 - \[Alpha] Log\[Sigma]w - Log @ NExpectation[
            Exp[\[Alpha] Log @ Abs[\[Phi][z Exp[Logq2 / \[Alpha]]]]], {z \[Distributed]
             StableDist[\[Alpha]]}], {Logq2, 1}]
    ]

LogSech2[x_ ? (VectorQ[#, NumericQ]&)] :=
    2 (Log[2] + x - LogSumExp[{2 x, ConstantArray[0, Length @ x]}]);

LogSech2[x_?NumericQ] :=
    First @ LogSech[{x}]

LogY[\[Alpha]_?NumericQ, Logr_?NumericQ, Log\[Chi]_ ? (VectorQ[#, NumericQ
    ]&), LogS_ ? (VectorQ[#, NumericQ]&), LogSp_ ? (VectorQ[#, NumericQ]&
    )] :=
    Logy /.
        FindRoot[
            \[Alpha] Logy - LogAvgExp[(\[Alpha] / 2) (2 Log\[Chi] + LogS
                 - LogSumExp[{ConstantArray[2 Logr - 2 Logy, Length @ Log\[Chi]], 2 Log\[Chi]
                 + LogS + LogSp}])], {Logy, 0}(*,
EvaluationMonitor:>Print[Logy]*) ]

LogY[\[Alpha]_?NumericQ, Logr_?NumericQ, \[Chi]Dist_?DistributionParameterQ,
     Log\[Chi]Fn_] :=
    Logy /. FindRoot[-\[Alpha] Logy + Log @ NExpectation[Exp[(\[Alpha]
         / 2) (2 Log\[Chi]Fn[\[Chi]] + Log @ S - LogSumExp[{2 Logr - 2 Logy, 
        2 Log\[Chi]Fn[\[Chi]] + Log @ S + Log @ Sp}])], {S \[Distributed] SDist[
        \[Alpha]], Sp \[Distributed] SDist[\[Alpha]], \[Chi] \[Distributed] \[Chi]Dist
        }], {Logy, 0}]

LogEigCDF[\[Alpha]_?NumericQ, Logr_?NumericQ, Log\[Chi]_ ? (VectorQ[#,
     NumericQ]&)] :=
    With[{LogS = Log @ RandomVariate[SDist[\[Alpha]], Length @ Log\[Chi]
        ]},
        With[{LogSp = RotateRight @ LogS},
            With[{Logy = LogY[\[Alpha], Logr, Log\[Chi], LogS, LogSp]
                },
                2 Logr - 2 Logy + LogAvgExp[-LogSumExp[{ConstantArray[
                    2 Logr - 2 Logy, Length @ Log\[Chi]], 2 Log\[Chi] + LogS + LogSp}]]
            ]
        ]
    ]

LogEigCDF[\[Alpha]_?NumericQ, Logr_?NumericQ, \[Chi]Dist_?DistributionParameterQ,
     Log\[Chi]Fn_] :=
    With[{Logy = LogY[\[Alpha], Logr, \[Chi]Dist, Log\[Chi]Fn]},
        2 Logr - 2 Logy + Log @ NExpectation[Exp[-LogSumExp[{2 Logr -
             2 Logy, 2 Log\[Chi]Fn[\[Chi]] + Log @ S + Log @ Sp}]], {S \[Distributed]
             SDist[\[Alpha]], Sp \[Distributed] SDist[\[Alpha]], \[Chi] \[Distributed]
             \[Chi]Dist}]
    ]

LogEigSurvival[\[Alpha]_?NumericQ, Logr_?NumericQ, Log\[Chi]_ ? (VectorQ[
    #, NumericQ]&)] :=
    With[{LogS = Log @ RandomVariate[SDist[\[Alpha]], Length @ Log\[Chi]
        ]},
        With[{LogSp = RotateRight @ LogS},
            With[{Logy = LogY[\[Alpha], Logr, Log\[Chi], LogS, LogSp]
                },
                LogAvgExp[2 Log\[Chi] + LogS + LogSp - LogSumExp[{ConstantArray[
                    2 Logr - 2 Logy, Length @ Log\[Chi]], 2 Log\[Chi] + LogS + LogSp}]]
            ]
        ]
    ]

LogEigSurvival[\[Alpha]_?NumericQ, Logr_?NumericQ, \[Chi]Dist_?DistributionParameterQ,
     Log\[Chi]Fn_] :=
    With[{Logy = LogY[\[Alpha], Logr, \[Chi]Dist, Log\[Chi]Fn]},
        Log @ NExpectation[Exp[2 Log\[Chi]Fn[\[Chi]] + Log @ S + Log 
            @ Sp - LogSumExp[{2 Logr - 2 Logy, 2 Log\[Chi]Fn[\[Chi]] + Log @ S + 
            Log @ Sp}]], {S \[Distributed] SDist[\[Alpha]], Sp \[Distributed] SDist[
            \[Alpha]], \[Chi] \[Distributed] \[Chi]Dist}]
    ]

(* here fn can be LogEigCDF or LogEigSurvival *)

JacobianDNN[fn_, \[Alpha]_?NumericQ, Log\[Sigma]w_?NumericQ, Log\[Sigma]b_,
     \[Phi]_, Logr_?NumericQ, Log\[Phi]p_, stableSamples_, LogfpStable_:None
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
            Log\[Chi] =
                Log\[Sigma]w +
                    Log\[Phi]p[
                        Exp[
                                If[LogfpStable === None,
                                        LogFPStable[\[Alpha], Log\[Sigma]w,
                                             Log\[Sigma]b, \[Phi], hList]
                                        ,
                                        LogfpStable
                                    ] / \[Alpha]
                            ] hList
                    ]
        },
            fn[\[Alpha], Logr, Log\[Chi]]
        ]
    ]

JacobianTanh[fn_, \[Alpha]_?NumericQ, Log\[Sigma]w_?NumericQ, Logr_?NumericQ,
     stableSamples_, LogfpStable_:None] :=
    JacobianDNN[fn, \[Alpha], Log\[Sigma]w, Log @ 0, Tanh, Logr, LogSech2,
         stableSamples, LogfpStable]

LogRHat[\[Alpha]_?NumericQ, Logs_?NumericQ, Log\[Chi]_ ? (VectorQ[#, 
    NumericQ]&), LogS_ ? (VectorQ[#, NumericQ]&), LogSp_ ? (VectorQ[#, NumericQ
    ]&), invCDFflag_:True] :=
    LogrHat /.
        If[invCDFflag,
            FindRoot[Logs - 2 LogrHat - LogAvgExp[-LogSumExp[{ConstantArray[
                2 LogrHat, Length @ Log\[Chi]], 2 Log\[Chi] + LogS + LogSp}]], {LogrHat,
                 0}]
            ,
(*in this case Logs is log the survival fn (i.e. log(1-cdf)
    *)
            FindRoot[Logs - LogAvgExp[2 Log\[Chi] + LogS + LogSp - LogSumExp[
                {ConstantArray[2 LogrHat, Length @ Log\[Chi]], 2 Log\[Chi] + LogS + LogSp
                }]], {LogrHat, 0}]
        ]

LogInvCDF[\[Alpha]_?NumericQ, Logs_?NumericQ, Log\[Chi]_ ? (VectorQ[#,
     NumericQ]&), invCDFFlag_:True] :=
    With[{LogS = Log @ RandomVariate[SDist[\[Alpha]], Length @ Log\[Chi]
        ]},
        With[{LogSp = RotateRight @ LogS},
            With[{LogrHat = LogRHat[\[Alpha], Logs, Log\[Chi], LogS, 
                LogSp, invCDFFlag]},
                LogrHat + (1 / \[Alpha]) LogAvgExp[2 LogrHat - LogSumExp[
                    {ConstantArray[2 LogrHat, Length @ Log\[Chi]], 2 Log\[Chi] + LogS + LogSp
                    }]]
            ]
        ]
    ]

JacobianLogInvCDF[\[Alpha]_?NumericQ, Log\[Sigma]w_?NumericQ, Log\[Sigma]b_,
     \[Phi]_, Logs_?NumericQ, Log\[Phi]p_, stableSamples_, invCDFflag_:True,
     LogfpStable_:None] :=
    With[{
        hList =
            If[IntegerQ @ stableSamples,
                RandomVariate[StableDist[\[Alpha]], stableSamples]
                ,
                stableSamples
            ]
    },
        With[{
            Log\[Chi] =
                Log\[Sigma]w +
                    Log\[Phi]p[
                        Exp[
                                If[LogfpStable === None,
                                        LogFPStable[\[Alpha], Log\[Sigma]w,
                                             Log\[Sigma]b, \[Phi], hList]
                                        ,
                                        LogfpStable
                                    ] / \[Alpha]
                            ] hList
                    ]
        },
            LogInvCDF[\[Alpha], Logs, Log\[Chi], invCDFflag]
        ]
    ]

EmpiricalH[\[Alpha]_, \[Sigma]w_, \[Sigma]b_, \[Phi]_, width_, depth_
    ] :=
    Module[{h = RandomReal[{-1, 1}, width]},
        Do[
            With[{M = \[Sigma]w width ^ (-1 / \[Alpha]) RandomVariate[
                StableDist[\[Alpha]], {width, width}], b = \[Sigma]b RandomVariate[StableDist[
                \[Alpha]], width]},
                h = M . \[Phi][h] + b
            ]
            ,
            depth
        ];
        h
    ]

(* Map functions *)
(* one should probably prefer more uniform samples with fewer stable samples to get an accurate average (but more stable samples are better for getting the shape of the CDF) *)

GetLogFPStable[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, prefix_:"fig/data"
    ] :=
    With[{paths = FileNames[StringTemplate["``/``/``/``/logfp_*"][prefix,
         \[Alpha]100, \[Sigma]w100, \[Sigma]b100]]},
        If[Length @ paths > 0,
            Get @ First @ paths
            ,
            With[{fname = FileNameJoin @ {prefix, StringTemplate["``/``/``/logfp_``.txt"
                ][\[Alpha]100, \[Sigma]w100, \[Sigma]b100, CreateUUID[]]}, logfpStable
                 = LogFPStable[\[Alpha]100 / 100., N @ Log[\[Sigma]w100 / 100], N @ Log[
                \[Sigma]b100 / 100], Tanh]},
                Quiet[CreateDirectory @ DirectoryName @ fname, CreateDirectory
                    ::eexist];
                Put[logfpStable, fname];
                logfpStable
            ]
        ]
    ]

SavePuts[label_, fmt_:"Table", red_:Identity, prefix_:"fig/data"] :=
    GroupBy[FileNames @ FileNameJoin @ {prefix, StringTemplate["*/*/*/``_*.txt"
        ][label]}, (ToExpression @ FileNameSplit[#][[-4 ;; -2]]&) -> (Import[
        #, fmt]&), red] // Export[FileNameJoin @ {prefix, StringTemplate["``.mx"
        ][label]}, #]&

PutJacobianLogInvCDF[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, stableSamples_,
     numAvgSamples_, prefix_:"fig/data"] :=
    With[{logfpStable = GetLogFPStable[\[Alpha]100, \[Sigma]w100, \[Sigma]b100
        ]},
            Table[
                Quiet @
                    JacobianLogInvCDF[
                        \[Alpha]100 / 100.
                        ,
                        N @ Log[\[Sigma]w100 / 100]
                        ,
                        N @ Log[\[Sigma]b100 / 100]
                        ,
                        Tanh
                        ,
                        If[s < 0.5,
                            Log @ s
                            ,
                            Log[1 - s]
                        ]
                        ,
                        LogSech2
                        ,
                        stableSamples
                        ,
                        s < 0.5
                        ,
                        logfpStable
                    ]
                ,
                {s, RandomReal[1, numAvgSamples]}
            ]
        ] // Export[FileNameJoin @ {prefix, StringTemplate["``/``/``/loginvCDF_``_``.txt"
            ][\[Alpha]100, \[Sigma]w100, \[Sigma]b100, stableSamples, CreateUUID[
            ]]}, #, "Table"]&

(* PutJacobianEigs[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, n_, prefix_
    :"fig/data"] :=
    With[{logfpStable = GetLogFPStable[\[Alpha]100, \[Sigma]w100, \[Sigma]b100
        ]},
        Export[
            FileNameJoin @ {prefix, StringTemplate["``/``/``/jacEigs_``_``.txt"
                ][\[Alpha]100, \[Sigma]w100, \[Sigma]b100, n, CreateUUID[]]}
            ,
            (\[Sigma]w100 / 100) DiagonalMatrix[Tanh'[Exp[logfpStable
                 / (\[Alpha]100 / 100.)] RandomVariate[StableDist[\[Alpha]100 / 100.],
                 n]]] . RandomVariate[StableDist[\[Alpha]100 / 100.], {n, n}] / n ^ (
                1 / (\[Alpha]100 / 100.)) //
            Eigenvalues //
            ReIm
            ,
            "Table"
        ]
    ] *)

PutLogNeuralNorm[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, Infinity, prefix_:
    "fig/data"] :=
    With[{logfpStable = GetLogFPStable[\[Alpha]100, \[Sigma]w100, \[Sigma]b100
        ]},
            Log @ NExpectation[Tanh[Exp[logfpStable / (\[Alpha]100 / 
                100.)] h] ^ 2, {h \[Distributed] StableDist[\[Alpha]100 / 100]}]
        ] // Export[FileNameJoin @ {prefix, StringTemplate["``/``/``/logNeuralNorm_``.txt"
            ][\[Alpha]100, \[Sigma]w100, \[Sigma]b100, CreateUUID[]]}, #, "Table"
            ]&

PutLogNeuralNorm[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, stableSamples_,
  prefix_ : "fig/data"] := 
 With[{logfpStable = 
     GetLogFPStable[\[Alpha]100, \[Sigma]w100, \[Sigma]b100]}, 
   LogAvgExp[
    2 Log@Abs@
       Tanh[Exp[logfpStable/(\[Alpha]100/100.)] RandomVariate[
          StableDist[\[Alpha]100/100.], 1000]]]] // 
  Export[FileNameJoin@{prefix, 
      StringTemplate[
        "``/``/``/logNeuralNorm_``.txt"][\[Alpha]100, \[Sigma]w100, \
\[Sigma]b100, CreateUUID[]]}, #, "Table"] &

PutEmpiricalLogNeuralNorm[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_,
     width_, depth_, prefix_:"fig/data"] :=
    With[{h = EmpiricalH[\[Alpha]100 / 100., \[Sigma]w100 / 100., \[Sigma]b100
         / 100., Tanh, width, depth]},
            LogAvgExp[2 Log @ Abs @ Tanh[h]]
        ] // Export[FileNameJoin @ {prefix, StringTemplate["``/``/``/empiricalLogNeuralNorm_``_``_``.txt"
            ][\[Alpha]100, \[Sigma]w100, \[Sigma]b100, width, depth, CreateUUID[]
            ]}, #, "Table"]&



PutEmpiricalLogSingVals[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, width_,
     depth_, \[Phi]_:Tanh, prefix_:"fig/data", label_:"empiricalLogSingVals"
    ] :=
    With[{\[Alpha] = \[Alpha]100 / 100., \[Sigma]w = \[Sigma]w100 / 100.,
         \[Sigma]b = \[Sigma]b100 / 100.},
            With[{h = EmpiricalH[\[Alpha], \[Sigma]w, \[Sigma]b, \[Phi],
                 width, depth], M = \[Sigma]w width ^ (-1 / \[Alpha]) RandomVariate[StableDist[
                \[Alpha]], {width, width}]},
                Log @ SingularValueList[M . DiagonalMatrix[\[Phi]'[h]
                    ]]
            ]
        ] // Export[FileNameJoin @ {prefix, StringTemplate["``/``/``/``_``_``_``.txt"
            ][\[Alpha]100, \[Sigma]w100, \[Sigma]b100, label, width, depth, CreateUUID[
            ]]}, #, "Table"]&

PutEmpiricalLogAbsEigs[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, width_,
     depth_, \[Phi]_:Tanh, prefix_:"fig/data", label_:"empiricalLogAbsEigs"
    ] :=
    With[{\[Alpha] = \[Alpha]100 / 100., \[Sigma]w = \[Sigma]w100 / 100.,
         \[Sigma]b = \[Sigma]b100 / 100.},
            With[{h = EmpiricalH[\[Alpha], \[Sigma]w, \[Sigma]b, \[Phi],
                 width, depth], M = \[Sigma]w width ^ (-1 / \[Alpha]) RandomVariate[StableDist[
                \[Alpha]], {width, width}]},
                Log @ Abs @ Eigenvalues[M . DiagonalMatrix[\[Phi]'[h]
                    ]]
            ]
        ] // Export[FileNameJoin @ {prefix, StringTemplate["``/``/``/``_``_``_``.txt"
            ][\[Alpha]100, \[Sigma]w100, \[Sigma]b100, label, width, depth, CreateUUID[
            ]]}, #, "Table"]&

(*
    Usage: wolframscript -f filename.wl Function1[...] ... FunctionN[...]
    In unix, enclose the functions in double quotes to escape the spaces.
    To pass through strings to wolfram, escape the quotes.
*)

If[Length @ $ScriptCommandLine > 0,
    c = (Print @* EchoTiming @* ToExpression) /@ $ScriptCommandLine[[
        2 ;; ]]
]
