(* functions for the DNN paper *)

(* Work with everything in log space *)

(* For the inverse CDF use two forms: one for cdf<0.5 (that works well close to 0), and one for cdf > 0.5 (for close to 1) *)

EvaluatePreviousCell = ResourceFunction["EvaluatePreviousCell"];

ParallelOuter[f_, args__, opts : OptionsPattern[ParallelMap]] :=
    With[{fullData = Map[Inactive[Identity], Outer[List, args], {2}]},
        
        Activate @ ArrayReshape[ParallelMap[Inactive[Identity] @* Apply[
            f], Activate @ Flatten @ fullData, {Length @ Dimensions @ fullData - 
            1}, opts], Dimensions @ fullData]
    ]

ParallelOuterWithData[f_, data_, args__, opts : OptionsPattern[ParallelMap
    ]] :=
    With[{fullData = MapThread[Inactive[Identity] @* Prepend, {Outer[
        List, args], data}, 2]},
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

GetLogFPStable[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, prefix_:"fig/data/logfp"
    ] :=
    With[{paths = FileNames[StringTemplate["``_``_``_``*"][prefix, \[Alpha]100,
         \[Sigma]w100, \[Sigma]b100]]},
        If[Length @ paths > 0,
            Get @ First @ paths
            ,
            With[{logfpStable = LogFPStable[\[Alpha]100 / 100., N @ Log[
                \[Sigma]w100 / 100], N @ Log[\[Sigma]b100 / 100], Tanh]},
                Put[logfpStable, StringTemplate["``_``_``_``_``.txt"][
                    prefix, \[Alpha]100, \[Sigma]w100, \[Sigma]b100, CreateUUID[]]];
                logfpStable
            ]
        ]
    ]

PutJacobianLogAvg[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, stableSamples_,
     numAvgSamples_, prefix_:"fig/data/jaclogavg"] :=
    With[{logfpStable = GetLogFPStable[\[Alpha]100, \[Sigma]w100, \[Sigma]b100
        ]},
        Put[
            Mean @
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
            ,
            numAvgSamples
            ,
            StringTemplate["``_``_``_``_``_``.txt"][prefix, \[Alpha]100,
                 \[Sigma]w100, \[Sigma]b100, stableSamples, CreateUUID[]]
        ]
    ]

GetJacobianLogAvg[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, stableSamples_,
     prefix_:"out/jaclogavg"] :=
    With[{paths = FileNames[StringTemplate["``_``_``_``_``*"][prefix,
         \[Alpha]100, \[Sigma]w100, \[Sigma]b100, stableSamples]]},
        Mean @ WeightedData[Sequence @@ Transpose[Import[#, "List"]& 
            /@ paths]]
    ]

PutNeuralNorm[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, 
  prefix_ : "fig/data/logneuralnorm"] := 
 With[{paths = 
    FileNames[
     StringTemplate["``_``_``_``*"][
      prefix, \[Alpha]100, \[Sigma]w100,
           \[Sigma]b100]]}, 
  If[Length@paths > 0, Get@First@paths, 
   With[{logfpStable = 
      GetLogFPStable[\[Alpha]100, \[Sigma]w100, \[Sigma]b100
               ]},
    Put[
     Log@
      NExpectation[
       Tanh[Exp[
           logfpStable/(\[Alpha]100/100.)] h]^2, {h \[Distributed] 
         StableDist[\[Alpha]100/100.]}],
     StringTemplate["``_``_``_``_``.txt"][prefix,
           \[Alpha]100, \[Sigma]w100, \[Sigma]b100, CreateUUID[]]
     ]]
   ]]

(* can also write a consolidating function for the estimates *)

SaveLogFPStable[path_:"fig/data"] :=
    First @* First /@ GroupBy[FileNames @ FileNameJoin[{path, "logfp_*_*_*_*.txt"
        }], (ToExpression @ StringSplit[FileBaseName[#], "_"][[2 ;; 4]]&) -> 
        (Import[#, "List"]&)] // Export[FileNameJoin[Most @ FileNameSplit[path
        ] // Append @ "logfp.mx"], #]&

SaveJacobianLogAvg[path_:"fig/data"] :=
    {Mean[#], Total @ #["Weights"]}& @* Apply[WeightedData] @* Transpose /@
         GroupBy[FileNames @ FileNameJoin[{path, "jaclogavg_*_*_*_*_*.txt"}],
         (ToExpression @ StringSplit[FileBaseName[#], "_"][[2 ;; 5]]&) -> (Import[
        #, "List"]&)] // Export[FileNameJoin[Most @ FileNameSplit[path] // Append
         @ "jaclogavg.mx"], #]&
        
SaveNeuralNorms[path_ : "fig/data"] := 
 First @* First /@ GroupBy[FileNames
          @ 
     FileNameJoin[{path, 
       "logneuralnorm_*_*_*_*.txt"}], (ToExpression @ StringSplit[
              FileBaseName[#], "_"][[2 ;; 4]] &) -> (Import[#, 
        "List"] &)] // Export[
        FileNameJoin[
     Most @ FileNameSplit[path] // Append @ "logneuralnorm.mx"], #] &

(* one should probably prefer more uniform samples with fewer stable samples to get an accurate average (but more stable samples are better for getting the shape of the CDF) *)

(* cluster computing function using the HPCs *)

LaunchPhysicsKernels[] :=
    (
        LaunchKernels[6];
        LaunchKernels["ssh://cartman?20"];
        (* LaunchKernels["ssh://bebe?16"]; *)
        LaunchKernels["ssh://stan?20"];
        Return[Length @ Kernels[]]
    )

(* wolframscript -f filename.wl Function1[...] ... FunctionN[...] *)

(* in unix can enclose the functions in double quotes to escape the spaces  *)

If[Length @ $ScriptCommandLine > 0,
    c = (Print @* EchoTiming @* ToExpression) /@ $ScriptCommandLine[[
        2 ;; ]]
]
