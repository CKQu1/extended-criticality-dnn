(* Functions for the DNN paper.

Notes:
- We work in log space to avoid numerical instabilities.
- For the inverse CDF we use two forms: one for cdf<0.5 (that works well close to 0), and one for cdf > 0.5 (for close to 1).

*)

prefix = "fig/data";

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

(* these are actually quantiles (deterministic) *)

GetStableSamples[\[Alpha]100_, numSamples_, label_:"stable"] :=
    With[{paths = FileNames @ FileNameJoin @ {prefix, ToString @ \[Alpha]100,
         StringTemplate["``_``_*"][label, numSamples]}},
        If[Length @ paths > 0,
            Import[First @ paths, "Table"][[ ;; , 1]]
            ,
            N @ InverseCDF[StableDist[\[Alpha]100 / 100], Most @ Rest
                 @ Subdivide[numSamples + 1]] //
                With[{fname = FileNameJoin @ {prefix, ToString @ \[Alpha]100,
                     StringTemplate["``_``_``.txt"][label, numSamples, CreateUUID[]]}},
                    Export[fname, #, "Table"];
                    #
                ]&
        ]
    ]

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

GetSSamples[\[Alpha]100_, numSamples_, label_:"s"] :=
    With[{paths = FileNames @ FileNameJoin @ {prefix, ToString @ \[Alpha]100,
         StringTemplate["``_``_*"][label, numSamples]}},
        If[Length @ paths > 0,
            Import[First @ paths, "Table"][[ ;; , 1]]
            ,
            N @ InverseCDF[SDist[\[Alpha]100 / 100], Most @ Rest @ Subdivide[
                numSamples + 1]] //
                With[{fname = FileNameJoin @ {prefix, ToString @ \[Alpha]100,
                     StringTemplate["``_``_``.txt"][label, numSamples, CreateUUID[]]}},
                    Export[fname, #, "Table"];
                    #
                ]&
        ]
    ]

LogQStar[\[Alpha]100_?NumericQ, -\[Infinity] | Indeterminate, -\[Infinity]
     | Indeterminate, \[Phi]_, numSamples_] :=
    -\[Infinity]

LogQStar[\[Alpha]100_?NumericQ, -\[Infinity] | Indeterminate, log\[Sigma]b_,
     \[Phi]_, numSamples_] :=
    With[{\[Alpha] = \[Alpha]100 / 100},
        \[Alpha] log\[Sigma]b
    ]

LogQStar[\[Alpha]100_?NumericQ, log\[Sigma]w_?NumericQ, -\[Infinity] 
    | Indeterminate, \[Phi]_, \[Infinity]] :=
    With[{\[Alpha] = \[Alpha]100 / 100},
        logq2 /. FindRoot[logq2 - \[Alpha] log\[Sigma]w - Log @ NExpectation[
            Exp[\[Alpha] Log @ Abs[\[Phi][z Exp[logq2 / \[Alpha]]]]], {z \[Distributed]
             StableDist[\[Alpha]]}], {logq2, 1}]
    ]

LogQStar[\[Alpha]100_?NumericQ, log\[Sigma]w_?NumericQ, -\[Infinity] 
    | Indeterminate, \[Phi]_, numSamples_] :=
    With[{z = GetStableSamples[\[Alpha]100, numSamples], \[Alpha] = \[Alpha]100
         / 100},
        logq2 /. FindRoot[logq2 - \[Alpha] log\[Sigma]w - LogAvgExp[\[Alpha]
             * Log @ Abs[\[Phi][z Exp[logq2 / \[Alpha]]]]], {logq2, 1}]
    ]

LogQStar[\[Alpha]100_?NumericQ, log\[Sigma]w_?NumericQ, log\[Sigma]b_
    ?NumericQ, \[Phi]_, \[Infinity]] :=
    With[{\[Alpha] = \[Alpha]100 / 100},
        LogSumExp[{logq2, \[Alpha] log\[Sigma]b}] /. FindRoot[logq2 -
             \[Alpha] log\[Sigma]w - Log @ NExpectation[Exp[\[Alpha] Log @ Abs[\[Phi][
            z Exp[LogSumExp[{logq2, \[Alpha] log\[Sigma]b}] / \[Alpha]]]]], {z \[Distributed]
             StableDist[\[Alpha]]}], {logq2, 1}]
    ]

LogQStar[\[Alpha]100_?NumericQ, log\[Sigma]w_?NumericQ, log\[Sigma]b_
    ?NumericQ, \[Phi]_, numSamples_] :=
    With[{z = GetStableSamples[\[Alpha]100, numSamples], \[Alpha] = \[Alpha]100
         / 100},
        LogSumExp[{logq2, \[Alpha] log\[Sigma]b}] /. FindRoot[logq2 -
             \[Alpha] log\[Sigma]w - LogAvgExp[\[Alpha] Log @ Abs[\[Phi][z * Exp[
            LogSumExp[{logq2, \[Alpha] log\[Sigma]b}] / \[Alpha]]]]], {logq2, 1}]
            
    ]

GetLogQStar[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, \[Phi]_, numSamples_,
     label_:"logqStar"] :=
    With[{paths = FileNames @ StringTemplate["``/``/``/``/``_``_*"][prefix,
         \[Alpha]100, \[Sigma]w100, \[Sigma]b100, label, numSamples]},
        If[Length @ paths > 0,
            Import[First @ paths, "Table"][[1, 1]]
            ,
            With[{log\[Sigma]w = Log[\[Sigma]w100 / 100], log\[Sigma]b
                 = Log[\[Sigma]b100 / 100]},
                    LogQStar[\[Alpha]100, log\[Sigma]w, log\[Sigma]b,
                         \[Phi], numSamples]
                ] //
                With[{fname = StringTemplate["``/``/``/``/``_``_``.txt"
                    ][prefix, \[Alpha]100, \[Sigma]w100, \[Sigma]b100, label, numSamples,
                     CreateUUID[]]},
                    Quiet[CreateDirectory @ DirectoryName @ fname, CreateDirectory
                        ::eexist];
                    Export[fname, #, "Table"];
                    #
                ]&
        ]
    ]

LogSech2[x_ ? (VectorQ[#, NumericQ]&)] :=
    2 (Log[2] + x - LogSumExp[{2 x, ConstantArray[0, Length @ x]}]);

LogSech2[x_?NumericQ] :=
    First @ LogSech[{x}]

LogY[\[Alpha]100_?NumericQ, logr_?NumericQ, log\[Chi]_ ? (VectorQ[#, 
    NumericQ]&), seed_:42] :=
    BlockRandom[
        SeedRandom[seed];
        With[{logS = RandomSample @ Log @ GetSSamples[\[Alpha]100, Length
             @ log\[Chi]], \[Alpha] = \[Alpha]100 / 100, logSp = RandomSample @ Log
             @ GetSSamples[\[Alpha]100, Length @ log\[Chi]]},
            logy /.
                FindRoot[
                    \[Alpha] logy - LogAvgExp[(\[Alpha] / 2) (2 log\[Chi]
                         + logS - LogSumExp[{ConstantArray[2 logr - 2 logy, Length @ log\[Chi]
                        ], 2 log\[Chi] + logS + logSp}])], {logy, 0}(*,EvaluationMonitor:>Print[logy]
                        
                        
                        
                        
                        
                        
                        
                        
                *) ]
        ]
    ]

(*if invCDFflag=False, logs is log the survival fn (i.e.log(1-cdf) instead of log the cdf*)

LogEigCDF[\[Alpha]100_?NumericQ, logr_?NumericQ, log\[Chi]_ ? (VectorQ[
    #, NumericQ]&), survivalFlag_:False, seed_:42] :=
    BlockRandom[
        SeedRandom[seed];
        With[{logS = RandomSample @ Log @ GetSSamples[\[Alpha]100, Length
             @ log\[Chi]], \[Alpha] = \[Alpha]100 / 100, logSp = RandomSample @ Log
             @ GetSSamples[\[Alpha]100, Length @ log\[Chi]], logy = LogY[\[Alpha]100,
             logr, log\[Chi], seed]},
            If[survivalFlag,
                LogAvgExp[2 log\[Chi] + logS + logSp - LogSumExp[{ConstantArray[
                    2 logr - 2 logy, Length @ log\[Chi]], 2 log\[Chi] + logS + logSp}]]
                ,
                2 logr - 2 logy + LogAvgExp[-LogSumExp[{ConstantArray[
                    2 logr - 2 logy, Length @ log\[Chi]], 2 log\[Chi] + logS + logSp}]]
            ]
        ]
    ]

LogRHat[\[Alpha]100_?NumericQ, logs_?NumericQ, log\[Chi]_ ? (VectorQ[
    #, NumericQ]&), survivalFlag_:False, seed_:42] :=
    BlockRandom[
        SeedRandom[seed];
        With[{logS = RandomSample @ Log @ GetSSamples[\[Alpha]100, Length
             @ log\[Chi]], logSp = RandomSample @ Log @ GetSSamples[\[Alpha]100, 
            Length @ log\[Chi]]},
            logrHat /.
                If[survivalFlag,
                    FindRoot[logs - LogAvgExp[2 log\[Chi] + logS + logSp
                         - LogSumExp[{ConstantArray[2 logrHat, Length @ log\[Chi]], 2 log\[Chi]
                         + logS + logSp}]], {logrHat, 0}]
                    ,
                    FindRoot[logs - 2 logrHat - LogAvgExp[-LogSumExp[
                        {ConstantArray[2 logrHat, Length @ log\[Chi]], 2 log\[Chi] + logS + logSp
                        }]], {logrHat, 0}]
                ]
        ]
    ]

LogInvCDF[\[Alpha]100_?NumericQ, logs_?NumericQ, log\[Chi]_ ? (VectorQ[
    #, NumericQ]&), survivalFlag_:False, seed_:42] :=
    BlockRandom[
        SeedRandom[seed];
        With[{logS = RandomSample @ Log @ GetSSamples[\[Alpha]100, Length
             @ log\[Chi]], logSp = RandomSample @ Log @ GetSSamples[\[Alpha]100, 
            Length @ log\[Chi]], logrHat = LogRHat[\[Alpha]100, logs, log\[Chi], 
            survivalFlag, seed], \[Alpha] = \[Alpha]100 / 100},
            logrHat + (1 / \[Alpha]) LogAvgExp[(\[Alpha] / 2) (2 log\[Chi]
                 + logS - LogSumExp[{ConstantArray[2 logrHat, Length @ log\[Chi]], 2 
                * log\[Chi] + logS + logSp}])]
        ]
    ]

PutJacobianLogInvCDF[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, numSamples_,
     seed_:42, label_:"loginvCDF"] :=
    Table[
            LogInvCDF[
                \[Alpha]100
                ,
                If[s < 0.5,
                    Log @ s
                    ,
                    Log[1 - s]
                ]
                ,
                Log\[Chi][\[Alpha]100, \[Sigma]w100, \[Sigma]b100, Tanh,
                     numSamples, LogSech2]
                ,
                s > 0.5
                ,
                seed
            ]
            ,
            {s, Most @ Rest @ Subdivide[numSamples + 1]}
        ] // Export[FileNameJoin @ {prefix, ToString @ \[Alpha]100, ToString
             @ \[Sigma]w100, ToString @ \[Sigma]b100, StringTemplate["``_``_``.txt"
            ][label, numSamples, seed]}, #, "Table"]&



Log\[Chi][\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, \[Phi]_, numSamples_,
     log\[Phi]p_] :=
    With[{\[Alpha] = \[Alpha]100 / 100, log\[Sigma]w = Log[\[Sigma]w100
         / 100]},
        log\[Sigma]w + log\[Phi]p[Exp[GetLogQStar[\[Alpha]100, \[Sigma]w100,
             \[Sigma]b100, \[Phi], numSamples] / \[Alpha]] GetStableSamples[\[Alpha]100,
             numSamples]]
    ]


(* EmpiricalH[\[Alpha]_, \[Sigma]w_, \[Sigma]b_, \[Phi]_, width_, depth_
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
    ] *)

(* Map functions *)

(* one should probably prefer more uniform samples with fewer stable samples to get an accurate average (but more stable samples are better for getting the shape of the CDF) *)

SavePuts[label_, fmt_:"Table", red_:Identity] :=
    GroupBy[FileNames @ FileNameJoin @ {prefix, StringTemplate["*/*/*/``_*.txt"
        ][label]}, (ToExpression @ FileNameSplit[#][[-4 ;; -2]]&) -> (Import[
        #, fmt]&), red] // Export[FileNameJoin @ {prefix, StringTemplate["``.mx"
        ][label]}, #]&

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

(* PutLogNeuralNorm[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, Infinity,
     prefix_:"fig/data"] :=
    With[{logfpStable = GetLogFPStable[\[Alpha]100, \[Sigma]w100, \[Sigma]b100
        ]},
            Log @ NExpectation[Tanh[Exp[logfpStable / (\[Alpha]100 / 
                100.)] h] ^ 2, {h \[Distributed] StableDist[\[Alpha]100 / 100]}]
        ] // Export[FileNameJoin @ {prefix, StringTemplate["``/``/``/logNeuralNorm_``.txt"
            ][\[Alpha]100, \[Sigma]w100, \[Sigma]b100, CreateUUID[]]}, #, "Table"
            ]&

PutLogNeuralNorm[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_, stableSamples_,
     prefix_:"fig/data"] :=
    With[{logfpStable = GetLogFPStable[\[Alpha]100, \[Sigma]w100, \[Sigma]b100
        ]},
            LogAvgExp[2 Log @ Abs @ Tanh[Exp[logfpStable / (\[Alpha]100
                 / 100.)] RandomVariate[StableDist[\[Alpha]100 / 100.], 1000]]]
        ] // Export[FileNameJoin @ {prefix, StringTemplate["``/``/``/logNeuralNorm_``.txt"
            ][\[Alpha]100, \[Sigma]w100, \[Sigma]b100, CreateUUID[]]}, #, "Table"
            ]&

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
            ]]}, #, "Table"]& *)

(*
    Usage: wolframscript -f filename.wl Function1[...] ... FunctionN[...]
    In unix, enclose the functions in double quotes to escape the spaces.
    To pass through strings to wolfram, escape the quotes.
*)

If[Length @ $ScriptCommandLine > 0,
    c = (Print @* EchoTiming @* ToExpression) /@ $ScriptCommandLine[[
        2 ;; ]]
]
