prefix = "fig/data";

(* Compute functions *)

ResolventPDF[G1_, G2_] :=
    (Total[Im @ G1] + Total[Im @ G2]) / (\[Pi] Length[G2])

SingularValueResolvent[\[Alpha]_?NumericQ, \[Chi]Dist_, \[Chi]Fn_, \[Sigma]_
    ?NumericQ, G1Init_ /; NumericQ[G1Init] || VectorQ[G1Init, NumericQ], 
    G2Init_ /; NumericQ[G2Init] || VectorQ[G2Init, NumericQ], numSteps_?NumericQ,
     numUpdates_:0] :=
    Module[{
        G1 =
            If[NumericQ[G1Init],
                RandomComplex[{-1 - I, 1 + I}, G1Init]
                ,
                G1Init
            ]
        ,
        G2 =
            If[NumericQ[G2Init],
                RandomComplex[{-1 - I, 1 + I}, G2Init]
                ,
                G2Init
            ]
    },
        With[{S = StableDistribution[\[Alpha], 0, 0, (Length @ G1 + Length
             @ G2) ^ (-1 / \[Alpha])], \[Chi]SqSamples = \[Chi]Fn[RandomVariate[\[Chi]Dist,
             Length @ G2]] ^ 2},
            Do[
                With[{g1i = 1 + Mod[step, Length @ G1], g2j = 1 + Mod[
                    step, Length @ G2]},
                    G1[[g1i]] = 1 / (-\[Sigma] - (RandomVariate[S, Length
                         @ G2] ^ 2) . (\[Chi]SqSamples G2));
                    G2[[g2j]] = 1 / (-\[Sigma] - \[Chi]SqSamples[[g2j
                        ]] (RandomVariate[S, Length @ G1] ^ 2) . G1);
                    If[numUpdates > 0 && Mod[step, IntegerPart[numSteps
                         / numUpdates]] === 0,
                        Print[{step, ResolventPDF[G1, G2]}]
                    ]
                ]
                ,
                {step, numSteps}
            ];
            {G1, G2}
        ]
    ]

QMap[\[Alpha]_?NumericQ, \[Sigma]w_?NumericQ, \[Sigma]b_?NumericQ, \[Phi]_,
     qInit : (_?NumericQ) : 3.0, opts : OptionsPattern[{NExpectation, MaxIterations
     -> 500, Tolerance -> 1*^-9}]] :=
    FixedPointList[q |-> \[Sigma]w^\[Alpha] NExpectation[Abs[\[Phi][q
         ^ (1 / \[Alpha]) z]] ^ \[Alpha], z \[Distributed] StableDistribution[
        \[Alpha], 0, 0, 2 ^ (-1 / \[Alpha])], Sequence @@ FilterRules[{opts},
         Options[NExpectation]]] + \[Sigma]b^\[Alpha], qInit, OptionValue[MaxIterations
        ], SameTest -> (Abs[#1 - #2] < OptionValue[Tolerance]&)]

(* Put functions *)

PutLogQMap[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_] :=
    With[{q = Log @ QMap[\[Alpha]100 / 100., \[Sigma]w100 / 100., \[Sigma]b100
         / 100., Tanh]},
        Export[FileNameJoin @ {prefix, ToString @ \[Alpha]100, ToString
             @ \[Sigma]w100, ToString @ \[Sigma]b100, "logQMap.txt"}, q, "Table"]
            
    ]

PutSingularValueResolventPDF[\[Alpha]100_, \[Sigma]w100_, \[Sigma]b100_,
     logx_?NumericQ, nIters_, label_:"cavityPopSVlogpdf"] :=
    With[{logQStar = Import[FileNameJoin[{prefix, ToString @ \[Alpha]100,
         ToString @ \[Sigma]w100, ToString @ \[Sigma]b100, "logQMap.txt"}], "Table"
        ][[-1, 1]]},
            With[{\[Chi]Dist = StableDistribution[\[Alpha], 0, 0, 2 ^
                 (-1 / \[Alpha])], \[Chi]Fn = \[Eta] |-> \[Sigma]w Tanh'[Exp[logQStar
                 / \[Alpha]] \[Eta]]},
                ResolventPDF @@ Nest[SingularValueResolvent[\[Alpha],
                     \[Chi]Dist, \[Chi]Fn, Exp[logx] + 1 / (Length @ #[[2]]) I, ArrayPad[
                    #[[1]], {0, Length @ #[[1]]}, "Periodic"], ArrayPad[#[[2]], {0, Length
                     @ #[[2]]}, "Periodic"], Length @ #[[1]] ^ 2]&, {RandomComplex[{-1, 1
                     + I}, 4], RandomComplex[{-1, 1 + I}, 4]}, nIters]
            ]
        ] // Export[FileNameJoin @ {prefix, ToString @ \[Alpha]100, ToString
             @ \[Sigma]w100, ToString @ \[Sigma]b100, StringTemplate["``_``_``.txt"
            ][label, nIters, logx]}, {logx, Log @ #}, "Table"]&

(* By default SavePuts[] gathers together all the files with the same label into a single key. But if red is None then separate the stems and do not gather *)

SavePuts[label_, red_:Identity, fmt_:"Table"] :=
    (Export[FileNameJoin[{prefix, StringTemplate["``.mx"][label]}], #1
        ]&)[
        GroupBy[
            FileNames[FileNameJoin[{prefix, StringTemplate["*/*/*/``_*.txt"
                ][label]}]]
            ,
            (
                    ToExpression[
                        FileNameSplit[#1][[-4 ;; -2]] ~ Join ~
                            If[red === None,
                                StringSplit[FileBaseName[#1], "_"][[2
                                     ;; ]]
                                ,
                                {}
                            ]
                    ]&
                ) -> (Import[#1, fmt]&)
            ,
            If[red === None,
                First
                ,
                red
            ]
        ]
    ]

(*
    Usage: wolframscript -f filename.wl Function1[...] ... FunctionN[...]
    In unix, enclose the functions in double quotes to escape the spaces.
    To pass through strings to wolfram, escape the quotes.
*)

If[Length @ $ScriptCommandLine > 0,
    c = (Print @* EchoTiming @* ToExpression) /@ $ScriptCommandLine[[
        2 ;; ]]
]
