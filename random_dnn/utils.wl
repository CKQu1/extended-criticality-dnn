ParallelOuter = ResourceFunction["ParallelOuter"];

ParallelOuterWithData[f_, data_, args__, 
  opts : OptionsPattern[ParallelMap]] := 
 With[{fullData = 
    MapThread[Inactive[Identity]@*Append, {Outer[List, args], data}, 
     2]}, Activate@
   ArrayReshape[
    ParallelMap[Inactive[Identity]@*Apply[f], 
     Activate@Flatten@fullData, {Length@Dimensions@fullData - 1}, 
     opts], Dimensions@fullData]]

(* This gets q, i.e. \[Sigma]w^\[Alpha] * <|\[Phi](h)|^\[Alpha]> + \[Sigma]b^\[Alpha] *)

(* FPStable[\[Alpha]_?NumericQ, \[Sigma]w_?NumericQ, \[Sigma]b_?NumericQ,
     \[Phi]_:Tanh, hList_:None] :=
    With[{
        hList2 =
            If[hList === None,
                RandomVariate[StableDist[\[Alpha]], 100000]
                ,
                hList
            ]
    },
        q /. FindRoot[\[Sigma]w^\[Alpha] Mean[Abs[\[Phi][hList2 Abs[q
            ] ^ (1 / \[Alpha])]] ^ \[Alpha]] + \[Sigma]b^\[Alpha] - q, {q, 1*^-2,
             1*^3}]
    ]; *)


FPStable[\[Alpha]_?NumericQ, \[Sigma]w_?NumericQ, \[Sigma]b_?NumericQ,
     \[Phi]_, stableSamples_] :=
    If[\[Sigma]w == 0,
        \[Sigma]b^\[Alpha]
        ,
        With[{
            hList =
                If[IntegerQ @ stableSamples,
                    RandomVariate[StableDist[\[Alpha]], stableSamples
                        ]
                    ,
                    stableSamples
                ]
        },
            Abs @ q2 \[Sigma]w^\[Alpha] /. FindRoot[Mean[Abs[\[Phi][hList
                 Abs[q2] ^ (1 / \[Alpha]) \[Sigma]w]] ^ \[Alpha]] + (\[Sigma]b / \[Sigma]w
                ) ^ \[Alpha] - Abs @ q2, {q2, 1., .5}]
        ]
    ]

FPStable[\[Alpha]_?NumericQ, \[Sigma]w_?NumericQ, \[Sigma]b_?NumericQ,
     \[Phi]_] :=
    If[\[Sigma]w == 0,
        \[Sigma]b^\[Alpha]
        ,
        Abs @ q2 \[Sigma]w^\[Alpha] /. FindRoot[NExpectation[Abs[\[Phi][
            h Abs[q2] ^ (1 / \[Alpha]) \[Sigma]w]] ^ \[Alpha], {h \[Distributed] 
            StableDist[\[Alpha]]}] + (\[Sigma]b / \[Sigma]w) ^ \[Alpha] - Abs @ q2,
             {q2, 1., .5}]
    ]

LaunchPhysicsKernels[] :=
    (
        LaunchKernels[6];
        LaunchKernels["ssh://cartman?20"];
        LaunchKernels["ssh://bebe?20"];
        LaunchKernels["ssh://stan?10"];
        Return[Length @ Kernels[]]
    )

StableDist[\[Alpha]_] :=
    StableDistribution[\[Alpha], 0, 0, 2 ^ (-1 / \[Alpha])]

SDist[\[Alpha]_?NumericQ] :=
    If[\[Alpha] < 2,
        With[{
            c =
                Function[\[Alpha]1,
                    Gamma[1 + \[Alpha]1] Sin[\[Pi] \[Alpha]1 / 2] / \[Pi]
                        
                ]
        },
            StableDistribution[\[Alpha] / 2, 1, 0, (c[\[Alpha]] / (4 
                c[\[Alpha] / 2])) ^ (2 / \[Alpha])]
        ]
        ,
        EmpiricalDistribution[{1}]
    ]
