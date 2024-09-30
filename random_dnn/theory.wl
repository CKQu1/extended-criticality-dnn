(* in this wl file we have some functions that need many cores to run on in parallel *)

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


(* Stable distribution functions *)

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
        TransformedDistribution[1, {x \[Distributed] NormalDistribution[]}]
    ]

(* Work with positive quantities in log space *)

LogFPStable[\[Alpha]_?NumericQ, \[Sigma]w_?NumericQ, \[Sigma]b_?NumericQ,
     \[Phi]_] :=
    If[\[Sigma]w == 0,
        \[Alpha] Log @ \[Sigma]b
        ,
        Logq2 + \[Alpha] Log @ \[Sigma]w /. FindRoot[NExpectation[Abs[
            \[Phi][h Exp[Logq2 / \[Alpha]] \[Sigma]w]] ^ \[Alpha], {h \[Distributed]
             StableDist[\[Alpha]]}] + (\[Sigma]b / \[Sigma]w) ^ \[Alpha] - Exp[Logq2
            ], {Logq2, 0}]
    ]

(* stuff specific to the wl file *)

(* this is important to get exactly because everything else depends on the value of this *)

GetLogFPStable[] :=
    With[{\[Alpha]List = Range[1, 2, .05], gList = Rest @ Range[0, 3,
         .05]},
            <|"\[Alpha]List" -> \[Alpha]List, "gList" -> gList, "logFPStable"
                 -> ParallelOuter[Quiet @ LogFPStable[#1, #2, 0, Tanh]&, \[Alpha]List,
                 gList, Method -> "FinestGrained"]|>
        ] // Export["fig/fpStable.wxf", #]&

(* wolframscript -f theory.wl LaunchPhysicsKernels[] GetLogFPStable[] *)

GetNeuralNorms[] :=
    With[{data = Import["fig/fpStable.wxf"], \[Phi] = Tanh, ord = 2},
        
            ParallelOuterWithData[Quiet @ NExpectation[\[Phi][Exp[#1 
                / #2] h] ^ ord, {h \[Distributed] StableDist[#2]}]&, data["logFPStable"
                ], data["\[Alpha]List"], data["gList"]]
        ] // Export["fig/neuralNorms.wxf", #]&



(* cluster computing functions *)

LaunchPhysicsKernels[] :=
    (
        LaunchKernels[6];
        LaunchKernels["ssh://cartman?20"];
        (* LaunchKernels["ssh://bebe?16"]; *)
        LaunchKernels["ssh://stan?20"];
        Return[Length @ Kernels[]]
    )


c = (Print @* EchoTiming @* ToExpression) /@ $ScriptCommandLine[[2 ;; ]]
