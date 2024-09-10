<<"utils.wl"

JacobianCharSpectralRadius[\[Alpha]_?NumericQ, g_?NumericQ, ngt_?NumericQ,
     \[Phi]_:Tanh, nSamples_:100000] :=
    With[{hList = RandomVariate[StableDist[\[Alpha]], nSamples]},
        With[{\[Chi]Samples = g \[Phi]'[FPStable[\[Alpha], g, 0, \[Phi],
             hList] ^ (1 / \[Alpha]) hList]},
            CharSpectralRadius[\[Alpha], ngt, \[Chi]Samples]
        ]
    ]

CharSpectralRadius[\[Alpha]_?NumericQ, ngt_?NumericQ, \[Chi]Samples_] :=
    With[{SDistSamples = RandomVariate[SDist[\[Alpha]], Length @ \[Chi]Samples
        ]},
        With[{\[Chi]2S = \[Chi]Samples^2 SDistSamples, Sp = RandomSample
             @ SDistSamples},
            With[{rym1 = Abs[rym1] /. FindRoot[ngt - Mean[(\[Chi]2S Sp
                ) / (rym1^2 + \[Chi]2S Sp)], {rym1, 1*^-2, 1*^3}]},
                rym1 Mean[(\[Chi]2S / (rym1^2 + \[Chi]2S Sp)) ^ (\[Alpha]
                     / 2)] ^ (1 / \[Alpha])
            ]
        ]
    ]

JacobianNVector[\[Alpha]_?NumericQ, g_?NumericQ, rList_, \[Phi]_:Tanh,
     nSamples_:100000] :=
    With[{hList = RandomVariate[StableDist[\[Alpha]], nSamples]},
        With[{\[Chi]Samples = g \[Phi]'[FPStable[\[Alpha], g, 0, \[Phi],
             hList] ^ (1 / \[Alpha]) hList]},
            nVector[\[Alpha], \[Chi]Samples, rList]
        ]
    ]

nVector[\[Alpha]_?NumericQ, \[Chi]Samples_, rList_] :=
    With[{SDistSamples = RandomVariate[SDist[\[Alpha]], Length @ \[Chi]Samples
        ]},
        With[{\[Chi]2S = \[Chi]Samples^2 SDistSamples, Sp = RandomSample
             @ SDistSamples},
            Table[
                With[{y = Abs[y] /. FindRoot[y^\[Alpha] - Mean[(\[Chi]2S
                     / ((r / y) ^ 2 + \[Chi]2S Sp)) ^ (\[Alpha] / 2)], {y, 1, 0.5}]},
                    Mean[(\[Chi]2S Sp) / ((r / y) ^ 2 + \[Chi]2S Sp)]
                        
                ]
                ,
                {r, rList}
            ]
        ]
    ]

(* save jacNVec *)

GetJacNVecs[] :=
    (
        With[{rList = Rest @ Range[0, 10, 0.01], \[Alpha]List = Range[
            1, 1.9, .1] // Append[1.99], gList = Rest @ Range[0, 3, .1]},
                <|"rList" -> rList, "\[Alpha]List" -> \[Alpha]List, "gList"
                     -> gList, "jacNVec" -> ParallelOuter[Quiet @ JacobianNVector[#1, #2,
                     rList, Tanh, 100000]&, \[Alpha]List, gList]|>
            ] // Export["fig/jacNVec.wxf", #]&
    )

c = (Print @* EchoTiming @* ToExpression) /@ $ScriptCommandLine[[2 ;; ]]