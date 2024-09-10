<<"utils.wl"



NeuralNorm[\[Alpha]_?NumericQ, q_?NumericQ, \[Phi]_:Tanh, ord_:2, hList_
    :None] :=
    With[{
        hList2 =
            If[hList === None,
                RandomVariate[StableDist[\[Alpha]], 100000]
                ,
                hList
            ]
    },
        Mean[Abs[\[Phi][hList2 q ^ (1 / \[Alpha])]] ^ ord]
    ];

GetFPStable[] :=
    With[{\[Phi] = Tanh, \[Sigma]b = 0, \[Alpha]List = Range[1, 2, 0.05
        ], \[Sigma]wList = Range[0, 3, 0.1]},
            <|
                "\[Phi]" -> \[Phi]
                ,
                "\[Sigma]b" -> \[Sigma]b
                ,
                "\[Alpha]List" -> \[Alpha]List
                ,
                "\[Sigma]wList" -> \[Sigma]wList
                ,
                "data" ->
                    ParallelTable[
                        Mean @
                            Table[
                                With[{hList = RandomVariate[StableDist[
                                    \[Alpha]], 100000]},
                                    Table[FPStable[\[Alpha], \[Sigma]w,
                                         \[Sigma]b, \[Phi], hList], {\[Sigma]w, \[Sigma]wList}]
                                ]
                                ,
                                10
                            ]
                        ,
                        {\[Alpha], \[Alpha]List}
                    ]
            |>
        ] // Export["fig/fpStable.wxf", #]&;

GetNeuralNorms[] :=
    (
        fpStable = Import["fig/fpStable.wxf"];
        WaitAll @ MapThread[Apply[ParallelSubmit @ Mean @ Table[NeuralNorm[
            #2, #3, fpStable["\[Phi]"], 2], 10] ^ (1 / 2)&] @* Append, {Outer[List,
             fpStable["\[Sigma]wList"], fpStable["\[Alpha]List"]], Transpose @ fpStable[
            "data"]}, 2] // Export["fig/neuralNorms.wxf", #]&
    )

PlotNeuralNorms[] :=
    (
        fpStable = Import["fig/fpStable.wxf"];
        neuralNorms = Import["fig/neuralNorms.wxf"];
        ListDensityPlot[neuralNorms, PlotLegends -> Automatic, PlotRange
             -> All, ColorFunction -> "BlueGreenYellow", FrameLabel -> {"\[Alpha]",
             "\!\(\*SubscriptBox[\(\[Sigma]\), \(w\)]\)"}, DataRange -> {MinMax @
             fpStable["\[Alpha]List"], MinMax @ fpStable["\[Sigma]wList"]}, ImageSize
             -> Small] ~ Show ~ ListContourPlot[neuralNorms, PlotLegends -> Automatic,
             PlotRange -> All, ColorFunction -> "BlueGreenYellow", FrameLabel -> 
            {"\[Alpha]", "\!\(\*SubscriptBox[\(\[Sigma]\), \(w\)]\)"}, DataRange 
            -> {MinMax @ fpStable["\[Alpha]List"], MinMax @ fpStable["\[Sigma]wList"
            ]}, ImageSize -> Small, Contours -> {0.01}, ContourShading -> None, ContourStyle
             -> Red] // Export["fig/neuralNorms.png", #]&;
        ListDensityPlot[Log @ neuralNorms, PlotLegends -> Automatic, 
            PlotRange -> All, ColorFunction -> "BlueGreenYellow", FrameLabel -> {
            "\[Alpha]", "\!\(\*SubscriptBox[\(\[Sigma]\), \(w\)]\)"}, DataRange ->
             {MinMax @ fpStable["\[Alpha]List"], MinMax @ fpStable["\[Sigma]wList"
            ]}, ImageSize -> Small] ~ Show ~ ListContourPlot[neuralNorms, PlotLegends
             -> Automatic, PlotRange -> All, ColorFunction -> "BlueGreenYellow", 
            FrameLabel -> {"\[Alpha]", "\!\(\*SubscriptBox[\(\[Sigma]\), \(w\)]\)"
            }, DataRange -> {MinMax @ fpStable["\[Alpha]List"], MinMax @ fpStable[
            "\[Sigma]wList"]}, ImageSize -> Small, Contours -> {0.01}, ContourShading
             -> None, ContourStyle -> Red] // Export["fig/neuralNorms_log.png", #
            ]&
    )

(* mixed selectivity plots using cifar-10 *)

IterNorm[\[Alpha]_?NumericQ, \[Sigma]w_?NumericQ, \[Sigma]b_?NumericQ,
     \[Phi]_, x_, nIters_, hList_:None, ord_:2] :=
    With[{
        hList2 =
            If[hList === None,
                RandomVariate[StableDist[\[Alpha]], 100000]
                ,
                hList
            ]
    },
        Prepend[Mean[Abs[x] ^ 2]] @ Map[Mean[Abs[Tanh[hList2 # ^ (1 /
             \[Alpha])]] ^ ord]&, NestList[\[Sigma]w^\[Alpha] Mean[Abs[\[Phi][hList2
             * Abs[#] ^ (1 / \[Alpha])]] ^ \[Alpha]] + \[Sigma]b^\[Alpha]&, \[Sigma]w
            ^\[Alpha] Mean[Abs[x] ^ \[Alpha]] + \[Sigma]b^\[Alpha], nIters]]
    ]

(* this is vectorised so no need for paralleltables *)

GetIterNormsDiffNorms[nSamples_:1000] :=
    (
        resource = ResourceObject["CIFAR-10"];
        trainingData = ResourceData[resource, "TrainingData"];
        testData = ResourceData[resource, "TestData"];
        Table[
                With[{hList = RandomVariate[StableDist[\[Alpha]], 10000
                    ]},
                    Table[IterNorm[\[Alpha], 1.5, 0, Tanh, x, 10, hList
                        ], {x, (Sqrt[RandomChoice[{.25, .75}]] Flatten @ ImageData @ # / Mean[
                        Abs[Flatten @ ImageData @ #] ^ 2] ^ .5)& /@ RandomSample[trainingData
                        [[ ;; , 1]], nSamples]}]
                ]
                ,
                {\[Alpha], 2, 1, -.25}
            ] // Export["fig/iterNormDiffNorms.wxf", #]&
    )

GetIterNormsDiffGains[nSamples_:1000] :=
    (
        resource = ResourceObject["CIFAR-10"];
        trainingData = ResourceData[resource, "TrainingData"];
        testData = ResourceData[resource, "TestData"];
        Table[
                With[{hList = RandomVariate[StableDist[\[Alpha]], 10000
                    ]},
                    Table[IterNorm[\[Alpha], RandomChoice[{5, 2, 1, 0.5
                        }], 0, Tanh, x, 10, hList], {x, (Sqrt[0.5] Flatten @ ImageData @ # / 
                        Mean[Abs[Flatten @ ImageData @ #] ^ 2] ^ .5)& /@ RandomSample[trainingData
                        [[ ;; , 1]], nSamples]}]
                ]
                ,
                {\[Alpha], 2, 1, -.25}
            ] // Export["fig/iterNormDiffGains.wxf", #]&
    )

PlotIterNorms[fname_] :=
    (
        iterNormDiffNorms = Import[fname];
        iterNormDiffNorms //
        Show[ListLinePlot[#, PlotStyle -> Thin, PlotRange -> All]& /@
             Transpose @ #, PlotRange -> All, Frame -> True, FrameLabel -> {"layer",
             "<\!\(\*SuperscriptBox[\(\[Phi]\), \(2\)]\)>"}]& //
        Export[FileBaseName[fname] <> ".png", #]&
    )

(* Use this script as
    wolframscript -f filename.wl Function1[...] ... FunctionN[...]
and it will run and echo the functions in order
*)

c = (Print @* EchoTiming @* ToExpression) /@ $ScriptCommandLine[[2 ;; ]]

(* 

fig 1:
    wolframscript -f theory_MFT.wl LaunchPhysicsKernels[] GetFPStable[] GetNeuralNorms[] PlotNeuralNorms[]

fig 2:
    wolframscript -f theory_MFT.wl GetIterNormsDiffNorms[1000] PlotIterNorms[\"fig/iterNormDiffNorms.wxf\"] GetIterNormsDiffGains[1000] PlotIterNorms[\"fig/iterNormDiffGains.wxf\"]

 *)
