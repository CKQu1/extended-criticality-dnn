#!/usr/bin/env wolframscript
(* ::Package:: *)

PrintArgs[args__] := Print@{args}



JacobianDOSVectorSave[alpha100_, g100_] := 
 With[{fname = 
    StringTemplate["theory_dnn/DOS_alpha_``_g_``.dat"][alpha100, 
     g100]}, If[Not@MemberQ[FileNames["*.dat", "theory_dnn"], fname], 
   Put[BinarySerialize@
     If[alpha100 == 200, JacobianDOSGaussianVector[g100/100.], 
      JacobianDOSVector[alpha100/100., g100/100., 0.01]], fname]]]

JacobianDOSVector[\[Alpha]_?NumericQ, g_?NumericQ, 
   yFrac_?NumericQ, \[Phi]_: Tanh, 
   nSamples_: 100000, \[Delta]r_: 0.01] := 
  With[{\[Chi]Fn = g \[Phi]'[#] &, \[Chi]Dist = 
     StableDistribution[\[Alpha], 0, 0, 
      g (Abs[s]/2)^(1/\[Alpha]) /. 
       FindRoot[
        Abs[s] - 
         NExpectation[Abs[\[Phi][h]]^\[Alpha], 
          h \[Distributed] 
           StableDistribution[\[Alpha], 0, 0, 
            g (Abs[s]/2)^(1/\[Alpha])]], {s, 1, .5}]]}, 
   DOSVector[\[Alpha], yFrac, \[Chi]Dist, \[Chi]Fn, 
    nSamples, \[Delta]r]];

DOSVector[\[Alpha]_?NumericQ, yFrac_?NumericQ, \[Chi]Dist_, \[Chi]Fn_,
   nSamples_: 100000, \[Delta]r_: .01] := 
 With[{SDistSamples = RandomVariate[SDist[\[Alpha]], nSamples], 
   y0 = ((
     4 Cos[(\[Pi] \[Alpha])/4] Csc[(\[Pi] \[Alpha])/
       2] Sin[(\[Pi] \[Alpha])/4])/Gamma[1 + \[Alpha]])^(1/\[Alpha])},
   With[{samples = {SDistSamples, 
      RandomSample@SDistSamples, \[Chi]Fn@
       RandomVariate[\[Chi]Dist, nSamples]}}, 
   Table[r~List~
     With[{y = 
        Abs[y] /. 
         FindRoot[
          1 - Mean[((#3^2 #1)/(r^2 + #3^2 y^2 #1 #2))^(\[Alpha]/2) &~
             MapThread~samples], {y, 1, 0.5}]}, 
      With[{\[Delta]y = -(
          Mean[(#3^2 #1)^(\[Alpha]/
              2)/(r^2 + #3^2 y^2 #1 #2)^(\[Alpha]/2 + 1) &~MapThread~
            samples]/
          Mean[((#3^2 #1)/(r^2 + #3^2 y^2 #1 #2))^(\[Alpha]/2 + 1)
               2 y #2 &~MapThread~samples])}, (y^2 - 
          2 r^2 y \[Delta]y)/\[Pi] Mean[(#3^2 #1 #2)/(r^2 + #3^2 y^2 \
#1 #2)^2 &~MapThread~samples]]], {r, \[Delta]r, 
     Abs@z /. 
      Quiet@FindRoot[
        1 - Mean[((#3^2 #1)/(
             z^2 + #3^2 (yFrac y0)^2 #1 #2))^(\[Alpha]/2) &~MapThread~
           samples], {z, 1, 2}], \[Delta]r}]]]

JacobianOrderedTransitionSave[alpha100_, nReps_: 1] := 
 With[{fname = 
    StringTemplate["theory_dnn/transition_alpha_``.dat"][alpha100]}, 
  If[Not@MemberQ[FileNames["transition_*.dat", "theory_dnn"], fname], 
   Put[BinarySerialize@
     If[alpha100 == 200, {1, 
       Table[{r, N[1/\[Pi]]}, {r, .01, .99, .01}]}, 
      With[{g = 
         JacobianOrderedTransition[alpha100/100., 0.01, Tanh, 
          nReps]}, {g, 
        JacobianDOSVector[alpha100/100., g, 0.01, Tanh]}]], fname]]]

JacobianOrderedTransitionEqnRHS[\[Alpha]_?NumericQ, 
  yFrac_?NumericQ, \[Phi]_, g_?NumericQ, nReps_: 1] := 
 With[{y0 = ((
     4 Cos[(\[Pi] \[Alpha])/4] Csc[(\[Pi] \[Alpha])/
       2] Sin[(\[Pi] \[Alpha])/4])/Gamma[1 + \[Alpha]])^(
    1/\[Alpha]), \[Chi]Fn = g \[Phi]'[#] &, \[Chi]Dist = 
    StableDistribution[\[Alpha], 0, 0, 
     g (Abs[s]/2)^(1/\[Alpha]) /.(*FindRoot[Abs[s]-NExpectation[
      Abs[\[Phi][h]]^\[Alpha],
      h\[Distributed]StableDistribution[\[Alpha],0,0,g(Abs[s]/2)^(
      1/\[Alpha])]],{s,1,.5}]*)
      With[{samples = 
         RandomVariate[
          StableDistribution[\[Alpha], 0, 0, (1/2)^(1/\[Alpha])], 
          10000]}, 
       FindRoot[
        Abs[s] - 
         Mean[Abs[\[Phi][g Abs[s]^(1/\[Alpha]) #]]^\[Alpha] &~Map~
           samples], {s, 1, .5}]]]}, 
  Mean@Table[
    NExpectation[((\[Chi]Fn[\[Chi]]^2 S)/(
      1 + (yFrac y0)^2 \[Chi]Fn[\[Chi]]^2 S Sp))^(\[Alpha]/
      2), {S \[Distributed] SDist[\[Alpha]], 
      Sp \[Distributed] 
       SDist[\[Alpha]], \[Chi] \[Distributed] \[Chi]Dist}, 
     Method -> "MonteCarlo"], nReps](*This has poor accuracy: 
  With[{nSamples=100000},With[{samples={RandomVariate[SDist[\[Alpha]],
  nSamples],RandomVariate[SDist[\[Alpha]],nSamples],\[Chi]Fn@
  RandomVariate[\[Chi]Dist,nSamples]}},Mean[((#3^2#1)/(
  1+#3^2(yFrac y0)^2#1 #2))^(\[Alpha]/2)&~MapThread~samples]]]*)]; 
JacobianOrderedTransition[\[Alpha]_?NumericQ, 
  yFrac_?NumericQ, \[Phi]_: Tanh, nReps_: 1] := 
 Abs@g /. FindRoot[
   1 - JacobianOrderedTransitionEqnRHS[\[Alpha], yFrac, \[Phi], Abs@g,
      nReps], {g, 1, .5}]

SDist[\[Alpha]_] := 
 With[{C = 
    Function[\[Alpha]1, 
     Gamma[1 + \[Alpha]1] Sin[\[Pi] \[Alpha]1/2]/\[Pi]]}, 
  StableDistribution[\[Alpha]/2, 1, 
   0, (C[\[Alpha]]/(4 C[\[Alpha]/2]))^(2/\[Alpha])]]

ClearAll[yStarGaussian]; 
yStarGaussian[z_?NumericQ, \[Chi]Dist_, \[Chi]Fn_] := 
 With[{\[Alpha] = 2, S = 1, Sp = 1}, 
  Abs@yStar /. 
   FindRoot[
    1 - NExpectation[((\[Chi]Fn[\[Chi]]^2 S)/(
       z^2 + yStar^2 \[Chi]Fn[\[Chi]]^2 S Sp))^(\[Alpha]/
       2), \[Chi] \[Distributed] \[Chi]Dist], {yStar, 1, .5}]]

ClearAll[DOSGaussian]; 
DOSGaussian[z_?NumericQ, \[Chi]Dist_, \[Chi]Fn_] := 
 With[{yStar = yStarGaussian[z, \[Chi]Dist, \[Chi]Fn], \[Alpha] = 2, 
   S = 1, Sp = 1}, 
  With[{\[Delta]yStar = -(NExpectation[(\[Chi]Fn[\[Chi]]^2 S)^(\
\[Alpha]/2)/(z^2 + yStar^2 \[Chi]Fn[\[Chi]]^2 S Sp)^(\[Alpha]/2 + 
          1), \[Chi] \[Distributed] \[Chi]Dist]/
        NExpectation[((\[Chi]Fn[\[Chi]]^2 S)^(\[Alpha]/2 + 1)
           2 yStar Sp)/(z^2 + 
           yStar^2 \[Chi]Fn[\[Chi]]^2 S Sp)^(\[Alpha]/2 + 
          1), \[Chi] \[Distributed] \[Chi]Dist])}, (yStar^2 - 
      2 z^2 yStar \[Delta]yStar)/\[Pi] \
NExpectation[(\[Chi]Fn[\[Chi]]^2 S Sp)/(z^2 + 
       yStar^2 \[Chi]Fn[\[Chi]]^2 S Sp)^2, \[Chi] \[Distributed] \
\[Chi]Dist]]]

ClearAll[JacobianDOSGaussianVector]; 
JacobianDOSGaussianVector[
  g_?NumericQ, \[Phi]_: Tanh, \[Delta]r_: 0.01] := 
 With[{\[Alpha] = 2}, 
  With[{\[Chi]Fn = g \[Phi]'[#] &, \[Chi]Dist = 
     StableDistribution[\[Alpha], 0, 0, 
      g (Abs[s]/2)^(1/\[Alpha]) /. 
       FindRoot[
        Abs[s] - 
         NExpectation[Abs[\[Phi][h]]^\[Alpha], 
          h \[Distributed] 
           StableDistribution[\[Alpha], 0, 0, 
            g (Abs[s]/2)^(1/\[Alpha])]], {s, 1, .5}]]}, 
   Table[{z, DOSGaussian[z, \[Chi]Dist, \[Chi]Fn]}, {z, \[Delta]r, 
     NExpectation[\[Chi]Fn[\[Chi]]^2, \[Chi] \[Distributed] \
\[Chi]Dist]^(1/2), \[Delta]r}]]];




helpstring = "
    Run this script as one of:
        math -script filename.wls FunctionName arg1 ... argN
        wolfram -script filename.wls FunctionName arg1 ... argN
        
    Arguments are passed in as expressions. Since bash consumes the outermost
    \"\", '' in the arguments, to pass in a string, use '\"arg\"' (enclose \"\"
    in '').
    Parentheses and most multi-element function expressions should also be
    enclosed in '', e.g.
        math -script file.wls '(Map@Head)@*List' '\"nice\"' '\"d\"' 1 symbol
        math -script nice.wls '#^2&' 23
";


If[Length@$CommandLine < 4,
    Print[helpstring];
    Quit[];
]

With[{c = ToExpression@$CommandLine[[4;;]]},
    Print[c[[1]]@@c[[2;;]]]
]
