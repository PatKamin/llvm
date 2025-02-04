; RUN: mlir-translate --import-llvm %s | FileCheck %s
; CHECK: %[[ROOT:.+]] = llvm.mlir.undef : !llvm.struct<"SimpleAggType", (i32, i8, i16, i32)>
; CHECK: %[[C0:.+]] = llvm.mlir.constant(9 : i32) : i32
; CHECK: %[[CHAIN0:.+]] = llvm.insertvalue %[[C0]], %[[ROOT]][0]
; CHECK: %[[C1:.+]] = llvm.mlir.constant(4 : i8) : i8
; CHECK: %[[CHAIN1:.+]] = llvm.insertvalue %[[C1]], %[[CHAIN0]][1]
; CHECK: %[[C2:.+]] = llvm.mlir.constant(8 : i16) : i16
; CHECK: %[[CHAIN2:.+]] = llvm.insertvalue %[[C2]], %[[CHAIN1]][2]
; CHECK: %[[C3:.+]] = llvm.mlir.constant(7 : i32) : i32
; CHECK: %[[CHAIN3:.+]] = llvm.insertvalue %[[C3]], %[[CHAIN2]][3]
; CHECK: llvm.return %[[CHAIN3]]
%SimpleAggType = type {i32, i8, i16, i32}
@simpleAgg = global %SimpleAggType {i32 9, i8 4, i16 8, i32 7}

; CHECK: %[[ROOT:.+]] = llvm.mlir.undef : !llvm.struct<"NestedAggType", (struct<"SimpleAggType", (i32, i8, i16, i32)>, ptr<struct<"SimpleAggType", (i32, i8, i16, i32)>>)>
; CHECK: %[[NESTED:.+]] = llvm.mlir.undef : !llvm.struct<"SimpleAggType", (i32, i8, i16, i32)>
; CHECK: %[[C1:.+]] = llvm.mlir.constant(1 : i32) : i32
; CHECK: %[[CHAIN0:.+]] = llvm.insertvalue %[[C1]], %[[NESTED]][0]
; CHECK: %[[C2:.+]] = llvm.mlir.constant(2 : i8) : i8
; CHECK: %[[CHAIN1:.+]] = llvm.insertvalue %[[C2]], %[[CHAIN0]][1]
; CHECK: %[[C3:.+]] = llvm.mlir.constant(3 : i16) : i16
; CHECK: %[[CHAIN2:.+]] = llvm.insertvalue %[[C3]], %[[CHAIN1]][2]
; CHECK: %[[C4:.+]] = llvm.mlir.constant(4 : i32) : i32
; CHECK: %[[CHAIN3:.+]] = llvm.insertvalue %[[C4]], %[[CHAIN2]][3]
; CHECK: %[[CHAIN4:.+]] = llvm.insertvalue %[[CHAIN3]], %[[ROOT]][0]
; CHECK: %[[NP:.+]] = llvm.mlir.null : !llvm.ptr<struct<"SimpleAggType", (i32, i8, i16, i32)>>
; CHECK: %[[CHAIN5:.+]] = llvm.insertvalue %[[NP]], %[[CHAIN4]][1]
; CHECK: llvm.return %[[CHAIN5]]
%NestedAggType = type {%SimpleAggType, %SimpleAggType*}
@nestedAgg = global %NestedAggType { %SimpleAggType{i32 1, i8 2, i16 3, i32 4}, %SimpleAggType* null }

; CHECK: %[[ROOT:.+]] = llvm.mlir.undef : !llvm.vec<2 x ptr<struct<"SimpleAggType", (i32, i8, i16, i32)>>>
; CHECK: %[[C0:.+]] = llvm.mlir.null : !llvm.ptr<struct<"SimpleAggType", (i32, i8, i16, i32)>>
; CHECK: %[[P0:.+]] = llvm.mlir.constant(0 : i32) : i32
; CHECK: %[[CHAIN0:.+]] = llvm.insertelement %[[C0]], %[[ROOT]][%[[P0]] : i32] : !llvm.vec<2 x ptr<struct<"SimpleAggType", (i32, i8, i16, i32)>>>
; CHECK: %[[C1:.+]] = llvm.mlir.null : !llvm.ptr<struct<"SimpleAggType", (i32, i8, i16, i32)>>
; CHECK: %[[P1:.+]] = llvm.mlir.constant(1 : i32) : i32
; CHECK: %[[CHAIN1:.+]] = llvm.insertelement %[[C1]], %[[CHAIN0]][%[[P1]] : i32] : !llvm.vec<2 x ptr<struct<"SimpleAggType", (i32, i8, i16, i32)>>>
; CHECK: llvm.return %[[CHAIN1]] : !llvm.vec<2 x ptr<struct<"SimpleAggType", (i32, i8, i16, i32)>>>
@vectorAgg = global <2 x %SimpleAggType*> <%SimpleAggType* null, %SimpleAggType* null>
