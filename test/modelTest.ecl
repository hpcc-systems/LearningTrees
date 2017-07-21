
IMPORT $.datasets.CovTypeDS;
IMPORT $.^ AS LT;
IMPORT LT.internal AS int;
IMPORT LT.LT_Types;
IMPORT ML_Core;
IMPORT ML_Core.Types;

numTrees := 10;
maxDepth := 255;
numFeatures := 7;

numWorkItems := 1;

t_Discrete := Types.t_Discrete;
t_FieldReal := Types.t_FieldReal;
DiscreteField := Types.DiscreteField;
NumericField := Types.NumericField;
GenField := LT_Types.GenField;
trainDat := CovTypeDS.trainRecs;
testDat := CovTypeDS.testRecs;
ctRec := CovTypeDS.covTypeRec;
nominalFields := CovTypeDS.nominalCols;
numCols := CovTypeDS.numCols;

// Test conversions between nodes and model:
// Call GetNodes, convert to a model, convert the model back to nodes and compare nodes
// Use CovType database as a convenient way to generate the initial forest nodes.


ML_Core.ToField(trainDat, trainNF);
ML_Core.ToField(testDat, testNF);
// Take out the first field from training set (Elevation) to use as the target value.  Re-number the other fields
// to fill the gap
X0 := PROJECT(trainNF(number != 1), TRANSFORM(GenField,
        SELF.isOrdinal := TRUE, SELF.number := LEFT.number -1, SELF := LEFT));
Y0 := PROJECT(trainNF(number = 1), TRANSFORM(GenField,
        SELF.isOrdinal := TRUE, SELF.number := 1, SELF := LEFT));
// Truncate samples since we are only training the model in order to test conversions
X1 := X0(id <= 500);
Y1 := Y0(id <= 500);
// Now create multiple work items
X := NORMALIZE(X1, numWorkItems, TRANSFORM(GenField, SELF.wi := COUNTER, SELF := LEFT));
Y := NORMALIZE(Y1, numWorkItems, TRANSFORM(GenField, SELF.wi := COUNTER, SELF := LEFT));
OUTPUT(X1, NAMED('X1'));
//OUTPUT(X, NAMED('X'));
F := int.RF_Regression(X1, Y1, numTrees:=numTrees, featuresPerNode:=numFeatures, maxDepth:=maxDepth);
nodes1 := SORT(DISTRIBUTE(F.GetNodes, HASH32(wi, treeId)), wi, treeId, level, nodeId, LOCAL);
mod := F.Nodes2Model(nodes1);
nodes2 := SORT(DISTRIBUTE(F.Model2Nodes(mod), HASH32(wi, treeId)), wi, treeId, level, nodeId, LOCAL);
cmp := JOIN(nodes1, nodes2, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND LEFT.level = RIGHT.level AND
                     LEFT.nodeId = RIGHT.nodeId,
              TRANSFORM({nodes1, UNSIGNED err}, SELF.err := IF(LEFT.number != RIGHT.number OR
                                                          LEFT.value != RIGHT.value OR
                                                          LEFT.isLeft != RIGHT.isLeft OR
                                                          LEFT.parentId != RIGHT.parentId OR
                                                          LEFT.isOrdinal != RIGHT.isOrdinal OR
                                                          LEFT.support != RIGHT.support OR
                                                          LEFT.depend != RIGHT.depend,
                                                          1, 0),
                                                 SELF := LEFT), FULL OUTER, LOCAL);
cmp2 := JOIN(nodes1, nodes2, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND LEFT.level = RIGHT.level AND
                     LEFT.nodeId = RIGHT.nodeId,
              TRANSFORM({nodes1, UNSIGNED err}, SELF.err := IF(LEFT.number != RIGHT.number OR
                                                          LEFT.value != RIGHT.value OR
                                                          LEFT.isLeft != RIGHT.isLeft OR
                                                          LEFT.parentId != RIGHT.parentId OR
                                                          LEFT.isOrdinal != RIGHT.isOrdinal OR
                                                          LEFT.support != RIGHT.support OR
                                                          LEFT.depend != RIGHT.depend,
                                                          1, 0),
                                                 SELF := RIGHT), FULL OUTER, LOCAL);
OUTPUT(SORT(cmp(err>0), wi, treeId, level, nodeId, LOCAL), {err, wi, treeId, level, nodeId, parentId, isLeft, number, value, depend, support, isOrdinal, id}, NAMED('Compare1'));
OUTPUT(SORT(cmp2(err>0), wi, treeId, level, nodeId, LOCAL), {err, wi, treeId, level, nodeId, parentId, isLeft, number, value, depend, support, isOrdinal, id}, NAMED('Compare2'));
OUTPUT(SORT(nodes1, wi, treeId, level, nodeId, LOCAL), {wi, treeId, level, nodeId, parentId, isLeft, number, value, depend, support, isOrdinal, id}, NAMED('InitialNodes'));
OUTPUT(SORT(nodes2, wi, treeId, level, nodeId, LOCAL), {wi, treeId, level, nodeId, parentId, isLeft, number, value, depend, support, isOrdinal, id}, NAMED('FinalNodes'));

nodes1cnt := COUNT(nodes1);
nodes2cnt := COUNT(nodes2);

errCnt := SUM(cmp, err);
zerCnt := COUNT(nodes2(wi=0));
summary := DATASET([{nodes1cnt, nodes2cnt, errCnt, zerCnt}], {UNSIGNED nodes1cnt, UNSIGNED nodes2cnt, UNSIGNED errCnt, UNSIGNED zerCnt});

OUTPUT(summary, NAMED('Summary'));
