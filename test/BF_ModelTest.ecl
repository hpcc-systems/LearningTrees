/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */
/**
  * Test Gradient Boosted Tree Regression.
  * Use the Cover Type database of Rocky Mountain Forest plots.
  * Perform a regression to predict the elevation given the other features.
  * Do not be confused by the fact that we are using Random Forests to analyze
  * tree species in an actual forest :)
  * @see test/datasets/CovTypeDS.ecl
  */
IMPORT $.datasets.CovTypeDsSmall as CovTypeDS;
IMPORT $.^ AS LT;
IMPORT LT.LT_Types;
IMPORT ML_Core;
IMPORT ML_Core.Types;
#OPTION('outputLimit', 100);
maxLevels := 255;
forestSize := 10;  // Zero indicates auto choice
// 5, 7, 12, 20
maxTreeDepth := 255;
//earlyStopThreshold := 0.0;
earlyStopThreshold := 0.0001;
// .1, .25, .5, .75, 1
learningRate := 1;
numFeatures := 0; // Zero is automatic choice
nonSequentialIds := FALSE; // True to renumber ids, numbers and work-items to test
                            // support for non-sequentiality
numWIs := 1;    // The number of independent work-items to create
maxRecs := 500; // Note that this has to be less than or equal to the number of records
                // in CovTypeDS (currently 5000)

t_Discrete := Types.t_Discrete;
t_FieldReal := Types.t_FieldReal;
DiscreteField := Types.DiscreteField;
NumericField := Types.NumericField;
GenField := LT_Types.GenField;
BfTreeNodeDat := LT_Types.BfTreeNodeDat;
trainDat := CovTypeDS.trainRecs;
testDat := CovTypeDS.testRecs;
ctRec := CovTypeDS.covTypeRec;
nominalFields := CovTypeDS.nominalCols;
numCols := CovTypeDS.numCols;

ML_Core.ToField(trainDat, trainNF);
ML_Core.ToField(testDat, testNF);
// Take out the first field from training set (Elevation) to use as the target value.  Re-number the other fields
// to fill the gap
X0 := PROJECT(trainNF(number != 1 AND id <= maxRecs), TRANSFORM(GenField,
        SELF.isOrdinal := FALSE,
        SELF.number := IF(nonSequentialIds, (5*LEFT.number -1), LEFT.number -1),
        SELF.id := IF(nonSequentialIds, 5*LEFT.id, LEFT.id),
        SELF := LEFT));
Y0 := PROJECT(trainNF(number = 1 AND id <= maxRecs), TRANSFORM(GenField,
        SELF.isOrdinal := FALSE,
        SELF.number := 1,
        SELF.id := IF(nonSequentialIds, 5*LEFT.id, LEFT.id),
        SELF := LEFT));
// Generate multiple work items
X := NORMALIZE(X0, numWIs, TRANSFORM(RECORDOF(LEFT),
          SELF.wi := IF(nonSequentialIds, 5*COUNTER, COUNTER),
          SELF := LEFT));
Y := NORMALIZE(Y0, numWIs, TRANSFORM(RECORDOF(LEFT),
          SELF.wi := IF(nonSequentialIds, 5*COUNTER, COUNTER),
          SELF := LEFT));

IMPORT Python;
SET OF UNSIGNED incrementSet(SET OF UNSIGNED s, INTEGER increment) := EMBED(Python)
  outSet = []
  for i in range(len(s)):
    outSet.append(s[i] + increment)
  return outSet
ENDEMBED;
// Fixup IDs of nominal fields to match
//nomFields := incrementSet(nominalFields, -1);
nomFields := [10,51];  // Temporary no python
card0 := SORT(X, number, value);
card1 := TABLE(card0, {number, value, valCnt := COUNT(GROUP)}, number, value);
card2 := TABLE(card1, {number, featureVals := COUNT(GROUP)}, number);
card := TABLE(card2, {cardinality := SUM(GROUP, featureVals)}, ALL);

X_nom := PROJECT(X, TRANSFORM(RECORDOF(LEFT),
                      SELF.isOrdinal := IF(LEFT.number in nomFields, FALSE, TRUE),
                      SELF := LEFT), LOCAL);
F := LT.internal.BF_Regression(X_nom, Y, maxLevels:=maxLevels, forestSize:=forestSize,
                                maxTreeDepth:=maxTreeDepth,
                                earlyStopThreshold := earlyStopThreshold,
                                learningRate := learningRate);
nodes0 := F.GetNodes : PERSIST('ROGER::Temp::Nodes', SINGLE, REFRESH(TRUE));
OUTPUT(nodes0, NAMED('Nodes0'));

//nodes := DATASET('ROGER::Temp::Nodes', BfTreeNodeDat, THOR)(bfLevel < 3 AND TreeId < 3 AND level < 4);
nodes := DATASET('ROGER::Temp::Nodes', BfTreeNodeDat, THOR);

nodes1 := SORT(DISTRIBUTE(nodes, HASH32(wi, treeId)), wi, bfLevel, treeId, level, nodeId, LOCAL);
mod := F.Nodes2Model(nodes1);
OUTPUT(SORT(mod[..3000], wi, indexes), ALL, NAMED('Model'));

nodes2 := SORT(DISTRIBUTE(F.Model2Nodes(mod), HASH32(wi, treeId)), wi, bfLevel, treeId, level, nodeId, LOCAL);
cmp := JOIN(nodes1, nodes2, LEFT.wi = RIGHT.wi AND LEFT.bfLevel = RIGHT.bfLevel AND
                     LEFT.treeId = RIGHT.treeId AND LEFT.level = RIGHT.level AND
                     LEFT.nodeId = RIGHT.nodeId,
              TRANSFORM({nodes1, UNSIGNED err}, SELF.err := IF(LEFT.number != RIGHT.number OR
                                                          LEFT.bfLevel != RIGHT.bfLevel OR
                                                          LEFT.value != RIGHT.value OR
                                                          LEFT.isLeft != RIGHT.isLeft OR
                                                          LEFT.parentId != RIGHT.parentId OR
                                                          LEFT.isOrdinal != RIGHT.isOrdinal OR
                                                          LEFT.support != RIGHT.support OR
                                                          LEFT.depend != RIGHT.depend,
                                                          1, 0),
                                                 SELF := LEFT), FULL OUTER, LOCAL);
cmp2 := JOIN(nodes1, nodes2, LEFT.wi = RIGHT.wi AND LEFT.bfLevel = RIGHT.bfLevel AND
                     LEFT.treeId = RIGHT.treeId AND LEFT.level = RIGHT.level AND
                     LEFT.nodeId = RIGHT.nodeId,
              TRANSFORM({nodes1, UNSIGNED err}, SELF.err := IF(LEFT.number != RIGHT.number OR
                                                          LEFT.bfLevel != RIGHT.bfLevel OR
                                                          LEFT.value != RIGHT.value OR
                                                          LEFT.isLeft != RIGHT.isLeft OR
                                                          LEFT.parentId != RIGHT.parentId OR
                                                          LEFT.isOrdinal != RIGHT.isOrdinal OR
                                                          LEFT.support != RIGHT.support OR
                                                          LEFT.depend != RIGHT.depend,
                                                          1, 0),
                                                 SELF := RIGHT), FULL OUTER, LOCAL);
OUTPUT(SORT(cmp(err>0), wi, bfLevel, treeId, level, nodeId, LOCAL), {err, wi, bfLevel, treeId, level, nodeId, parentId, isLeft, number, value, depend, support, isOrdinal, id}, NAMED('Compare1'));
OUTPUT(SORT(cmp2(err>0), wi, bfLevel, treeId, level, nodeId, LOCAL), {err, wi, bfLevel, treeId, level, nodeId, parentId, isLeft, number, value, depend, support, isOrdinal, id}, NAMED('Compare2'));
OUTPUT(SORT(nodes1, wi, bfLevel, treeId, level, nodeId, LOCAL), {wi, bfLevel, treeId, level, nodeId, parentId, isLeft, number, value, depend, support, isOrdinal, id}, ALL, NAMED('InitialNodes'));
OUTPUT(SORT(nodes2, wi, bfLevel, treeId, level, nodeId, LOCAL), {wi, bfLevel, treeId, level, nodeId, parentId, isLeft, number, value, depend, support, isOrdinal, id}, ALL, NAMED('FinalNodes'));

nodes1cnt := COUNT(nodes1);
nodes2cnt := COUNT(nodes2);

modCnt := COUNT(mod);
OUTPUT(modCnt, NAMED('ModelRecs'));
errCnt := SUM(cmp, err);
zerCnt := COUNT(nodes2(wi=0));
summary := DATASET([{nodes1cnt, nodes2cnt, errCnt, zerCnt}], {UNSIGNED nodes1cnt, UNSIGNED nodes2cnt, UNSIGNED errCnt, UNSIGNED zerCnt});

OUTPUT(summary, NAMED('Summary'));

OUTPUT(nodes(treeId=1), ALL, NAMED('NodesTree1'));
