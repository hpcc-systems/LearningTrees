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
IMPORT $.datasets.CovTypeDsLarge as CovTypeDS;
IMPORT $.^ AS LT;
IMPORT LT.LT_Types;
IMPORT ML_Core;
IMPORT ML_Core.Types;
#OPTION('outputLimit', 100);
maxLevels := 255;
forestSize := 0;  // Zero indicates auto choice
// 5, 7, 12, 20
maxTreeDepth := 255;
//earlyStopThreshold := 0.0;
earlyStopThreshold := 0.0001;
// .1, .25, .5, .75, 1
learningRate := 1;
numFeatures := 0; // Zero is automatic choice
nonSequentialIds := FALSE; // True to renumber ids, numbers and work-items to test
                            // support for non-sequentiality
numWIs := 2;    // The number of independent work-items to create
maxRecs := 5000; // Note that this has to be less than or equal to the number of records
                // in CovTypeDS (currently 5000)
maxTestRecs := 5000;
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
Layout_Model2 := Types.Layout_Model2;

ML_Core.ToField(trainDat, trainNF);
ML_Core.ToField(testDat, testNF);
// Take out the first field from training set (Elevation) to use as the target value.  Re-number the other fields
// to fill the gap
X0 := PROJECT(trainNF(number != 1 AND id <= maxRecs), TRANSFORM(NumericField,
        SELF.number := IF(nonSequentialIds, (5*LEFT.number -1), LEFT.number -1),
        SELF.id := IF(nonSequentialIds, 5*LEFT.id, LEFT.id),
        SELF := LEFT));
Y0 := PROJECT(trainNF(number = 1 AND id <= maxRecs), TRANSFORM(NumericField,
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
OUTPUT(X, NAMED('Xtrain'));
OUTPUT(Y, NAMED('Ytrain'));

F := LT.BoostedRegForest(maxLevels:=maxLevels,
                                forestSize:=forestSize,
                                maxTreeDepth:=maxTreeDepth,
                                earlyStopThreshold := earlyStopThreshold,
                                learningRate := learningRate,
                                nominalFields := nomFields);
//mod := F.GetModel(X, Y) : PERSIST('ROGER::Temp::Model', SINGLE, REFRESH(TRUE));
mod := F.GetModel(X, Y);

//mod2 := DATASET('ROGER::Temp::Model', Layout_Model2, THOR);
mod2 := mod;

OUTPUT(mod, NAMED('Model'));
nodes := SORT(F.Model2Nodes(mod2), wi, treeId, level, nodeId);
OUTPUT(nodes, {wi, treeId, level, nodeId, parentId, isLeft, number, value, depend, support, ir}, NAMED('TreeNodes'));
modStats := F.GetModelStats(mod2);
OUTPUT(modStats, NAMED('ModelStatistics'));
maxTestId := MIN(testNF, id) + maxTestRecs;
testNF2 := testNF(id < maxTestId);
//testNF2 := trainNF(id < maxTestRecs); // Temp test with training data
Xtest0 := PROJECT(testNF2(number != 1), TRANSFORM(NumericField,
                    SELF.number := IF(nonSequentialIds, (5*LEFT.number -1), LEFT.number -1),
                    SELF.id := IF(nonSequentialIds, 5*LEFT.id, LEFT.id),
                    SELF := LEFT));
Ycmp0 := PROJECT(testNF2(number = 1), TRANSFORM(NumericField,
                    SELF.number := 1,
                    SELF.id := IF(nonSequentialIds, 5*LEFT.id, LEFT.id),
                    SELF := LEFT));
// Generate multiple work items
Xtest := NORMALIZE(Xtest0, numWIs, TRANSFORM(NumericField,
          SELF.wi := IF(nonSequentialIds, 5*COUNTER, COUNTER),
          SELF := LEFT));
Ycmp := NORMALIZE(Ycmp0, numWIs, TRANSFORM(RECORDOF(LEFT),
          SELF.wi := IF(nonSequentialIds, 5*COUNTER, COUNTER),
          SELF := LEFT));


OUTPUT(learningRate, NAMED('learningRate'));
//Yhat := F.BfPredict(nodes2, Xtest): INDEPENDENT;
Yhat := F.Predict(mod2, Xtest);
OUTPUT(Yhat, ALL, NAMED('Pred'));
cmp := JOIN(Yhat, Ycmp, LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id, TRANSFORM({UNSIGNED wi, UNSIGNED id, REAL y, REAL yhat, REAL err, REAL err2},
                  SELF.y := RIGHT.value, SELF.yhat := LEFT.value, SELF.err2 := POWER(LEFT.value - RIGHT.value, 2),
                  SELF.err := ABS(LEFT.value - RIGHT.value), SELF := LEFT));

OUTPUT(cmp, NAMED('Details'));



accuracy :=F.Accuracy(mod2, Ycmp, Xtest);
OUTPUT(accuracy, NAMED('Accuracy'));

fi := F.FeatureImportance(mod2);

OUTPUT(fi, NAMED('FeatureImportance'));

//fi := F.FeatureImportance(mod);
//OUTPUT(fi, NAMED('FeatureImportance'));
