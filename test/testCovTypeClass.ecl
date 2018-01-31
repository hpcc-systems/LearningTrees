/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */
/**
  * Use the Cover Type database of Rocky Mountain Forest plots.
  * Perform a Random Forest classification to determine the primary Cover Type
  * (i.e. tree species) for each plot of land.
  * Do not be confused by the fact that we are using Random Forests to predict
  * tree species in an actual forest :)
  * @see test/datasets/CovTypeDS.ecl
  */
IMPORT $.datasets.CovTypeDS;
IMPORT $.^ AS LT;
IMPORT LT.LT_Types;
IMPORT ML_Core;
IMPORT ML_Core.Types;

numTrees := 20;
maxDepth := 255;
numFeatures := 0; // Zero is automatic choice
balanceClasses := FALSE;
nonSequentialIds := TRUE; // True to renumber ids, numbers and work-items to test
                            // support for non-sequentiality
numWIs := 2;     // The number of independent work-items to create
maxRecs := 5000; // Note that this has to be less than or equal to the number of records
                    // in CovTypeDS (currently 5000)
t_Discrete := Types.t_Discrete;
t_FieldReal := Types.t_FieldReal;
DiscreteField := Types.DiscreteField;
NumericField := Types.NumericField;
trainDat := CovTypeDS.trainRecs;
testDat := CovTypeDS.testRecs;
ctRec := CovTypeDS.covTypeRec;
nominalFields := CovTypeDS.nominalCols;
numCols := CovTypeDS.numCols;

ML_Core.ToField(trainDat, trainNF);
ML_Core.ToField(testDat, testNF);
X0 := PROJECT(trainNF(number != 52 AND id <= maxRecs), TRANSFORM(NumericField,
        SELF.number := IF(nonSequentialIds, 5*LEFT.number, LEFT.number),
        SELF.id := IF(nonSequentialIds, 5*LEFT.id, LEFT.id),
        SELF := LEFT));
Y0 := PROJECT(trainNF(number = 52 AND id <= maxRecs), TRANSFORM(DiscreteField,
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

card0 := SORT(X, number, value);
card1 := TABLE(card0, {number, value, valCnt := COUNT(GROUP)}, number, value);
card2 := TABLE(card1, {number, featureVals := COUNT(GROUP)}, number);
card := TABLE(card2, {cardinality := SUM(GROUP, featureVals)}, ALL);
OUTPUT(card, NAMED('X_Cardinality'));

F := LT.ClassificationForest(numTrees, numFeatures, maxDepth, nominalFields, balanceClasses);

mod := F.GetModel(X, Y);
OUTPUT(mod, NAMED('model'));
nodes := SORT(F.Model2Nodes(mod), wi, treeId, level, nodeId);
OUTPUT(CHOOSEN(nodes, 500), {wi, treeId, level, nodeId, parentId, isLeft, number, value, depend, support, ir}, ALL, NAMED('TreeNodes'));
modStats := F.GetModelStats(mod);
OUTPUT(modStats, NAMED('ModelStatistics'));
classWeights := F.Model2ClassWeights(mod);
OUTPUT(classWeights, NAMED('ClassWeights'));

Y_S := SORT(Y, value);
classCounts0 := TABLE(Y, {wi, class := value, cnt := COUNT(GROUP)}, wi, value);
classCounts := TABLE(classCounts0, {wi, classes := COUNT(GROUP)}, wi);

Xtest0 :=  PROJECT(testNF(number != 52), TRANSFORM(NumericField,
        SELF.number := IF(nonSequentialIds, 5*LEFT.number, LEFT.number),
        SELF.id := IF(nonSequentialIds, 5*LEFT.id, LEFT.id),
        SELF := LEFT));
Ycmp0 := PROJECT(testNF(number = 52), TRANSFORM(DiscreteField,
        SELF.id := IF(nonSequentialIds, 5*LEFT.id, LEFT.id),
        SELF := LEFT));
// Generate multiple work items
Xtest := NORMALIZE(Xtest0, numWIs, TRANSFORM(RECORDOF(LEFT),
          SELF.wi := IF(nonSequentialIds, 5*COUNTER, COUNTER),
          SELF := LEFT));
Ycmp := NORMALIZE(Ycmp0, numWIs, TRANSFORM(RECORDOF(LEFT),
          SELF.wi := IF(nonSequentialIds, 5*COUNTER, COUNTER),
          SELF.number := 1;
          SELF := LEFT));

classProbs := F.GetClassProbs(mod, Xtest);
OUTPUT(classProbs, NAMED('ClassProbabilities'));
// OUTPUT(COUNT(classProbs), NAMED('CP_Size'));
Yhat := F.Classify(mod, Xtest);

cmp := JOIN(Yhat, Ycmp, LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id, TRANSFORM({DiscreteField, t_Discrete cmpValue, UNSIGNED errors},
                  SELF.cmpValue := RIGHT.value, SELF.errors := IF(LEFT.value != RIGHT.value, 1, 0), SELF := LEFT));
OUTPUT(cmp, NAMED('Details'));

accuracy := F.Accuracy(mod, Ycmp, Xtest);
OUTPUT(accuracy, NAMED('Accuracy'));

accuracyByClass := F.AccuracyByClass(mod, Ycmp, Xtest);
OUTPUT(accuracyByClass, NAMED('AccuracyByClass'));

confusion := F.ConfusionMatrix(mod, Ycmp, Xtest);
OUTPUT(confusion, NAMED('ConfusionMatrix'));

fi := F.FeatureImportance(mod);
OUTPUT(fi, NAMED('FeatureImportance'));
