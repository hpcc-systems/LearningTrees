/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */

/**
  * Use the Cover Type database of Rocky Mountain Forest plots.
  * Compute the Decision Distance Matrix and the Uniqueness Factor
  *
  * Do not be confused by the fact that we are using Random Forests to analyze
  * tree species in an actual forest :)
  * @see test/datasets/CovTypeDS.ecl
  */
IMPORT $.datasets.CovTypeDS;
IMPORT $.^ AS LT;
IMPORT LT.LT_Types;
IMPORT ML_Core;
IMPORT ML_Core.Types;

// Can run as either Regression or Classification
useRegression := FALSE;
numTrees := 20;
maxDepth := 255;
numFeatures := 0;

t_Discrete := Types.t_Discrete;
t_FieldReal := Types.t_FieldReal;
DiscreteField := Types.DiscreteField;
NumericField := Types.NumericField;
trainDat := CovTypeDS.trainRecs;
testDat := CovTypeDS.testRecs;
ctRec := CovTypeDS.covTypeRec;
nominalFields := CovTypeDS.nominalCols;
numCols := CovTypeDS.numCols;

noiseLevel := 5;
noise(REAL level=noiseLevel) := FUNCTION
  RETURN ((RANDOM() / 4294967295 - .5) * 2) * noiselevel;
END;

ML_Core.ToField(trainDat, trainNF);
ML_Core.ToField(testDat, testNF);
// Take out the first field from training set (Elevation) to use as the target value.  Re-number the other fields
// to fill the gap
X1 := PROJECT(trainNF(number < 52), TRANSFORM(NumericField,
        SELF.number := LEFT.number, SELF := LEFT));
Y1 := PROJECT(trainNF(number = 52), TRANSFORM(DiscreteField,
        SELF.number := 1, SELF := LEFT));
X2 := PROJECT(trainNF(number != 1), TRANSFORM(NumericField,
        SELF.number := LEFT.number -1, SELF := LEFT));
Y2 := PROJECT(trainNF(number = 1), TRANSFORM(NumericField,
        SELF.number := 1, SELF := LEFT));

IMPORT Python;
SET OF UNSIGNED incrementSet(SET OF UNSIGNED s, INTEGER increment) := EMBED(Python)
  outSet = []
  for i in range(len(s)):
    outSet.append(s[i] + increment)
  return outSet
ENDEMBED;
// Fixup IDs of nominal fields to match
nomFields1 := nominalFields;
nomFields2 := incrementSet(nominalFields, -1);
nomFields := IF(useRegression, nomFields2, nomFields1);
F1 := LT.ClassificationForest(numTrees:=numTrees, featuresPerNode:=numFeatures, maxDepth:=maxDepth);
F2 := LT.RegressionForest(numTrees:=numTrees, featuresPerNode:=numFeatures, maxDepth:=maxDepth);

mod1 := F1.GetModel(X1, Y1, nomFields1);
mod2 := F2.GetModel(X2, Y2, nomFields2);

mod := IF(useRegression, mod2, mod1);

OUTPUT(mod, NAMED('Model'));
F3 := LT.LearningForest();
nodes := SORT(F3.Model2Nodes(mod), wi, treeId, level, nodeId);
OUTPUT(nodes, {wi, treeId, level, nodeId, parentId, isLeft, number, value, depend, support, ir}, NAMED('TreeNodes'));
modStats := F3.GetModelStats(mod);
OUTPUT(modStats, NAMED('ModelStatistics'));

Xtest1 := PROJECT(testNF(number < 52), TRANSFORM(NumericField,
                    SELF.number := LEFT.number, SELF := LEFT));
Xtest2 := PROJECT(testNF(number != 1), TRANSFORM(NumericField,
                    SELF.number := LEFT.number - 1, SELF := LEFT));
Xtest := IF(useRegression, Xtest2, Xtest1);

XtestR := Xtest(id <= 5025);

// Now create a similar point for each of the original test points.  We should see high similarity between
// points with ids 1000 apart. E.g., 5001 and 6001 should have high similarity.

XtestR2 := NORMALIZE(XtestR, 2, TRANSFORM(NumericField,
                                          SELF.id := IF(COUNTER = 1, LEFT.id, LEFT.id + 1000),
                                          SELF.value := IF(COUNTER = 1 OR LEFT.number IN nomFields OR LEFT.number > 10,
                                                              LEFT.value, LEFT.value + noise()),
                                          SELF := LEFT));
uniqueIds := DEDUP(XtestR, id);
minId := MIN(uniqueIds, id);
uidCount := COUNT(uniqueIds);
// Create a really unique datapoint by combining features from other datapoints.
NumericField makeUnique(NumericField protorec) := TRANSFORM
  UNSIGNED a := 1;
  valueRecId := (RANDOM() % uidCount) + 1;
  valueRec := XtestR2(id = valueRecId AND number = protorec.number)[1];
  SELF.value := IF(protorec.number IN nomFields OR protorec.number > 10, valueRec.value, valueRec.value + noise());
  SELF.id := 10000;
  SELF := protorec;
END;

// Start with the min-id as the prototype (really only used for 'number')
uniquePoint := PROJECT(XtestR2(id = minId), makeUnique(LEFT));
XtestR3 := XtestR2 + uniquePoint;

OUTPUT(XtestR3, NAMED('TestData'));

ddm := F3.DecisionDistanceMatrix(mod, XtestR3);

OUTPUT(SORT(ddm, id, number), NAMED('DecisionDistance'));

uf := F3.UniquenessFactor(mod, XtestR3);

ufS := SORT(uf, -value, id);
OUTPUT(ufS, NAMED('Uniqueness'));

withRank := PROJECT(ufS, TRANSFORM({ufS, UNSIGNED rank}, SELF.rank := COUNTER, SELF := LEFT));

unique := withRank(id = 10000)[1];

OUTPUT(unique.rank, NAMED('UniquePointRank'));

OUTPUT(unique.rank / COUNT(ufS), NAMED('UniquePointPercentile'));