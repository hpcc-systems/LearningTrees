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

numTrees := 100;
maxDepth := 255;
numFeatures := 7;
balanceClasses := FALSE;

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

ML_Core.ToField(trainDat, trainNF);
ML_Core.ToField(testDat, testNF);
X := PROJECT(trainNF(number != 52), TRANSFORM(LEFT));
Y := PROJECT(trainNF(number = 52), TRANSFORM(DiscreteField, SELF.number := 1, SELF := LEFT));
card0 := SORT(X, number, value);
card1 := TABLE(card0, {number, value, valCnt := COUNT(GROUP)}, number, value);
card2 := TABLE(card1, {number, featureVals := COUNT(GROUP)}, number);
card := TABLE(card2, {cardinality := SUM(GROUP, featureVals)}, ALL);
OUTPUT(card, NAMED('X_Cardinality'));

F := LT.ClassificationForest(numTrees, numFeatures, maxDepth);
mod := F.GetModel(X, Y, nominalFields);
OUTPUT(mod, NAMED('model'));
nodes := F.Model2Nodes(mod);
OUTPUT(nodes, {wi, treeId, level, nodeId, parentId, isLeft, number, value, depend, support, id}, NAMED('TreeNodes'));
modStats := F.GetModelStats(mod);
OUTPUT(modStats, NAMED('ModelStatistics'));
classWeights := F.Model2ClassWeights(mod);
OUTPUT(classWeights, NAMED('ClassWeights'));

Y_S := SORT(Y, value);
classCounts0 := TABLE(Y, {wi, class := value, cnt := COUNT(GROUP)}, wi, value);
classCounts := TABLE(classCounts0, {wi, classes := COUNT(GROUP)}, wi);

Xtest := testNF(number != 52);
Ycmp := PROJECT(testNF(number = 52), DiscreteField);
classProbs := F.GetClassProbs(Xtest, mod, balanceClasses);
OUTPUT(classProbs, NAMED('ClassProbabilities'));
// OUTPUT(COUNT(classProbs), NAMED('CP_Size'));
Yhat := F.Classify(Xtest, mod, balanceClasses);

cmp := JOIN(Yhat, Ycmp, LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id, TRANSFORM({DiscreteField, t_Discrete cmpValue, UNSIGNED errors},
                  SELF.cmpValue := RIGHT.value, SELF.errors := IF(LEFT.value != RIGHT.value, 1, 0), SELF := LEFT));

OUTPUT(cmp, NAMED('Details'));

errStats := F.GetErrorStats(Xtest, Ycmp, mod, balanceClasses);
OUTPUT(errStats, NAMED('Accuracy'));

confusion := F.ConfusionMatrix(Xtest, Ycmp, mod, balanceClasses);
OUTPUT(confusion, NAMED('ConfusionMatrix'));
