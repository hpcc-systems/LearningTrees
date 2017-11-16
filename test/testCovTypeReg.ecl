/**
  * Use the Cover Type database of Rocky Mountain Forest plots.
  * Perform a regression to predict the elevation given the other features.
  * Do not be confused by the fact that we are using Random Forests to analyze
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
// Take out the first field from training set (Elevation) to use as the target value.  Re-number the other fields
// to fill the gap
X := PROJECT(trainNF(number != 1), TRANSFORM(NumericField,
        SELF.number := LEFT.number -1, SELF := LEFT));
Y := PROJECT(trainNF(number = 1), TRANSFORM(NumericField,
        SELF.number := 1, SELF := LEFT));
card0 := SORT(X, number, value);
card1 := TABLE(card0, {number, value, valCnt := COUNT(GROUP)}, number, value);
card2 := TABLE(card1, {number, featureVals := COUNT(GROUP)}, number);
card := TABLE(card2, {cardinality := SUM(GROUP, featureVals)}, ALL);
OUTPUT(card, NAMED('X_Cardinality'));
F := LT.RegressionForest(numTrees:=numTrees, featuresPerNode:=numFeatures, maxDepth:=maxDepth);
mod := F.GetModel(X, Y, nominalFields);
OUTPUT(Y, NAMED('Ytrain'));
Y_S := SORT(Y, value);
classCounts0 := TABLE(Y, {wi, class := value, cnt := COUNT(GROUP)}, wi, value);
classCounts := TABLE(classCounts0, {wi, classes := COUNT(GROUP)}, wi);

OUTPUT(mod, NAMED('Model'));
modStats := F.GetModelStats(mod);
OUTPUT(modStats, NAMED('ModelStatistics'));
Xtest := PROJECT(testNF(number != 1), TRANSFORM(GenField, SELF.isOrdinal := IF(LEFT.number IN nominalFields, FALSE, TRUE),
                    SELF.number := LEFT.number - 1, SELF := LEFT));
Ycmp := PROJECT(testNF(number = 1), TRANSFORM(GenField, SELF.isOrdinal := IF(LEFT.number IN nominalFields, FALSE, TRUE), SELF.number := 1, SELF := LEFT));

Yhat := F.Predict(Xtest, mod);

cmp := JOIN(Yhat, Ycmp, LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id, TRANSFORM({UNSIGNED wi, UNSIGNED id, REAL y, REAL yhat, REAL err, REAL err2},
                  SELF.y := RIGHT.value, SELF.yhat := LEFT.value, SELF.err2 := POWER(LEFT.value - RIGHT.value, 2),
                  SELF.err := ABS(LEFT.value - RIGHT.value), SELF := LEFT));

OUTPUT(cmp, NAMED('Details'));

Yvar := VARIANCE(Ycmp, value);
rsq := F.Rsquared(Xtest, Ycmp, mod);
MSE := TABLE(cmp, {wi, mse := AVE(GROUP, err2), rmse := SQRT(AVE(GROUP, err2)), stdevY := SQRT(VARIANCE(GROUP, y))}, wi);
ErrStats := JOIN(MSE, rsq, LEFT.wi = RIGHT.wi, TRANSFORM({mse, REAL R2}, SELF.R2 := RIGHT.R2, SELF := LEFT));

OUTPUT(ErrStats, NAMED('ErrorStats'));
