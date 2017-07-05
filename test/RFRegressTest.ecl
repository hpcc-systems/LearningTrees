IMPORT $.^ AS LT;
IMPORT LT.LT_Types AS Types;
IMPORT ML_Core;
IMPORT ML_Core.Types AS CTypes;

NumericField := CTypes.NumericField;
wiCount := 1;
numTrainingRecs := 20000;
numTestRecs := 10000;
numTrees := 50;
numVarsPerTree := 3;
maxTreeDepth := 1000;

// Return TRUE with probability p
prob(REAL p) := FUNCTION
  rnd := RANDOM() % 1000000 + 1;
  isTrue := IF(rnd / 1000000 <= p, TRUE, FALSE);
  RETURN isTrue;
END;

// Test Function:
// Y := IF X1 < 0:  X2 + X3
//      ELSE: X3 - X2
//


dsRec := {UNSIGNED id, REAL X1, REAL X2, REAL X3, REAL Y};
dummy := DATASET([{0, 0, 0, 0, 0}], dsRec);

dsRec make_data(dsRec d, UNSIGNED c) := TRANSFORM
  SELF.id := c;
  // Pick random X1:  -100 < X1 < 100
  r1 := __COMMON__(RANDOM());
  r2 := __COMMON__(RANDOM());
  r3 := __COMMON__(RANDOM());
  SELF.X1 := ROUND(r1%1000000 / 10000 * 2 - 100);
  // Pick random X2 and X3
  SELF.X2 := ROUND(r2%1000000 / 10000 * 2 - 100);
  BOOLEAN x2B := SELF.X2=1;
  SELF.X3 := ROUND(r3%1000000 / 10000 * 2 -100);
  SELF.Y := IF(SELF.X1 <= 0, SELF.X2 + SELF.X3, SELF.X3 - SELF.X2);
END;
ds := NORMALIZE(dummy, numTrainingRecs, make_data(LEFT, COUNTER));
OUTPUT(ds, NAMED('TrainingData'));

X1 := PROJECT(ds, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 1,
                            SELF.value := LEFT.X1));
X2 := PROJECT(ds, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 2,
                            SELF.value := LEFT.X2));
X3 := PROJECT(ds, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 3,
                            SELF.value := LEFT.X3));
Y := PROJECT(ds, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 1,
                            SELF.value := LEFT.Y));

X := X1 + X2 + X3;

// Expand to number of work items
Xe := NORMALIZE(X, wiCount, TRANSFORM(NumericField, SELF.wi := COUNTER, SELF := LEFT));
Ye := NORMALIZE(Y, wiCount, TRANSFORM(NumericField, SELF.wi := COUNTER, SELF := LEFT));
// Repeat the data multiple times for training
OUTPUT(Ye, NAMED('Y_train'));

F := LT.RegressionForest(numTrees, numVarsPerTree, maxTreeDepth);

//mod := F.GetModel(Xe, Ye);
mod := F.GetModel(X, Y);

OUTPUT(mod, NAMED('Model'));
modStats := F.GetModelStats(mod);
OUTPUT(modStats, NAMED('ModelStatistics'));
dsTest := DISTRIBUTE(SORT(NORMALIZE(dummy, numTestRecs, make_data(LEFT, COUNTER)), id, LOCAL), id);
X1t := PROJECT(dsTest, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 1,
                            SELF.value := LEFT.X1));
X1t0 := PROJECT(dsTest, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 1,
                            SELF.value := LEFT.X1));
X2t := PROJECT(dsTest, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 2,
                            SELF.value := LEFT.X2));
X3t := PROJECT(dsTest, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 3,
                            SELF.value := LEFT.X3));
Xt := X1t + X2t + X3t;
Ycmp := PROJECT(dsTest, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 1,
                            SELF.value := LEFT.Y));
Yhat0 := F.Predict(Xt, mod);
Yhat := DISTRIBUTE(SORT(PROJECT(Yhat0, TRANSFORM(NumericField, SELF := LEFT)), id, LOCAL),  id);
OUTPUT(Yhat, NAMED('rawPredict'));

dseRec := RECORD(dsRec)
  REAL Yhat;
  REAL err2;
  STRING4 Status;
  UNSIGNED wi;
END;

dseRec dseFromXY(dsRec l, NumericField r) := TRANSFORM
  SELF.wi := r.wi;
  SELF.Yhat := r.value;
  SELF.err2 := POWER(l.y - SELF.Yhat, 2);
  SELF.Status := IF(l.y = SELF.Yhat, '', 'FAIL');
  SELF := l;
END;
dsCmp := SORT(JOIN(dsTest, Yhat, LEFT.id = RIGHT.id, dseFromXY(LEFT, RIGHT), LOCAL), id);

OUTPUT(dsCmp, NAMED('Details'));

MSE := TABLE(dsCmp, {wi, mse := AVE(GROUP, err2), rmse := POWER(AVE(GROUP, err2), .5)}, wi);

OUTPUT(MSE, NAMED('MeanSquareError'));