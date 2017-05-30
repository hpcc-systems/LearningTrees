IMPORT $.^ AS LT;
IMPORT ML_Core.Types AS CTypes;
IMPORT LT.LT_Types AS Types;

NumericField := CTypes.NumericField;
DiscreteField := CTypes.DiscreteField;
GenField := Types.GenField;
errorProb := 0;
wiCount := 1;
numTrainingRecs := 1000;
numTestRecs := 100;
numTrees := 1;
numVarsPerTree := 3;

// Return TRUE with probability p
prob(REAL p) := FUNCTION
  rnd := RANDOM() % 1000000 + 1;
  isTrue := IF(rnd / 1000000 <= p, TRUE, FALSE);
  RETURN isTrue;
END;

// Test Pattern -- Ordinal variable X1 determines OP(X2, X3) => Y
//                 OP (OR, AND, XOR, NOR), is determined as follows:
//                 X1 < -50 => OR(X2, X3); -50 <= X1 < 0 => AND(X2, X3);
//                 0 < X1 <= 50 => XOR(X2, X3); X1 >= 50 => NOR(X2, X3);

dsRec := {UNSIGNED id, REAL X1, UNSIGNED X2, UNSIGNED X3, UNSIGNED Y};
dsRec0 := {UNSIGNED id, UNSIGNED X1, UNSIGNED X2, UNSIGNED X3, UNSIGNED Y};
dummy := DATASET([{0, 0, 0, 0, 0}], dsRec);
dsRec make_data0(dsRec d, UNSIGNED c) := TRANSFORM
  SELF.id := c;
  // Pick random X1:  -100 < X1 < 100
  r1 := __COMMON__(RANDOM());
  r2 := __COMMON__(RANDOM());
  r3 := __COMMON__(RANDOM());
  SELF.X1 := r1%4;
  // Pick random X2 and X3: Choose val between 0 and 1 and round to 0 or 1.
  SELF.X2 := ROUND(r2%1000000 / 1000000);
  BOOLEAN x2B := SELF.X2=1;
  SELF.X3 := ROUND(r3%1000000 / 1000000);
  BOOLEAN x3B := SELF.X3=1;
  BOOLEAN y := MAP(SELF.X1 = 0 => x2B OR x3B, // OR
                         SELF.X1 = 1 => x2B AND x3B, // AND
                         SELF.X1 = 2 => (x2B OR x3B) AND (NOT (x2B AND x3B)), // XOR
                         (NOT (x2B OR x3B)));  // NOR
  SELF.Y := IF(y, 1, 0);
END;
dsRec make_data(dsRec d, UNSIGNED c) := TRANSFORM
  SELF.id := c;
  // Pick random X1:  -100 < X1 < 100
  r1 := __COMMON__(RANDOM());
  r2 := __COMMON__(RANDOM());
  r3 := __COMMON__(RANDOM());
  SELF.X1 := ROUND(r1%1000000 / 10000 * 2 - 100);
  // Pick random X2 and X3: Choose val between 0 and 1 and round to 0 or 1.
  SELF.X2 := ROUND(r2%1000000 / 1000000);
  BOOLEAN x2B := SELF.X2=1;
  SELF.X3 := ROUND(r3%1000000 / 1000000);
  BOOLEAN x3B := SELF.X3=1;
  BOOLEAN y := MAP(SELF.X1 < -50 => x2B OR x3B, // OR
                         SELF.X1 >= -50 AND SELF.X1 < 0 => x2B AND x3B, // AND
                         SELF.X1 >= 0 AND SELF.X1 < 50 => (x2B OR x3B) AND (NOT (x2B AND x3B)), // XOR
                         (NOT (x2B OR x3B)));  // NOR
  SELF.Y := IF(y, 1, 0);
END;
ds := NORMALIZE(dummy, numTrainingRecs, make_data(LEFT, COUNTER));
OUTPUT(ds, NAMED('TrainingData'));

X1 := PROJECT(ds, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 1,
                            SELF.value := LEFT.X1));
X2 := PROJECT(ds, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 2,
                            SELF.value := LEFT.X2));
X3 := PROJECT(ds, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 3,
                            SELF.value := LEFT.X3));
// Add noise to Y by randomly flipping the value according to PROBABILITY(errorProb).
Y := PROJECT(ds, TRANSFORM(DiscreteField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 1,
                            SELF.value := IF(prob(errorProb), (LEFT.Y + 1)%2, LEFT.Y)));
nominals := [];
X := X1 + X2 + X3;

// Expand to number of work items
Xe := NORMALIZE(X, wiCount, TRANSFORM(NumericField, SELF.wi := COUNTER, SELF := LEFT));
Ye := NORMALIZE(Y, wiCount, TRANSFORM(DiscreteField, SELF.wi := COUNTER, SELF := LEFT));

OUTPUT(Ye, NAMED('Y_train'));

F := LT.ClassificationForest(numTrees, numVarsPerTree);

//mod := F.GetModel(Xe, Ye, nominals);

// With this line it runs fine.
mod := F.GetModel(X, Y, nominals);

OUTPUT(mod, NAMED('Model'));
modStats := F.GetModelStats(mod);
OUTPUT(modStats, NAMED('ModelStats'));
tn0 := F.Model2Nodes(mod);
tn := SORT(tn0, wi, treeId, level, nodeid, id, number, depend);
OUTPUT(tn, {wi, level, treeId, nodeId, parentId, isLeft, id, number, value, depend, support, isOrdinal},NAMED('Tree'));

dsTest := DISTRIBUTE(SORT(NORMALIZE(dummy, numTestRecs, make_data(LEFT, COUNTER)), id, LOCAL), id);
X1t := PROJECT(dsTest, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 1,
                            SELF.value := LEFT.X1));
X2t := PROJECT(dsTest, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 2,
                            SELF.value := LEFT.X2));
X3t := PROJECT(dsTest, TRANSFORM(NumericField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 3,
                            SELF.value := LEFT.X3));
Xt := X1t + X2t + X3t;
Ycmp := PROJECT(dsTest, TRANSFORM(DiscreteField, SELF.wi := 1, SELF.id := LEFT.id, SELF.number := 1,
                            SELF.value := LEFT.Y));
Yhat0 := F.Classify(Xt, mod);
Yhat := DISTRIBUTE(SORT(Yhat0, id, LOCAL),  id);
OUTPUT(Yhat, NAMED('rawPredict'));

dseRec := RECORD(dsRec)
  UNSIGNED Yhat;
  STRING4 Status;
END;

dsCmp := SORT(JOIN(dsTest, Yhat, LEFT.id = RIGHT.id, TRANSFORM(dseRec, SELF.Yhat := RIGHT.value,
                    SELF.Y := LEFT.Y, SELF.Status := IF(SELF.Y = SELF.Yhat, '', 'FAIL'), SELF := LEFT), LOCAL), id);

OUTPUT(dsCmp, NAMED('Details'));

summary := TABLE(dsCmp(Status = 'FAIL'), {UNSIGNED errors := COUNT(GROUP), REAL errorRate := COUNT(GROUP) / numTestRecs});

OUTPUT(summary, NAMED('Summary'));
