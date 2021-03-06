IMPORT $.datasets.CovTypeDS;
IMPORT $.^ AS LT;
IMPORT LT.LT_Types;
IMPORT ML_Core;
IMPORT ML_Core.Types;

numTrees := 20;
maxDepth := 255;
numFeatures := 0; // Zero is automatic choice
numRecords := 2000; // Max is number of records in datasets/CovTypeDS.ecl (typically 5000)

t_Discrete := Types.t_Discrete;
t_FieldReal := Types.t_FieldReal;
DiscreteField := Types.DiscreteField;
NumericField := Types.NumericField;
trainDat := CovTypeDS.trainRecs;
testDat := CovTypeDS.testRecs;
ctRec := CovTypeDS.covTypeRec;
nominalFields := CovTypeDS.nominalCols;
numCols := CovTypeDS.numCols;
LUCI_Scorecard := LT_Types.LUCI_Scorecard;

ML_Core.ToField(trainDat, trainNF);
OUTPUT(trainNF_map, NAMED('DataMap'));
ML_Core.ToField(testDat, testNF);
// Take out the first field from training set (Elevation) to use as the target value.  Re-number the other fields
// to fill the gap
X := PROJECT(trainNF(number != 1 AND id < numRecords), TRANSFORM(NumericField,
        SELF.number := LEFT.number -1, SELF := LEFT));
Y := PROJECT(trainNF(number = 1 AND id < numRecords), TRANSFORM(NumericField,
        SELF.number := 1, SELF := LEFT));
IMPORT Python;
SET OF UNSIGNED incrementSet(SET OF UNSIGNED s, INTEGER increment) := EMBED(Python)
  outSet = []
  for i in range(len(s)):
    outSet.append(s[i] + increment)
  return outSet
ENDEMBED;
// Fixup IDs of nominal fields to match
nomFields := incrementSet(nominalFields, -1);
// Fixup IDs of the field map
fieldMap := PROJECT(trainNF_map, TRANSFORM({trainNF_map}, // Note the trainNF_map is generated by ToField macro
                                            SELF.assigned_name := (STRING)(((INTEGER) LEFT.assigned_name) - 1),
                                            SELF.orig_name := IF(LEFT.assigned_name = '1', SKIP, LEFT.orig_name),
                                            SELF := LEFT));
OUTPUT(fieldMap, NAMED('FieldMap'));

F := LT.RegressionForest(numTrees:=numTrees, featuresPerNode:=numFeatures, maxDepth:=maxDepth, nominalFields:=nomFields);
mod := F.GetModel(X, Y);

Y_S := SORT(Y, value);
classCounts0 := TABLE(Y, {wi, class := value, cnt := COUNT(GROUP)}, wi, value);
classCounts := TABLE(classCounts0, {wi, classes := COUNT(GROUP)}, wi);

OUTPUT(mod, NAMED('Model'));
nodes := SORT(F.Model2Nodes(mod), wi, treeId, level, nodeId);
OUTPUT(nodes, {wi, treeId, level, nodeId, parentId, isLeft, number, value, depend, support, ir}, NAMED('TreeNodes'));
modStats := F.GetModelStats(mod);
OUTPUT(modStats, NAMED('ModelStatistics'));
scorecards := DATASET([{1, 'MyScorecardName', '', fieldMap}], LUCI_Scorecard);

luci := LT.LUCI_Export(mod, 'MyModelId', 'MyModelName', scorecards);

OUTPUT(CHOOSEN(luci, 3000), ALL, NAMED('LUCI_Model_single'));
OUTPUT(luci, {line}, 'lucitest.csv', CSV, OVERWRITE, EXPIRE(1), NAMED('LUCI_csv_file_single'));

// Copy the model several times to create a multi-scorecard model
modMult := NORMALIZE(mod, 3, TRANSFORM({mod}, SELF.wi := COUNTER, SELF := LEFT));
sc_mult := DATASET([{1, 'MyScorecard1', 'covtype = 1', fieldMap},
                    {2, 'MyScorecard2', 'covtype = 2', fieldMap},
                    {3, 'MyScorecard3', '(STRING)covtype = \'3\'', fieldMap} // Make sure escaped strings work okay.
                    ], LUCI_Scorecard);
luciMult := LT.LUCI_Export(modMult, 'MyMultModId', 'MyMultModName', sc_mult);

OUTPUT(CHOOSEN(luciMult, 3000), ALL, NAMED('LUCI_Model_multi'));
