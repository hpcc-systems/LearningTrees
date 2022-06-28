/*##############################################################################

    HPCC SYSTEMS software Copyright (C) 2022 HPCC Systems®.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
############################################################################## */

#ONWARNING(2007, ignore);
#ONWARNING(4531, ignore);
#ONWARNING(4550, ignore);

// Modified version of the testCovTypeClass test file that works with the
// OBT test system

IMPORT $.^.test.datasets.CovTypeDS;
IMPORT $.^ AS LT;
IMPORT LT.LT_Types;
IMPORT ML_Core;
IMPORT ML_Core.Types;

numTrees := 100;
maxDepth := 255;
numFeatures := 0; // Zero is automatic choice
balanceClasses := FALSE;
nonSequentialIds := TRUE; // True to renumber ids, numbers and work-items to test
                            // support for non-sequentiality
numWIs := 2;     // The number of independent work-items to create
maxRecs := 5000; // Note that this has to be less than or equal to the number of records
                 // in CovTypeDS (currently 5000)
DependentVar := 52; // Dependent Variable meant for this function

										
DiscreteField := Types.DiscreteField;
NumericField := Types.NumericField;
trainDat := CovTypeDS.trainRecs;
testDat := CovTypeDS.testRecs;
nominalFields := CovTypeDS.nominalCols;

ClassTest() := FUNCTION
	ML_Core.ToField(trainDat, trainNF); // Get training data as a field
	ML_Core.ToField(testDat, testNF); // Get test data as a field
        
	//Ind = independent, Dep = dependent
	Ind1 := PROJECT(trainNF(number != DependentVar AND id <= maxRecs), TRANSFORM(NumericField,
					SELF.number := IF(nonSequentialIds, 5*LEFT.number, LEFT.number),
					SELF.id := IF(nonSequentialIds, 5*LEFT.id, LEFT.id),
					SELF := LEFT));
	Dep1 := PROJECT(trainNF(number = DependentVar AND id <= maxRecs), TRANSFORM(DiscreteField,
					SELF.number := 1,
					SELF.id := IF(nonSequentialIds, 5*LEFT.id, LEFT.id),
					SELF := LEFT));
					
	// Generate multiple work items
	Ind2 := NORMALIZE(Ind1, numWIs, TRANSFORM(RECORDOF(LEFT),
						SELF.wi := IF(nonSequentialIds, 5*COUNTER, COUNTER),
						SELF := LEFT));
	Dep2 := NORMALIZE(Dep1, numWIs, TRANSFORM(RECORDOF(LEFT),
						SELF.wi := IF(nonSequentialIds, 5*COUNTER, COUNTER),
						SELF := LEFT));

	Forest := LT.ClassificationForest(numTrees, numFeatures, maxDepth, nominalFields, balanceClasses);
	model := Forest.GetModel(Ind2, Dep2);

	IndTest1 :=  PROJECT(testNF(number != DependentVar), TRANSFORM(NumericField,
					SELF.number := IF(nonSequentialIds, 5*LEFT.number, LEFT.number),
					SELF.id := IF(nonSequentialIds, 5*LEFT.id, LEFT.id),
					SELF := LEFT));
	DepCmp1 := PROJECT(testNF(number = DependentVar), TRANSFORM(DiscreteField,
					SELF.id := IF(nonSequentialIds, 5*LEFT.id, LEFT.id),
					SELF := LEFT));
					
	// Generate multiple work items
	IndTest2 := NORMALIZE(IndTest1, numWIs, TRANSFORM(RECORDOF(LEFT),
						SELF.wi := IF(nonSequentialIds, 5*COUNTER, COUNTER),
						SELF := LEFT));
	DepCmp2 := NORMALIZE(DepCmp1, numWIs, TRANSFORM(RECORDOF(LEFT),
						SELF.wi := IF(nonSequentialIds, 5*COUNTER, COUNTER),
						SELF.number := 1;
						SELF := LEFT));

	RETURN Forest.Accuracy(model, DepCmp2, IndTest2);
END;
       
accuracy := ClassTest();

// Both work items should be at least 78% accurate
OUTPUT(accuracy, {passing := IF((COUNT(GROUP, raw_accuracy >= 0.78) = numWIs), 'Pass', 'Fail: ' + raw_accuracy + ' < 0.78')}, NAMED('Result'));
