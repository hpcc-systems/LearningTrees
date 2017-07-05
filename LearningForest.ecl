IMPORT LT_Types AS Types;
IMPORT ML_Core;
IMPORT ML_Core.Types as CTypes;
IMPORT $ as LT;
IMPORT LT.internal AS int;

GenField := Types.GenField;
DiscreteField := CTypes.DiscreteField;
NumericField := CTypes.NumericField;
Layout_Model2 := Types.Layout_Model2;
ModelStats := Types.ModelStats;

/**
  * LearningForest
  *
  * This module is the base module for Random Forests.
  * It implements the Random Forest algorithms as described by Breiman, 2001
  * (see https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
  */
  EXPORT LearningForest(UNSIGNED numTrees=100,
              UNSIGNED featuresPerNode=0,
              UNSIGNED maxDepth=255) := MODULE
    SHARED DATASET(GenField) NF2GenField(DATASET(NumericField) ds, SET OF UNSIGNED nominalFields=[]) := FUNCTION
      dsOut := PROJECT(ds, TRANSFORM(GenField, SELF.isOrdinal := LEFT.number NOT IN nominalFields, SELF := LEFT));
      RETURN dsOut;
    END;
    SHARED DATASET(GenField) DF2GenField(DATASET(DiscreteField) ds) := FUNCTION
      dsOut := PROJECT(ds, TRANSFORM(GenField, SELF.isOrdinal := TRUE, SELF := LEFT));
      RETURN dsOut;
    END;
    /**
      * Get information about the model
      *
      * Returns a set of information about the provided model
      *
      * @param mod A model previously returned from GetModel
      * @return A single ModelStats record containing information about the model
      */
    EXPORT ModelStats GetModelStats(DATASET(Layout_Model2) mod) := FUNCTION
      myRF := int.RF_Classification();  // Could use Regression of Classification.  Same result.
      RETURN myRF.GetModelStats(mod);
    END;
  END;
